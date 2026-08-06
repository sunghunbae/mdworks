from pathlib import Path
from io import StringIO
from importlib.metadata import version

from pdbfixer import PDBFixer  
from pdb2pqr.main import main_driver, build_main_parser

import shutil
import logging

from .editor import Editor
from .utils import setup_logger


logger = logging.getLogger(__name__)


       
class ReadyPipeline:
    def __init__(self, 
                 in_file: str | Path,
                 ligand_resname: str | None = None,
                 remove_altloc: bool = True,
                 keep_waters: bool = False,
                 separate_hetgens: bool = False,
                 zinc: bool = True,
                 keep_terminals: bool = True,
                 obabel: str | None = shutil.which("obabel"),
                 target_ph: float = 7.4,         
                 workdir: Path | str | None = None, 
                 quiet: bool = False):
        assert isinstance(in_file, str) or isinstance(in_file, Path)
        in_path = Path(in_file)
        assert in_path.exists()

        self.ligand_resname : str = ligand_resname
        self.remove_altloc : bool = remove_altloc
        self.keep_waters : bool = keep_waters
        self.separate_hetgens : bool = separate_hetgens
        self.zinc : bool = zinc
        self.target_ph : float = target_ph
        self.keep_terminals : bool = keep_terminals
        self.obabel = obabel

        # setup prefix and workdir
        # remove all extensions and get the true stem: ex. x.cif.gz -> x 
        self.prefix : str = in_path.name.removesuffix("".join(in_path.suffixes))
        if isinstance(workdir, str) or isinstance(workdir, Path):
            self.workdir = Path(workdir)
            self.workdir.mkdir(exist_ok=True)
        else:
            self.workdir = in_path.parent
        self.quiet = quiet

        logging.getLogger().handlers.clear()
        setup_logger(logger, self.workdir, self.prefix, quiet=self.quiet)
        
        logger.info(f"mdworks {version('mdworks')}")
        logger.info(f"pdbfixer {version('pdbfixer')}")
        logger.info(f"rdkit {version('rdkit')}")
        logger.info(f"workdir= {self.workdir}")
        logger.info(f"prefix= {self.prefix}")

        self.editor = Editor.load(str(in_path))
        self.ligand = None
        self.receptor = None
        self.zn_cys = set()
        self.fixer = None

    
    def run(self):
        self.editor = self.editor.standardize_chain_id()
        # _atom_site.auth_asym_id is changed to standard chain id,
        # but _atom_site.label_asym_id is not changed, so we need to update it to match the new chain id.
        # st = st.update_label_asym_id()

        if self.remove_altloc:
            self.editor = self.editor.remove_alternative_conformations()

        if not self.keep_waters:
            self.editor = self.editor.remove_waters()

        if self.separate_hetgens:
            self.editor = self.editor.new_chains_for_non_std_residues()

        if self.zinc:
            self.zn_cys = self.editor.find_zn_coord_cys(model_idx=0)

        if self.ligand_resname and self.obabel:
            logger.info(f"Step 1 - Gemmi extracting ligand {self.ligand_resname} ...")
            self.editor = self.editor.select(expr=self.ligand_resname)
            self.ligand = self.editor.extract().set_as_ligand(resname=self.ligand_resname)
            self.receptor = self.editor.delete()
        else:
            self.receptor = self.editor
            
        self._pdbfixer()
        self._pdb2pqr()

        if self.ligand:
            receptor_pdb = f"{self.prefix}_H.pdb" # pdb2pqr generated receptor
            complex_cif  = f"{self.prefix}_complex.cif"
            logger.info(f"Step 4 - Gemmi merging receptor and ligand {self.ligand_resname} ...")
            Editor.load(receptor_pdb).merge(self.ligand).write(complex_cif)


    def _pdbfixer(self) -> None:
        logger.info(f"Step 2 - PDBFixer fixing receptor ...")
        cif_string = self.receptor.to_mmcif_str()
        self.fixer = PDBFixer(pdbxfile= StringIO(cif_string))
        
        # PDBFixer uses geometry template to fill in missing residues and atoms, 
        # and to replace nonstandard residues with standard ones.
        self.fixer.findMissingResidues()
        self.fixer.findNonstandardResidues()
        self.fixer.replaceNonstandardResidues()
        self.fixer.findMissingAtoms()
        
        # By default, when you run fixer.findMissingAtoms(), PDBFixer automatically 
        # calculates proximity between all cysteine sulfur atoms. 
        # If a cysteine has multiple partners within a standard distance cutoff, 
        # it logs the warning and picks one, which can lead to incorrect topology generation.
        # The missingResidues dictionary stores missing residues as: 
        # (chainIndex, residueIndex): [list of residue names]
        # example:
        # {
        #   (0, 0): ['MET', 'GLU', ..., 'PRO', 'SER']), 
        #   (0, 108): ['PRO', 'VAL', ... , 'VAL'], 
        #   (0, 232): ['VAL', ..., 'ARG', 'LEU']
        # }

        if self.keep_terminals:
            self._do_not_add_missing_atoms_at_terminals()
            logger.info(f"  Skipping to add missing atoms at terminals")

        if self.zn_cys:
            logger.info(f"  Stripping spurious S-S bonds around a zinc")
            self._remove_Zn_bonds()
            self._strip_spurious_SS_bonds()
            
        # Add missing heavy atoms (but do not add hydrogens yet; PDB2PQR will do that)
        self.fixer.addMissingAtoms()

        fixed_receptor_pdb = f"{self.prefix}_X.pdb"
        self.editor.write(fixed_receptor_pdb)
        logger.info(f"  PDBFixer fixed heavy atoms and saved to {fixed_receptor_pdb}")

        
    def _pdb2pqr(self) -> None:
        logger.info(f"Step 3 - PDB2PQR protonating with target pH {self.target_ph} ...")
        # PDB2PQR automatically removes ligand if it is present in the input PDB file.
        fixed_receptor_pdb      = f"{self.prefix}_X.pdb"
        protonated_receptor_pqr = f"{self.prefix}_H.pqr"
        protonated_receptor_pdb = f"{self.prefix}_H.pdb"    

        pdb2pqr_args = [
            "--ff=AMBER", 
            f"--with-ph={self.target_ph}",
            f"--pdb-output={protonated_receptor_pdb}",
            fixed_receptor_pdb,
            protonated_receptor_pqr]

        parser = build_main_parser()
        parsed_pdb2pqr_args = parser.parse_args(pdb2pqr_args)
        main_driver(parsed_pdb2pqr_args)
        logger.info(f"  PDB2PQR saved protonated receptor to {protonated_receptor_pdb}")

        
    def _do_not_add_missing_atoms_at_terminals(self) -> None:
        chains = list(self.fixer.topology.chains())
        residues = list(self.fixer.topology.residues())
        skipped_missing_residues = {}

        for key, resnames in sorted(self.fixer.missingResidues.items()):
            (chain_idx, res_idx) = key
            # residues will be inserted at res_idx in chain_idx
            chain = chains[chain_idx]
            residue = residues[res_idx]
            n = len(resnames)
            at = int(residue.id)
            n_present = len(list(chain.residues()))
            is_n_terminal = res_idx == 0
            is_c_terminal = res_idx == n_present
            if is_n_terminal or is_c_terminal:
                skipped_missing_residues[key] = self.fixer.missingResidues.pop(key)
                if is_n_terminal:
                    logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (N-ter; skipped)")
                if is_c_terminal:
                    logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (C-ter; skipped)")
            else:
                logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}..")


    def _remove_Zn_bonds(self) -> None:
        # # Track down the Zinc atom index
        zinc_atom_indices = [atom.index for atom in self.fixer.topology.atoms() if "ZN" in atom.name.upper()]
        # Rebuild the bond network, explicitly excluding any bonds involving the Zinc atom
        clean_bonds = []
        for bond in self.fixer.topology.bonds():
            i, j = bond[0], bond[1]
            if i.index in zinc_atom_indices or j.index in zinc_atom_indices:
                logger.info(f"  Break Zn bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
                continue
            clean_bonds.append(bond)
        self.fixer.topology._bonds = clean_bonds


    def _strip_spurious_SS_bonds(self) -> None:
        """Strip any spurious disulfide bond (SG-SG) created between two zinc-coordinating cysteines

        Note:
            Real disulfid bonds between two non-zinc coordinating cysteines are left untouched.
            Keeping Zn-CYS bond is recommended because it stabilizes the structure during MD simulation.

        Args:
            fixer (PDBFixer): PDBFixer class instance
            zinc_cys (set): output from `find_zn_coord_cys()`, {(chain_id, resseq), ...}
        """
        kept_bonds = []
        for bond in self.fixer.topology.bonds():
            i, j = bond[0], bond[1]
            is_i_zn = i.name == 'ZN'
            is_i_zn_cys = i.name == 'SG' and i.residue.name == 'CYS' and (i.residue.chain.id, int(i.residue.id)) in self.zn_cys
            is_j_zn = j.name == 'ZN'
            is_j_zn_cys = j.name == 'SG' and j.residue.name == 'CYS' and (j.residue.chain.id, int(j.residue.id)) in self.zn_cys
            
            # handle CYS-CYS
            if is_i_zn_cys and is_j_zn_cys:
                logger.info(f"  Break S-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
                continue

            # inform Zn-CYS
            if (is_i_zn and is_j_zn_cys) or (is_i_zn_cys and is_j_zn):
                logger.info(f"  Zn-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")

            kept_bonds.append(bond)
        self.fixer.topology._bonds = kept_bonds
        # rename zinc-coordinating cysteines: CYS to CYM
        # for res in zn_cys_residues:
        #     res.name = 'CYM'
