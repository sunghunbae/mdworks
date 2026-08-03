from pathlib import Path
from collections import defaultdict
from io import StringIO

from pdbfixer import PDBFixer  
from openmm.app import PDBFile, PDBxFile, Topology, Modeller
from openmm.unit import angstroms, nanometers
from pdb2pqr.main import main_driver, build_main_parser

import re
import shutil
import logging


from .editor import Editor
from .utils import setup_logger


logger = logging.getLogger(__name__)


def remove_Zn_bonds(fixer: PDBFixer) -> None:
    # # Track down the Zinc atom index
    zinc_atom_indices = [atom.index for atom in fixer.topology.atoms() if "ZN" in atom.name.upper()]
    # Rebuild the bond network, explicitly excluding any bonds involving the Zinc atom
    clean_bonds = []
    for bond in fixer.topology.bonds():
        i, j = bond[0], bond[1]
        if i.index in zinc_atom_indices or j.index in zinc_atom_indices:
            logger.info(f"  Break Zn bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
            continue
        clean_bonds.append(bond)
    fixer.topology._bonds = clean_bonds


def strip_spurious_SS_bonds(fixer: PDBFixer, zn_cys: set) -> None:
    """Strip any spurious disulfide bond (SG-SG) created between two zinc-coordinating cysteines

    Note:
        Real disulfid bonds between two non-zinc coordinating cysteines are left untouched.
        Keeping Zn-CYS bond is recommended because it stabilizes the structure during MD simulation.

    Args:
        fixer (PDBFixer): PDBFixer class instance
        zinc_cys (set): output from `find_zn_coord_cys()`, {(chain_id, resseq), ...}
    """
    kept_bonds = []
    for bond in fixer.topology.bonds():
        i, j = bond[0], bond[1]
        is_i_zn = i.name == 'ZN'
        is_i_zn_cys = i.name == 'SG' and i.residue.name == 'CYS' and (i.residue.chain.id, int(i.residue.id)) in zn_cys
        is_j_zn = j.name == 'ZN'
        is_j_zn_cys = j.name == 'SG' and j.residue.name == 'CYS' and (j.residue.chain.id, int(j.residue.id)) in zn_cys
        
        # handle CYS-CYS
        if is_i_zn_cys and is_j_zn_cys:
            logger.info(f"  Break S-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
            continue

        # inform Zn-CYS
        if (is_i_zn and is_j_zn_cys) or (is_i_zn_cys and is_j_zn):
            logger.info(f"  Zn-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")

        kept_bonds.append(bond)
    fixer.topology._bonds = kept_bonds
    # rename zinc-coordinating cysteines: CYS to CYM
    # for res in zn_cys_residues:
    #     res.name = 'CYM'


def do_not_add_missing_atoms_at_terminals(fixer: PDBFixer) -> None:
    chains = list(fixer.topology.chains())
    residues = list(fixer.topology.residues())
    skipped_missing_residues = {}

    for key, resnames in sorted(fixer.missingResidues.items()):
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
            skipped_missing_residues[key] = fixer.missingResidues.pop(key)
            if is_n_terminal:
                logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (N-ter; skipped)")
            if is_c_terminal:
                logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (C-ter; skipped)")
        else:
            logger.info(f"missing {n:<2d} residues at {chain.id}:{at:<4d} {','.join(resnames[:3])}..")

       

def complex(
        filename: str | None = None, 
        ligand_resname: str | None = None,
        waters: bool = False,
        separate_hetgens: bool = False,
        zinc: bool = True,
        terminals: bool = False,
        obabel: str | None = shutil.which("obabel"),
        target_ph: float = 7.4,
        quiet: bool = False) -> None:  
    """  
    Fix complex/receptor structural issues and set protonation states.
    PDBFixer/PDB2PQR workflow excludes non-standard residus including ligands, cofactors, and water molecules.
    So, if the receptor structure contains a ligand, it should be extracted and processed separately.
    """  
    p = Path(filename)
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    workdir = p.parent

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    fixed_receptor_pdb = f"{output_prefix}_noH.pdb"
    protonated_receptor_pqr = f"{output_prefix}_H.pqr"  
    protonated_receptor_pdb = f"{output_prefix}_H.pdb"
    protonated_complex_cif = f"{output_prefix}_complex.cif"

    st = Editor.load(filename)
    ligand = None

    st = st.standardize_chain_id()
    # _atom_site.auth_asym_id is changed to standard chain id,
    # but _atom_site.label_asym_id is not changed, so we need to update it to match the new chain id.
    # st = st.update_label_asym_id()

    if not waters:
        st = st.remove_waters()

    if separate_hetgens:
        st = st.new_chains_for_non_std_residues()

    if zinc:
        zn_cys = st.find_zn_coord_cys(model_idx=0)
    else:
        zn_cys = set()

    if ligand_resname and obabel:
        logger.info(f"[Step 0] Extracting Ligand {ligand_resname} ...")
        # ligand_pdb = f"{output_prefix}_{ligand_resname}.pdb"
        # ligand_smi = f"{output_prefix}_{ligand_resname}.smi"
        st = st.select(expr=ligand_resname)
        ligand = st.extract().set_as_ligand(resname=ligand_resname)
        # structure may have multiple ligand molecules, and they will be saved in a single PDB file
        # ligand.write(ligand_pdb)
        # logger.info(f"Extracted ligand {ligand_resname} saved to {ligand_pdb}")
        # Editor.pdb_to_smiles(pdbfile=ligand_pdb, smifile=ligand_smi, obabel=obabel)
        # if multiple ligand molecules are present, the SMILES file will contain multiple structures
        # separated by period symbols
        # logger.info(f"Extracted ligand {ligand_resname} saved to {ligand_smi}")
        # Remove ligand from the structure to be fixed, so that PDBFixer/PDB2PQR will not remove it
        receptor = st.delete()
    else:
        receptor = st
        
    logger.info(f"[Step 1] PDBFixer fixing ...")

    cif_string = receptor.to_mmcif_str()
    fixer = PDBFixer(pdbxfile= StringIO(cif_string))
    
    # PDBFixer uses geometry template to fill in missing residues and atoms, 
    # and to replace nonstandard residues with standard ones.
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
      
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

    if not terminals:
        do_not_add_missing_atoms_at_terminals(fixer)

    if zn_cys:
        logger.info(f"Stripping spurious S-S bonds around a zinc")
        remove_Zn_bonds(fixer)
        strip_spurious_SS_bonds(fixer, zn_cys)
        
    # Add missing heavy atoms (but do not add hydrogens yet; PDB2PQR will do that)
    fixer.addMissingAtoms()

    # Write fixed heavy-atom receptor structure
    with open(fixed_receptor_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)  
        logger.info(f"PDBFixer fixed heavy atoms and saved to {fixed_receptor_pdb}")

    logger.info(f"[Step 2] PDB2PQR protonating with target pH {target_ph} ...")
    # PDB2PQR automatically removes ligand if it is present in the input PDB file.
    pdb2pqr_args = [
        "--ff=AMBER", 
        f"--with-ph={target_ph}",
        f"--pdb-output={protonated_receptor_pdb}",
        fixed_receptor_pdb,
        protonated_receptor_pqr]
    parser = build_main_parser()
    parsed_pdb2pqr_args = parser.parse_args(pdb2pqr_args)
    main_driver(parsed_pdb2pqr_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR saved protonated receptor to {protonated_receptor_pdb}")

    if ligand:
        logger.info(f"[Step 3] Merging protonated receptor and original ligand {ligand_resname} ...")
        Editor.load(protonated_receptor_pdb).merge(ligand).write(protonated_complex_cif)
        logger.info(f"Merged complex saved to {protonated_complex_cif}")