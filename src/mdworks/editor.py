from __future__ import annotations

__all__ = ['Editor',]

from pathlib import Path
from typing import Iterable, Sequence
from collections import defaultdict

import re
import gzip
import shutil
import subprocess
import string
import logging
import numpy as np
import gemmi

from rdkit import Chem
from rdkit.Chem import AllChem

from .utils import setup_logger
logger = logging.getLogger(__name__)


class LigandFixer:
    def __init__(self, 
                 max_displacement: float = 0.5,
                 k: float = 1000.0,
                 max_iter: int = 500,
                 ):
        """Fix small molecule ligand.

        Args:
            max_displacement (float, optional): maximum displacement during optimization. Defaults to 0.5.
            k (float, optional): force constant for positional restraints (kJ/mol/A**2). Defaults to 1000.0.
            max_iter (int, optional): maximum number of iteration. Defaults to 500.
        """
        self.prefix : str = ''
        self.mol_resname : str = ''
        self.mol_smiles : str = ''
        self.mol_pdb_block : str = ''
        self.mol_source : Chem.Mol | None = None
        self.mol_target : Chem.Mol | None = None
        self.mol : Chem.Mol | None = None
        # optimization
        self.max_displacement: float = max_displacement
        self.k: float = k
        self.max_iter: int = max_iter
         

    def _ligand_convert_pdb_to_smiles(self, obabel: str = shutil.which("obabel")) -> None:
        pdbfile = f"{self.prefix}_{self.mol_resname}.pdb"
        smifile = f"{self.prefix}_{self.mol_resname}.smi"
        assert self.mol_pdb_block, "Molecule PDB block is not set"
        try:
            with open(pdbfile, 'w') as f:
                f.write(self.mol_pdb_block)

            result = subprocess.run([obabel, "-ipdb", pdbfile, "-osmi"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True
                                    )
            output = result.stdout.strip()
            if output:
                smiles, name = output.split(maxsplit=1) # ex. <SMILES> <Name>
                with open(smifile, "w") as f:
                    f.write(f"{smiles}\n")

                self.mol_smiles = smiles
                self.mol_target = Chem.MolFromSmiles(smiles)
                self.mol_source = Chem.MolFromPDBBlock(self.mol_pdb_block, removeHs=True, sanitize=False)

            else:
                raise ValueError(f"Openbabel returned no SMILES from {pdbfile}.")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Openbabel failed on {pdbfile}: {e}")
        

    
    def _ligand_create_atom_map(self) -> dict:
        """Create atom map between target and source molecule based on connectivity only."""
        assert self.mol_source is not None, "Source molecule is not set"
        assert self.mol_target is not None, "Target molecule is not set"

        # Use SMARTS with any bonds (~)
        for b in self.mol_target.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        
        smarts = Chem.MolToSmarts(self.mol_target).replace("-", "~")
        query = Chem.MolFromSmarts(smarts)
        
        match = self.mol_source.GetSubstructMatch(query)
        if not match:
            raise ValueError("No connectivity match found")
            
        return dict(enumerate(match))
    
    
    def _ligand_copy_positions(self) -> None:
        """Import 3D coordinates from source molecule."""
        atom_map = self._ligand_create_atom_map()

        target = Chem.RWMol(self.mol_target) # copy
        # Ensure the destination molecule has a writable conformer (add one if necessary)
        # The default behavior when setting positions is to add a conformer if none exists
        conf = target.GetConformer(0) if target.GetNumConformers() > 0 else Chem.Conformer(target.GetNumAtoms())
        
        # Iterate over the map numbers and copy positions
        for target_idx, source_idx in atom_map.items():
            # Get the position from the source conformer
            pos = self.mol_source.GetConformer(0).GetAtomPosition(source_idx)
            # Set the position in the destination conformer
            conf.SetAtomPosition(target_idx, pos)
                
        # Add the conformer back to the molecule if a new one was created
        if target.GetNumConformers() == 0:
            target.AddConformer(conf, assignId=True)

        self.mol = target

            
    def _ligand_sync_positions(self) -> None:
        """Synchronize self.mol atomic positions with gemmi structure Atom() positions"""
        individual_mols = list(Chem.GetMolFrags(self.mol, asMols=True))
        num_residues = sum([1 for chain in self.model for residue in chain])
        assert num_residues == len(individual_mols), "number of residues do not match"
        
        individual_residues = [] # list of list

        for _rdmol in individual_mols:
            _conformer = _rdmol.GetConformer()
            _residue_atoms = []
            for i, rd_atom in enumerate(_rdmol.GetAtoms()):
                _atom = gemmi.Atom()
                _atom.name = rd_atom.GetSymbol() + str(i+1)
                _atom.element = gemmi.Element(rd_atom.GetSymbol())
                _atom.charge = rd_atom.GetFormalCharge()
                # Set spatial properties
                pos = _conformer.GetAtomPosition(i)
                _atom.pos = gemmi.Position(pos.x, pos.y, pos.z)               
                # Set standard crystallographic defaults
                _atom.occ = 1.0
                _atom.b_iso = 20.0
                # Append atom up through the Gemmi hierarchy
                _residue_atoms.append(_atom)
            individual_residues.append(_residue_atoms)

        for chain in self.model:
            for residue in chain:
                logger.info(f"synchronizing {residue.name} {residue.seqid.num} {chain.name}")
                # clear all atoms
                while len(residue) > 0:
                    del residue[0]
                # re-populate atoms
                _residue_atoms = individual_residues.pop(0)
                for _atom in _residue_atoms:
                    residue.add_atom(_atom)


    def _ligand_optimize(self, save: bool = True) -> None:
        """Optimize the molecule using MMFF94 with positional restraints."""
        # adding hydrogens
        mol = Chem.AddHs(self.mol, addCoords=True)
        
        conf = mol.GetConformer()
        original_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

        # Get MMFF properties
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties for molecule")
    
        # Create force field
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
        if ff is None:
            raise ValueError("Could not create MMFF force field")

        # Add positional restraints
        restraint_count = 0
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() > 1:
                ff.MMFFAddPositionConstraint(i, self.max_displacement, self.k)
                restraint_count += 1
    
        # Optimize
        initial_energy = ff.CalcEnergy()
        converged = ff.Minimize(maxIts=self.max_iter)
        final_energy = ff.CalcEnergy()
    
        # Calculate RMSD
        optimized_coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        rmsd = np.sqrt(np.mean(np.sum((original_coords - optimized_coords)**2, axis=1)))
        logger.info(f"ligand optimized with MMFF:")
        logger.info(f"  positional restraints on {restraint_count} atoms with k= {self.k} kJ/mol/A^2")
        logger.info(f"  energy initial: {initial_energy:.2f} kcal/mol")
        logger.info(f"  energy final: {final_energy:.2f} kcal/mol")
        logger.info(f"  rmsd from original: {rmsd:.3f} Å")
        
        self.mol = mol
        self._ligand_sync_positions()

        if save:
            with Chem.SDWriter(f'{self.prefix}_{self.mol_resname}.sdf') as w:
                for m in Chem.GetMolFrags(self.mol, asMols=True):
                    w.write(m)




class Editor(LigandFixer):
    """Load once, chain edits, write once.
    A thin, chainable wrapper around gemmi.Structure for the structure-editing
    tasks that come up repeatedly when preparing PDB/mmCIF files for MD setup:
    reordering chains, deleting chains, deleting residues, selecting subsets,
    and renumbering. Gemmi does the heavy lifting (parsing, format handling,
    PDB/mmCIF round-tripping); this class just gives you a task-oriented API
    on top of it so pipeline code reads like a sequence of intentions rather
    than gemmi object-model plumbing.

    Example
    -------
    >>> (
    ...     Editor.load("input.cif")
    ...     .remove_chains(["C"])
    ...     .remove_residues("A", [45, 46, 47])
    ...     .reorder_chains(["B", "A"])
    ...     .write("output.pdb")
    ... )
    """

    std_chain_ids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

    std_residues_protein = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY",
        "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
        "THR","TRP","TYR","VAL",
        }
            
    std_residues_solvent = {
        "HOH","NA", "K", "CL",
        }
    
    std_residues_divalent_ion = {
        "MG" , "ZN", "CA", "MN", "FE", "CU", "CO", "CD", "NI", "SR", "BA", 
        }
    

    def __init__(self, 
                 structure: gemmi.Structure | None = None,
                 model_index: int = 0,
                 prefix: str = '',
                 quiet: bool = False):
        super().__init__()
        self.structure = structure
        self.model_index = model_index
        self.prefix = prefix
        self.quiet = quiet
        self.sel = gemmi.Selection()
        self.is_ligand : bool = False
        workdir = Path('.')
        # if structure is ligand, we can generate SMILES and optimize geometry with RDKit

        logging.getLogger().handlers.clear()
        setup_logger(logger, workdir, self.prefix, quiet=quiet)
        
    # ------------------------------------------------------------------ #
    # Construction / IO
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path, model_index: int = 0, quiet: bool = False) -> "Editor":
        """Read a PDB, mmCIF, or mmJSON file (format inferred from contents)."""
        p = Path(path)
        structure = gemmi.read_structure(str(p))
        structure.setup_entities()
        """
        Entity
        ======

        Entity is a new concept introduced in the mmCIF format -a chemically distinct part, such as polymer, 
        ligand, ion or water. Ligands with the same residue name correspond to the same entity. 
        Polymers that have the same sequence — the same entity. In the mmCIF format entities are explicitly 
        linked with structural units that we call here subchains. PDB files do not have this concept. 
        If we read the structure from a PDB file, we can assign entities by calling setup_entities. 
        This method uses a heuristic to group residues into subchains, which are then mapped to entities.
        Internally, setup_entities() runs four functions (in this order):
            add_entity_types()
            add_subchains()
            ensure_entities()
            deduplicate_entities()
        """

        prefix =  p.name.removesuffix("".join(p.suffixes))
        return cls(structure, model_index= model_index, prefix=prefix, quiet=quiet)

    
    def show_entity(self) -> None:
        for e in self.structure.entities:
            logger.info(f"entity {e.name} ({e.entity_type}) {e.polymer_type} {e.full_sequence[:5]}...")
            for subchain in e.subchains:
                logger.info(f"  subchain {subchain}")


    def remove_altloc(self) -> "Editor":
        # This mutates the structure object, keeping 'A' (or first) 
        # and setting all remaining altloc characters to blank
        self.structure.remove_alternative_conformations()
        return self


    def write(self, 
              path: str | Path | None = None, 
              minimal: bool = False,
              format: str = 'cif', # cif | pdb
              tag: str = '', 
              split: bool = False,
              compress: bool = True) -> "Editor":
        """Write structure to a file"""
        if path is None:
            if compress:
                outfile_path = Path(f'{self.prefix}_{tag}.{format}.gz')
            else:
                outfile_path = Path(f'{self.prefix}_{tag}.{format}')

        else:
            outfile_path = Path(path)
            format = 'pdb' if 'pdb' in outfile_path.name else 'cif'
            compress = True if outfile_path.name.endswith('.gz') else False

        outfile =  str(outfile_path)

        if not split:
            if format in ('cif', 'mmcif'):
                doc = self.structure.make_mmcif_document() if not minimal \
                    else self.structure.make_mmcif_headers()
                if compress:
                    with gzip.open(outfile, "wt", encoding="utf-8") as f:
                        f.write(doc.as_string())
                else:
                    doc.write_file(outfile)
            elif format == 'pdb':
                if compress:
                    with gzip.open(outfile, "wt", encoding="utf-8") as f:
                        f.write(self.structure.make_pdb_string())
                else:
                    self.structure.write_pdb(outfile)                
            logger.info(f"write to {outfile}")
        else:
            # Note: tag is ignored
            for model_idx, model in enumerate(self.structure, start=1):
                outfile_path = Path(f"{self.prefix}_{model_idx}.{format}")
                outfile = str(outfile_path)

                single_model_st = gemmi.Structure()   

                # Preserve original metadata if desired (e.g., cell, spacegroup)
                try:
                    single_model_st.cell = self.cell
                except:
                    pass
                try:
                    single_model_st.spacegroup_name = self.spacegroup_name
                except:
                    pass
                
                # Add a copy of the current model to the new structure
                # (Using .clone() prevents altering or corrupting the source object)
                single_model_st.add_model(model.clone())

                if format in  ("cif", "mmcif"):
                    doc = single_model_st.structure.make_mmcif_document() if not minimal \
                        else single_model_st.structure.make_mmcif_headers()
                    if compress:
                        with gzip.open(outfile, "wt", encoding="utf-8") as f:
                            f.write(doc.as_string())
                    else:
                        doc.write_file(outfile)
                elif format == 'pdb':
                    if compress:
                        with gzip.open(outfile, "wt", encoding="utf-8") as f:
                            f.write(doc.as_string())
                    else:
                        single_model_st.structure.write_pdb(outfile)

                logger.info(f"write to {outfile}")

        return self
        

    def clone(self) -> "Editor":
        """Return an independent deep copy, useful before a destructive edit
        when you still need the original in memory."""
        return Editor(self.structure.clone(), self.model_index)


    def select(self, expr: str) -> "Editor":
            """Parse Ambertools style chain/residue selection expressions and 
            generate gemmi.Selection filter.
    
            Coordinate ID (or CID):
                https://gemmi.readthedocs.io/en/stable/analysis.html#selections-cid
    
            Args:
                expr (str): example - "A:10-30,A:100-120,B:1-50,C:1,D,UNL"
            """
            pattern = r"^(?:(?P<chain>[A-Za-z0-9-]+):)?(?P<resname_or_resseq>[A-Za-z0-9]+)(?:-(?P<resseq_end>[0-9]+))?$"
    
            self.sel = gemmi.Selection()
            self._reset_residue_flag()
    
            selections = []
            for sub_expr in expr.split(","):
                match = re.match(pattern, sub_expr)
                if match:
                    spec = match.groupdict()
                    chain_id = spec.get('chain')
                    if chain_id is None:
                        chain_id = '*'
                    res_start = spec.get('resname_or_resseq')
                    res_end = spec.get('resseq_end')
                    if res_start.isdigit():
                        if res_end and res_end.isdigit():
                            cid = f'//{chain_id}/{res_start}-{res_end}'
                        else:
                            cid = f'//{chain_id}/{res_start}'    
                    else:
                        cid = f'//{chain_id}/({res_start})'
                    selections.append(gemmi.Selection(cid))
            
            flag = 'x'
            for sel in selections:
                for chain in sel.chains(self.model):
                    for residue in sel.residues(chain):
                        residue.flag = flag
    
            self.sel = gemmi.Selection().set_residue_flags(flag)
            atom_count = self.model.count_atom_sites(self.sel)
    
            logger.info(f"select residues: {self._count_sel()} ({atom_count} atoms)")
            
            return self


    def extract(self) -> "Editor":
        return Editor(self.sel.copy_structure_selection(self.structure), 
                      self.model_index,
                      self.prefix,
                      self.quiet)
    

    def delete(self, invert: bool = False) -> "Editor":
        if invert:
            self.sel.remove_not_selected(self.model)
            logger.info("remove NOT selected")
        else:
            self.sel.remove_selected(self.model)
            logger.info("remove selected")

        # clear old metadata associations
        self.structure.entities.clear()

        # force gemmi to recalculate entities from current atoms
        self.structure.setup_entities()

        return self


    def merge(self, other: Editor) -> "Editor":
        for chain in other.model:
            self.model.add_chain(chain.clone())
        return self
    

    def to_mmcif_str(self) -> str:
        cif_doc = self.structure.make_mmcif_document()
        cif_string = cif_doc.as_string()
        return cif_string


    def to_pdb_str(self) -> str:
        pdb_string = self.structure.make_pdb_string()
        return pdb_string

        
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> gemmi.Model:
        return self.structure[self.model_index]


    def chain_names(self) -> list[str]:
        return [c.name for c in self.model]


    def update_label_asym_id(self) -> "Editor":
        """Update _atom_site.label_asym_id to match _atom_site.auth_asym_id (chain id)"""
        for model in self.structure:
            for chain in model:
                # Loop over residues to update their assigned subchain (label_asym_id)
                for residue in chain:
                    # Overwrite the subchain label with the author's chain name
                    residue.subchain = chain.name

        return self

    
    def standardize_chain_id(self) -> "Editor":
        """Standardize chain IDs to uppercase letters, lowercase letters, and digits."""
        used_ids = set(self.chain_names())
        for chain in self.model:
            if chain.name not in self.std_chain_ids:
                # Find the next available standard chain id and rename the chain
                for new_id in self.std_chain_ids:
                    if new_id not in used_ids:
                        logger.info(f"standardizing chain {chain.name} to {new_id}")
                        chain.name = new_id
                        used_ids.add(new_id)
                        for residue in chain:
                            residue.subchain = new_id  # Update subchain label to match new chain id
                        break
        return self


    def _get_chain(self, chain_id: str) -> gemmi.Chain:
        chain = self.model.find_chain(chain_id)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_id}' not found. Available: {self.chain_names()}"
            )
        return chain


    def _get_residue(self, chain_id: str, seqid: str) -> gemmi.Residue:
        return self.model.sole_residue(chain_id, gemmi.SeqId(str(seqid)))


    @staticmethod
    def _chain_id_order(char):
        if char.isupper():
            return (0, char)  # Highest priority (0)
        elif char.islower():
            return (1, char)  # Medium priority (1)
        elif char.isdigit():
            return (2, char)  # Lowest priority (2)
        else:
            return (3, char)  # Fallback for symbols/punctuation


    def _get_unused_chain_ids(self) -> list[str]:
        chain_ids = set(self.chain_names())
        return sorted(list(set(self.std_chain_ids) - chain_ids), key=Editor._chain_id_order)


    def _reset_residue_flag(self):
        # Reset flags first — they persist across runs/selections.
        for chain in self.model:
            for residue in chain:
                residue.flag = '\0'


    def _count_sel(self) -> dict:
        c = {}
        for chain in self.sel.chains(self.model):
            c[chain.name] = 0
            for residue in self.sel.residues(chain):
                c[chain.name] += 1

        return {k:v for k, v in c.items() if v > 0}


    def _merge_sel(self, other_sel: gemmi.Selection, flag: chr = 'x') -> gemmi.Selection:
        """
        Merge other Selection with self.sel gemmi.Selection object
        """
        for sel in (self.sel, other_sel):
            for chain in sel.chains(self.model):
                for residue in sel.residues(chain):
                    residue.flag = flag
                    print(residue)

        return gemmi.Selection().set_residue_flags(flag)
    

    def new_chains_for_non_std_residues(self) -> "Editor":
        """
        Moves specified non-standard residues out of their original chains 
        and puts them all into a brand new chain with a unique ID.
        """
        ununsed_chain_ids = self._get_unused_chain_ids()
        for model_idx, model in enumerate(self.structure, start=1):
            new_chains = defaultdict(list)
            for chain in model:
                residues = [f"{res.name:<3} {res.seqid.num}" for res in chain]
                n = len(residues)
                residues_to_delete = []
                for res_idx, residue in enumerate(chain):
                    # look up the chemical component details in Gemmi's table
                    chem_comp = gemmi.find_tabulated_residue(residue.name)
                    # check if the residue name is missing or explicitly non-standard
                    if n == 1 or chem_comp is None or not chem_comp.is_standard():
                        # Identify if it is a modified polymer or a ligand block
                        w = ununsed_chain_ids.pop(0)
                        new_chains[w].append(residue.clone())
                        residues_to_delete.append(res_idx)
                        logger.info(f"assign a new chain id {w} to {chain.name} {residue.name:<3} {residue.seqid.num}")
                # remove residue(s)
                for res_idx in sorted(residues_to_delete, reverse=True):
                    del chain[res_idx] # delete residues from the original chain backward
            # add new chain(s)
            for new_chain_id, residues in sorted(new_chains.items()):
                if residues:
                    new_chain = gemmi.Chain(new_chain_id)
                    for res in residues:
                        new_chain.add_residue(res)
                    model.add_chain(new_chain)

        return self


    # ------------------------------------------------------------------ #
    # Chain-level operations
    # ------------------------------------------------------------------ #

    def remove_chains(self, chain_ids: Iterable[str]) -> "Editor":
        """Delete one or more chains by ID.

        Note: gemmi's underlying remove_chain() silently no-ops on an
        unknown chain ID, so we validate up front and raise instead.
        """
        model = self.model
        available = set(self.chain_names())
        chain_ids = list(chain_ids)
        missing = [cid for cid in chain_ids if cid not in available]
        if missing:
            raise KeyError(
                f"Chain(s) {missing} not found. Available: {sorted(available)}"
            )
        for cid in chain_ids:
            model.remove_chain(cid)
        return self

    def keep_chains(self, chain_ids: Iterable[str]) -> "Editor":
        """Keep only the listed chains, dropping everything else."""
        keep = set(chain_ids)
        drop = [c for c in self.chain_names() if c not in keep]
        return self.remove_chains(drop)


    def reorder_chains(self, order: Sequence[str] | None = None) -> "Editor":
        """Reorder chains in the model. `order` must be a permutation of
        the existing chain IDs (use keep_chains/remove_chains first if you
        also want to drop some)."""
        model = self.model
        current = self.chain_names()
        if not order:
            order = sorted(list(current))
        if set(order) != set(current):
            raise ValueError(
                f"reorder_chains requires a full permutation of {sorted(current)}, "
                f"got {sorted(order)}"
            )
        # Clone chains out before clearing, since chain objects reference
        # into the model's storage.
        chains_by_name = {c.name: c.clone() for c in model}
        del model[:]
        for name in order:
            model.add_chain(chains_by_name[name])
        return self


    @staticmethod
    def _parse_chain_id_mapping(spec_string: str) -> dict:
        """Parse chain id mapping expressions

        Args:
            spec_string (str): example - "A:B,B:A,X:C"

        Returns:
            dict : {old_chain_id : new_chain_id, ...}
        """
        pattern = r"^(?P<old_chain>[A-Za-z0-9]+):(?P<new_chain>[A-Za-z0-9]+)$"
        parsed_specs = {}
        for sub_spec_string in spec_string.split(","):
            match = re.match(pattern, sub_spec_string)
            if match:
                d = match.groupdict()
                old = d.get('old_chain')
                new = d.get('new_chain')
                parsed_specs[old] = new
        return parsed_specs


    def rename(self, subs: str) -> "Editor":
        """Parse chain id mapping expressions

        Args:
            spec_string (str): example - "A/B,B/C", "A:10/A:100"
        """
        pattern = r"^(?P<chain_id>[A-Za-z0-9]+)(?::(?P<seqid>\d+))?$"
        for sub_spec_string in subs.split(","):
            source, target = sub_spec_string.split("/")
            source_match = re.match(pattern, source)
            target_match = re.match(pattern, target)
            s = source_match.groupdict()
            source_chain_id = s.get('chain_id')
            source_seqid = s.get('seqid')
            t = target_match.groupdict()
            target_chain_id = t.get('chain_id')
            target_seqid = t.get('seqid')
            if source_chain_id and target_chain_id:
                if source_chain_id == target_chain_id and source_seqid and target_seqid:
                    self.rename_residue(source_chain_id, source_seqid, target_seqid)
                else:
                    self.rename_chain(source_chain_id, target_chain_id)
        return self


    def rename_residue(self, chain_id: str, old_seqid: str, new_seqid: str) -> "Editor":
        residue = self._get_residue(chain_id, old_seqid)
        residue.seqid = gemmi.SeqId(str(new_seqid))
        return self
    
    def rename_chain(self, old_id: str, new_id: str) -> "Editor":
        chain = self._get_chain(old_id)
        chain.name = new_id
        return self


    def rename_chains(self, chain_map: str) -> "Editor":
        """Rename chain id(s)"""
        parsed_map = Editor._parse_chain_id_mapping(chain_map)
        old_ids = list(parsed_map.keys())
        chain_ids = self.chain_names()
        assert set(old_ids).issubset(set(chain_ids)), "invalid chain id(s)"
        unused_chain_ids = self._get_unused_chain_ids()
        # resolve conflict with intermediate chain id(s)
        resolved = {}
        for k, v in parsed_map.items():
            if v in chain_ids:
                w = unused_chain_ids.pop(0)
                self = self.rename_chain(v, w)
                resolved[v] = w

        for k, v in parsed_map.items():
            if k in resolved:
                self = self.rename_chain(resolved[k], v)
            else:
                self = self.rename_chain(k, v)

        self.reorder_chains()
        return self


    def remove_waters(self) -> "Editor":
        self.model.remove_waters()
        return self


    def remove_ligands_and_waters(self) -> "Editor":
        self.model.remove_ligands_and_waters()
        return self


    def remove_hydrogens(self) -> "Editor":
        self.model.remove_hydrogens()
        return self


    def remove_alternative_conformations(self) -> "Editor":
        """Collapse altlocs down to a single conformer (highest occupancy)."""
        self.model.remove_alternative_conformations()
        return self


    def renumber_residues(self, chain_id: str, start: int = 1) -> "Editor":
        """Renumber a chain's residues sequentially starting at `start`,
        clearing insertion codes."""
        chain = self._get_chain(chain_id)
        for offset, res in enumerate(chain):
            res.seqid = gemmi.SeqId(start + offset, " ") 
            # residue number, icode (empty/null character if no insertion code)
        return self


    def find_zn_coord_cys(self, model_idx: int = 0, cutoff: float = 3.5) -> set:
        """Find Cysteine residues possibly coordinating a zinc atom.

        Args:
            filename (str): _description_
            model_idx (int, optional): _description_. Defaults to 0.
            cutoff (float, optional): _description_. Defaults to 3.5.
                Although ideal Zn-SG(CYS) distance is 2.34 A, here we use 3.5 A as initial
                cutoff to identify Cys residues around a zinc atom.

        Returns:
            set: {(chain_id, resseq), ...}
        """
        model = self.structure[model_idx]
        ns = gemmi.NeighborSearch(model, self.structure.cell, 5.0).populate()
        zn_cys = set()
        for chain in model:
            for res in chain:
                if res.name != 'ZN':
                    continue
                zn_atom = res[0] # there must be only one atom in the ZN residue
                logger.info(f"Found a zinc atom {chain.name} {res.seqid.num}")
                for mark in ns.find_atoms(zn_atom.pos, '\0', radius=cutoff):
                    cra = mark.to_cra(model)
                    if cra.residue.name == 'CYS' and cra.atom.name == 'SG':
                        zn_cys.add((cra.chain.name, cra.residue.seqid.num))
                        logger.info(f"Found a zinc coordinating CYS at {cra.chain.name} {cra.residue.seqid.num}")
        return zn_cys


    def group_residues(self) -> dict[str, set]:
        """Sort protein and ligand."""
        groups: list[str] = ['protein', 'solvent', 'divalent', 'ligand']
        residue_group = {k: set() for k in groups}

        for chain in self.model:
            for res in chain:
                if res.name in self.std_residues_protein:
                    residue_group['protein'].add(res.name)

                elif res.name in self.std_residues_solvent:
                    residue_group['solvent'].add(res.name)
                
                elif res.name in self.std_residues_divalent_ion:
                    residue_group['divalent'].add(res.name)
                
                else:
                    residue_group['ligand'].add(res.name)
                    logger.info(f"ligand residue found {res.name}")

        return residue_group
    
    
    def set_as_ligand(self, resname: str, obabel: str = shutil.which("obabel")) -> "Editor":
        """Set ligand name and SMILES for structure (ligand)"""
        self.is_ligand = True
        self.mol_resname = resname
        self.mol_pdb_block = self.structure.make_pdb_string()
        self._ligand_convert_pdb_to_smiles(obabel=obabel)
        self._ligand_copy_positions()
        self._ligand_optimize(save=True)
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def residue_ids(self, chain_id: str) -> list[tuple[str, int]]:
        """[(resname, seqnum), ...] for a chain, handy for sanity-checking
        edits before writing to disk."""
        return [(r.name, r.seqid.num) for r in self._get_chain(chain_id)]


    def summary(self) -> None:
        lines = []
        for model_idx, model in enumerate(self.structure, start=1):
            lines.append(f"model {model_idx}")
            for chain in model:
                residues = [f"{res.name:>5} {res.seqid.num:>4}" for res in chain]
                n = len(residues)
                if n == 1:
                    lines.append(f"  chain {chain.name:<4} ({n:>3} residues): {residues[0]:<8}")
                else:
                    lines.append(f"  chain {chain.name:<4} ({n:>3} residues): {residues[0]:<8} ... {residues[-1]}")
                # non standard residues
                for residue in chain:
                    if residue.is_water():
                        continue
                    # look up the chemical component details in Gemmi's table
                    chem_comp = gemmi.find_tabulated_residue(residue.name)
                    # check if the residue name is missing or explicitly non-standard
                    if chem_comp is None or not chem_comp.is_standard():
                        na = len(residue)
                        lines.append(f"       non-standard residue  {residue.name:>5} {residue.seqid.num:>4} ({na} atoms)")

        print("\n".join(lines))
