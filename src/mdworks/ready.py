from operator import ne
from pathlib import Path
from pdbfixer import PDBFixer  
from openmm.app import PDBFile, Topology, Modeller
from openmm.unit import angstroms, nanometers
from pdb2pqr.main import main_driver, build_main_parser
from Bio import PDB

import MDAnalysis as mda

import re
import numpy as np
import shutil
import subprocess
import numpy
import logging

from copy import deepcopy
from .utils import setup_logger


logger = logging.getLogger(__name__)


def isolate(
        filename: str | Path, 
        ligand_resname: str, 
        output_ligand_file: str | Path) -> None:
    """
    Extract ligand from PDB file and save to a separate PDB file.
    """
    filename = Path(filename)
    output_ligand_file = Path(output_ligand_file)

    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist")

    fixer = PDBFixer(filename= filename.as_posix())
    ligand_atoms = [atom for atom in fixer.topology.atoms() if atom.residue.name == ligand_resname]

    if not ligand_atoms:
        raise ValueError(f"No atoms found for ligand with resname '{ligand_resname}' in {filename}")

    # Create a new topology and positions for the ligand
    ligand_topology = Topology()
    new_chain = ligand_topology.addChain()
    new_residue = ligand_topology.addResidue(ligand_resname, new_chain)

    ligand_positions = []
    atom_mapping = {}

    for atom in ligand_atoms:
        new_atom = ligand_topology.addAtom(atom.name, atom.element, new_residue)
        ligand_positions.append(fixer.positions[atom.index].value_in_unit(angstroms))
        atom_mapping[atom] = new_atom
    
    ligand_positions = numpy.array(ligand_positions)

    for bond in fixer.topology.bonds():
        if bond.atom1 in atom_mapping and bond.atom2 in atom_mapping:
            ligand_topology.addBond(atom_mapping[bond.atom1], atom_mapping[bond.atom2])

    # Write the ligand to a new PDB file
    with open(output_ligand_file, "w") as f:
        PDBFile.writeFile(ligand_topology, ligand_positions, f)



def merge(protein_pdb: str | Path, 
          ligand_pdb: str | Path, 
          complex_path: str | Path) -> None:
    """
    Programmatically merges the fixed protein and original ligand structuresusing OpenMM's Modeller framework 
    instead of raw string manipulation.
    """
    # 1. Load the fixed and protonated protein 
    receptor = PDBFile(protein_pdb)
    
    # 2. Load the isolated ligand file
    ligand = PDBFile(ligand_pdb)
    
    # 3. Initialize the Modeller object with the protein structure
    modeller = Modeller(receptor.topology, receptor.positions)
    
    # 4. Programmatically combine the ligand's topology and spatial coordinates
    modeller.add(ligand.topology, ligand.positions)
    
    # 5. Write the combined system out to a standard, compliant PDB file
    with open(complex_path, "w") as f_out:
        PDBFile.writeFile(modeller.topology, modeller.positions, f_out)


def receptor(
        filename: str | None = None, 
        pdb_id: str | None = None,
        ligand_resname: str | None = None,
        output_prefix: str | None = None, 
        target_pH: float = 7.4,
        quiet: bool = False) -> None:  
    """  
    Fix receptor structural issues and set protonation states.
    PDBFixer/PDB2PQR workflow excludes non-standard residus including ligands, cofactors, and water molecules.
    So, if the receptor structure contains a ligand, it should be extracted and processed separately.
    """  
    if filename:
        filename = Path(filename)
        receptor_name = filename.stem
        workdir = filename.parent
    elif pdb_id:
        receptor_name = pdb_id
        workdir = Path.cwd()
    else:
        raise ValueError("Either filename or pdb_id must be provided")

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    fixed_pdb_path = f"{output_prefix}_fixed.pdb"  
    final_pqr_path = f"{output_prefix}_H.pqr"  
    final_pdb_path = f"{output_prefix}_H.pdb"

    complex_path = f"{output_prefix}_complex.pdb"

    if filename:
        logger.info(f"PDBFixer reading a file: {filename}")
        fixer = PDBFixer(filename= filename.as_posix())
    elif pdb_id:
        logger.info(f"PDBFixer downloading the PDB {pdb_id} structure from RCSB Protein Data Bank")
        fixer = PDBFixer(pdbid= pdb_id)

    if ligand_resname:
        ligand_output_file = f"{output_prefix}_{ligand_resname}.pdb"
        logger.info(f"[Step 0] Isolating Ligand {ligand_resname} ...")
        isolate(filename or f"{pdb_id}.pdb", ligand_resname, ligand_output_file)
        logger.info(f"Isolated ligand {ligand_resname} saved to {ligand_output_file}")

    logger.info(f"[Step 0] Disconnect bonds with Zinc atom(s) ...")
    # Track down the Zinc atom index
    zinc_atom_indices = [
        atom.index for atom in fixer.topology.atoms() if "ZN" in atom.name.upper()
    ]

    # Rebuild the bond network, explicitly excluding any bonds involving the Zinc atom
    clean_bonds = []
    for bond in fixer.topology.bonds():
        if bond[0].index not in zinc_atom_indices and bond[1].index not in zinc_atom_indices:
            clean_bonds.append(bond)

    # Overwrite the topology's bond dictionary with the filtered list
    fixer.topology._bonds = clean_bonds

    logger.info(f"[Step 1] Fetching and Fixing {receptor_name} via PDBFixer ...")

    # PDBFixer uses geometry template to fill in missing residues and atoms, 
    # and to replace nonstandard residues with standard ones.
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    # Add missing heavy atoms (but do not add hydrogens yet; PDB2PQR will do that)  

    # The missingResidues dictionary stores missing residues as: 
    # (chainIndex, residueIndex): [list of residue names]
    # example:
    # {
    #   (0, 0): ['MET', 'GLU', ..., 'PRO', 'SER']), 
    #   (0, 108): ['PRO', 'VAL', ... , 'VAL'], 
    #   (0, 232): ['VAL', ..., 'ARG', 'LEU']
    # }
    chains = list(fixer.topology.chains())
    residues = list(fixer.topology.residues())
    for (chain_idx, res_idx), resnames in sorted(fixer.missingResidues.items()):
        # residues will be inserted at res_idx in chain_idx
        chain = chains[chain_idx]
        residue = residues[res_idx]
        n = len(resnames)
        insertion_at = int(residue.id)
        logger.info(f"{n} residues to be inserted before {chain.id}:{insertion_at:<4d} {','.join(resnames[:4])}..")
    fixer.addMissingAtoms()
    fixer.removeChains(chainIndices=[-1])
    
    # Write intermediate fixed heavy-atom structure  
    with open(fixed_pdb_path, "w") as f:  
        PDBFile.writeFile(fixer.topology, fixer.positions, f)  
        logger.info(f"PDBFixer fixed heavy atoms and saved to {fixed_pdb_path}")

    logger.info(f"[Step 2] Predicting pKa and Protonating via PDB2PQR at pH {target_pH} ...")  
    # Build PDB2PQR command-line arguments to run programmatically  
    # Using AMBER forcefield naming convention for downstream MD compatibility  
    pdb2pqr_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}", 
        fixed_pdb_path, 
        final_pqr_path]
    # Correct programmatic call sequence: parse options list via internal parser object
    parser = build_main_parser()
    parsed_args = parser.parse_args(pdb2pqr_args)
    # Execute driver engine
    main_driver(parsed_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR protonated PQR file and saved to {final_pqr_path}")
    logger.info(f"[Step 3] Generating clean, final PDB structure with hydrogens via PDB2PQR ...")
    
    # Generate companion structural pdb containing the newly calculated hydrogens
    pdb_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}",
        f"--pdb-output={final_pdb_path}",
        fixed_pdb_path,
        final_pqr_path]
    parsed_pdb_args = parser.parse_args(pdb_args)
    main_driver(parsed_pdb_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR saved final receptor to {final_pdb_path}")

    if ligand_resname:
        logger.info(f"[Step 4] Merging final receptor and original ligand {ligand_resname} ...")
        merge(final_pdb_path, ligand_output_file, complex_path)
        logger.info(f"Merged complex saved to {complex_path}")



def guess_smiles_from_pdb(filename: str, obabel: str = shutil.which("obabel")) -> str:
    try:
        in_path = Path(filename)
        prefix = in_path.name.removesuffix("".join(in_path.suffixes))
        assert in_path.exists(), "file not found"
        result = subprocess.run([obabel, "-ipdb", in_path.as_posix(), "-osmi"], 
                                capture_output=True, 
                                text=True, 
                                check=True
                                )
        output = result.stdout.strip()
        if output:
            # ex. <SMILES> <Name>
            smiles, name = output.split(maxsplit=1)
            with open(f"{prefix}.smi", "w") as f:
                f.write(f"{smiles}\n")
            return smiles
        else:
            raise ValueError(f"Could not guess SMILES from {filename}.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error occurred while running obabel on {filename}: {e}")



def parse_selection(spec_string: str) -> list[tuple]:
    """Parse residue range expressions

    Args:
        spec_string (str): example - "A:10-30,A:100-120,B:1-50,C:1,D"

    Returns:
        list : [(chain_id, int(start), int(end)), ...]
    """
    # pattern = r"^(?:#(?P<model>\d+(?:\.\d+)*))?(?:\/(?P<chain>[A-Za-z0-9]+))?(?::(?P<residue>\d+(?:-\d+)?))?(?:@(?P<atom>[A-Za-z0-9*?]+))?$"
    pattern = r"^(?P<chain>[A-Za-z0-9]+)?(?::(?P<residue>\d+(?:-\d+)?))?$"

    parsed_specs = []
    for sub_spec_string in spec_string.split(","):
        match = re.match(pattern, sub_spec_string)
        if match:
            d = match.groupdict()
            chain_id = d.get('chain', None)
            i = None
            j = None
            if d.get('residue'):
                ij = d.get('residue').split('-')
                if len(ij) == 2:
                    i = int(ij[0])
                    j = int(ij[1])
                elif len(ij) == 1:
                    i = int(ij[0])
                    j = None
            parsed_specs.append((chain_id, i, j))

    return parsed_specs


def cut(
        filename: str, 
        selection: str,
        output_prefix: str | None = None,
        quiet: bool = False) -> None:
    """Cut residues

    Args:
        filename (str): Input .pdb or .cif file
        selection (str): Selection of chain:residues. 
            Example: `A:1-10,A:130-150,B:1-30`
            - chain and residue range is separated by `:`
            - residue range is defined by `-`
            - multiple selections are separated by `,`
        output_prefix (str): Output prefix
        quiet (bool): If True, no output is shown.
    """
    receptor_name = Path(filename).stem
    workdir = Path(filename).parent
    targets = parse_selection(selection)

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_cut.pdb"  

    with open(filename, "r") as f:
        logger.info(f"PDBFixer reading a PDB file: {filename}")
        pdb = PDBFixer(pdbfile=f)
        modeller = Modeller(pdb.topology, pdb.positions)
        
    residues_to_delete = []
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            resseq = int(residue.id)
            for (chain_id, i, j) in targets:
                if all([
                    chain_id is None or chain_id == chain.id,
                    i is None or (i <= resseq),
                    j is None or (j >= resseq),
                    ]):
                    residues_to_delete.append(residue)
    modeller.delete(residues_to_delete)

    # Write intermediate fixed heavy-atom structure  
    with open(output_filename, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Modified coordinates are saved to {output_filename}")



def reorder(filename: str, 
            output_prefix: str | None = None, 
            quiet: bool = False):

    receptor_name = Path(filename).stem
    workdir = Path(filename).parent

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_ordered.pdb"  

    # 1. Load the structure
    u = mda.Universe(filename)
    
    # 2. Sort the atoms by chainID using Python's sorted()
    # MDAnalysis allows sorting by arbitrary atom attributes like 'chainID'
    sorted_atoms = sorted(u.atoms, key=lambda atom: atom.chainID)

    # 3. Group the sorted list back into a functional AtomGroup
    sorted_group = mda.AtomGroup(sorted_atoms)

    # 4. Write directly to a new PDB file
    with mda.Writer(output_filename, sorted_group.n_atoms) as W:
        W.write(sorted_group)
        logger.info(f"Reordered coordinates saved to {output_filename}")


def split_traj(filename: str, 
            output_prefix: str | None = None, 
            quiet: bool = False):

    receptor_name = Path(filename).stem
    workdir = Path(filename).parent

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    # 1. Load the structure
    u = mda.Universe(filename)

    # iterate through each model (frame)
    for model in u.trajectory:
        output_filename = f"{output_prefix}_{model.frame}.pdb"
        # Select all atoms and write the current frame to a new PDB file
        with mda.Writer(output_filename, u.atoms.n_atoms) as W:
            W.write(u.atoms)
            logger.info(f"Model {model.frame} saved to {output_filename}")


class ModelSelect(PDB.Select):
    """Custom selection class to filter for a specific model."""
    def __init__(self, model_id):
        self.model_id = model_id

    def accept_model(self, model):
        # Only accept the model matching the targeted ID
        return 1 if model.id == self.model_id else 0


def split(filename: str, 
          output_prefix: str | None = None, 
          quiet: bool = False):

    receptor_name = Path(filename).stem
    workdir = Path(filename).parent

    if output_prefix is None:
        output_prefix = receptor_name

    # Initialize the PDB parser and IO object
    parser = PDB.PDBParser(QUIET=quiet)
    io = PDB.PDBIO()
    
    # Load the structure hierarchy
    structure = parser.get_structure("protein", filename)
    io.set_structure(structure)
    
    # Iterate through every model in the structure
    for model in structure:
        output_filename = f"{output_prefix}_{model.id}.pdb"
        
        # Write the structure out, applying the model filter
        io.save(output_filename, select=ModelSelect(model.id))
        logger.info(f"Model {model.id} saved to {output_filename}")


def rename(filename: str, 
           chain_mapping: dict, 
           output_prefix: str | None = None, 
           quiet: bool = False):

    receptor_name = Path(filename).stem
    workdir = Path(filename).parent

    if output_prefix is None:
        output_prefix = receptor_name
    
    output_filename = f"{output_prefix}_rename.pdb"

    parser = PDB.PDBParser(QUIET=quiet)
    
    structure = parser.get_structure("protein", filename)
    
    for model in structure:
        # Extract and detach targeted chains to prevent collision errors
        chains_to_modify = {}
        for old_id in list(model.child_dict.keys()):
            if old_id in chain_mapping:
                chains_to_modify[old_id] = model.child_dict[old_id]
                model.detach_child(old_id)
        
        # Assign new IDs and reattach them to the model hierarchy
        for old_id, chain_obj in chains_to_modify.items():
            new_id = chain_mapping[old_id]
            chain_obj.id = new_id
            model.add(chain_obj)
            
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_filename)



def _reorder(filename: str, 
            output_prefix: str | None = None, 
            quiet: bool = False):
    """Unsuccessful implementation"""
    receptor_name = Path(filename).stem
    workdir = Path(filename).parent

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_ordered.pdb"  

    with open(filename, "r") as f:
        logger.info(f"PDBFixer reading a PDB file: {filename}")
        pdb = PDBFixer(pdbfile=f)
        modeller = Modeller(pdb.topology, pdb.positions)

    # Assuming `modeller` is your existing Modeller object
    old_topology = modeller.topology
    old_positions = modeller.positions

    # 1. Get the list of chains and sort them by chain.id
    sorted_chains = sorted(old_topology.chains(), key=lambda chain: chain.id)

    # 2. Create a new Topology object and mapping for atoms
    new_topology = Topology()
    new_positions = []
    atom_map = {}

    # 3. Add chains and residues in the sorted order
    for chain in sorted_chains:
        new_chain = new_topology.addChain(chain.id)
        for residue in chain.residues():
            new_residue = new_topology.addResidue(
                residue.name, new_chain, residue.id, residue.insertionCode
            )
            for atom in residue.atoms():
                new_atom = new_topology.addAtom(
                    atom.name, atom.element, new_residue, atom.id, atom.formalCharge
                )
                atom_map[atom] = new_atom
                new_positions.append(deepcopy(old_positions[atom.index]))

    # 4. Copy over the bonds using the new atom references
    for bond in old_topology.bonds():
        # Ensure both atoms in the bond exist in the new map
        if bond[0] in atom_map and bond[1] in atom_map:
            new_topology.addBond(
                atom_map[bond[0]], atom_map[bond[1]], bond.type, bond.order
            )

    if old_topology.getPeriodicBoxVectors() is not None:
        new_topology.setPeriodicBoxVectors(old_topology.getPeriodicBoxVectors())

    # 5. Initialize the updated Modeller
    modeller = Modeller(new_topology, new_positions)

    # Write intermediate fixed heavy-atom structure  
    with open(output_filename, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Reordered coordinates are saved to {output_filename}")
