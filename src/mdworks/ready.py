from operator import ne
from pathlib import Path
from pdbfixer import PDBFixer  
from openmm.app import PDBFile, Topology, Modeller
from openmm.unit import angstroms, nanometers
from pdb2pqr.main import main_driver, build_main_parser

import re
import shutil
import subprocess
import numpy
import logging

from .utils import setup_logger


logger = logging.getLogger(__name__)


def isolate(
        pdb_file: str | Path, 
        ligand_resname: str, 
        output_ligand_file: str | Path) -> None:
    """
    Extract ligand from PDB file and save to a separate PDB file.
    """
    pdb_file = Path(pdb_file)
    output_ligand_file = Path(output_ligand_file)

    if not pdb_file.exists():
        raise FileNotFoundError(f"{pdb_file} does not exist")

    with open(pdb_file, "r") as f:
        fixer = PDBFixer(pdbfile=f)
        ligand_atoms = [atom for atom in fixer.topology.atoms() if atom.residue.name == ligand_resname]

    if not ligand_atoms:
        raise ValueError(f"No atoms found for ligand with resname '{ligand_resname}' in {pdb_file}")

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
        receptor_name = Path(filename).stem
        workdir = Path(filename).parent
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
        with open(filename, "r") as f:
            logger.info(f"PDBFixer reading a PDB file: {filename}")
            fixer = PDBFixer(pdbfile=f)
    elif pdb_id:
        logger.info(f"PDBFixer downloading the PDB {pdb_id} structure from RCSB Protein Data Bank")
        fixer = PDBFixer(pdbid=pdb_id)

    if ligand_resname:
        ligand_output_file = f"{output_prefix}_{ligand_resname}.pdb"
        logger.info(f"[Step 0] Isolating Ligand {ligand_resname} ...")
        isolate(filename or f"{pdb_id}.pdb", ligand_resname, ligand_output_file)
        logger.info(f"Isolated ligand {ligand_resname} saved to {ligand_output_file}")


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



def parse_residue_ranges(range_string: str) -> list[tuple]:
    """Parse residue range expressions

    Args:
        range_string (str): example - "A:10-30,A:100-120,B:1-50"

    Returns:
        list : [(chain_id, int(start), int(end)), ...]
    """
    # Regex captures: chain ID (group 1), start residue (group 2), end residue (group 3)
    pattern = r"([A-Za-z0-9]+):(\d+)-(\d+)"
    matches = re.findall(pattern, range_string)
    
    parsed_ranges = []
    for chain_id, start, end in matches:
        parsed_ranges.append( (chain_id, int(start), int(end)) )
    return parsed_ranges


def cut(
        filename: str, 
        residues: str,
        output_prefix: str | None = None,
        quiet: bool = False) -> None:
    """Cut residues

    Args:
        filename (str): Input .pdb or .cif file
        residues (str): Selection of residues. 
            Example: `A:1-10,A:130-150,B:1-30`
            - chain and residue range is separated by `:`
            - residue range is defined by `-`
            - multiple selections are separated by `,`
        output_prefix (str): Output prefix
        quiet (bool): If True, no output is shown.
    """
    receptor_name = Path(filename).stem
    workdir = Path(filename).parent
    targets = parse_residue_ranges(residues)

    if output_prefix is None:
        output_prefix = receptor_name

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_pdb_path = f"{output_prefix}_cut.pdb"  

    with open(filename, "r") as f:
        logger.info(f"PDBFixer reading a PDB file: {filename}")
        pdb = PDBFixer(pdbfile=f)
        modeller = Modeller(pdb.topology, pdb.positions)

    residues_to_delete = []
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            resseq = int(residue.id)
            if any([(chain.id == chain_id) and (start <= resseq <= end) for (chain_id, start, end) in targets]):
                residues_to_delete.append(residue)

    modeller.delete(residues_to_delete)

    # Write intermediate fixed heavy-atom structure  
    with open(output_pdb_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Modified coordinates are saved to {output_pdb_path}")