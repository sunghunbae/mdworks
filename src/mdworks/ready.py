from pathlib import Path
from pdbfixer import PDBFixer  
from openmm.app import PDBFile, Topology, Modeller
from openmm.unit import angstroms, nanometers
from pdb2pqr.main import main_driver, build_main_parser

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



def get_receptor_ready(
        filename: str | Path | None = None, 
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
    # Validate input arguments
    if filename and pdb_id:
        raise ValueError("Provide either filename or pdb_id, not both.")
    if filename:
        assert isinstance(filename, (str, Path)), "filename must be a string or Path"
        if isinstance(filename, str):
            filename = Path(filename)
        assert Path(filename).exists(), f"{filename} does not exist"
        receptor_name = filename.stem
    elif pdb_id:
        assert isinstance(pdb_id, str), "pdb_id must be a string"
        assert len(pdb_id) == 4, "pdb_id must be a 4-character string"
        receptor_name = pdb_id
    else:
        raise ValueError("Either filename or pdb_id must be provided")

    if output_prefix is None:
        if filename:
            output_prefix = Path(filename).stem
        elif pdb_id:
            output_prefix = pdb_id

    if filename:
        workdir = Path(filename).parent
    elif pdb_id:
        workdir = Path.cwd()

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    fixed_pdb_path = f"{output_prefix}_fixed.pdb"  
    final_pqr_path = f"{output_prefix}_H.pqr"  
    final_pdb_path = f"{output_prefix}_ready.pdb"
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


def app():
    import argparse
    parser = argparse.ArgumentParser(
        description="Get receptor structure ready using PDBFixer and PDB2PQR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", type=str, help="Path to the input PDB/MMCIF file.")
    parser.add_argument("--pdb-id", type=str, help="PDB ID for the input structure.")
    parser.add_argument("--ligand", type=str, default=None, help="Residue name of the ligand to extract.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output files.")
    parser.add_argument("--pH", type=float, default=7.4, help="Target pH for protonation.")
    args = parser.parse_args()
    
    if args.pdb_id and args.file:
        raise ValueError("Provide either --file or --pdb-id, not both.")
    if args.pdb_id:
        get_receptor_ready(pdb_id=args.pdb_id, 
                           ligand_resname=args.ligand, 
                           output_prefix=args.prefix, 
                           target_pH=args.pH)
    if args.file:
        get_receptor_ready(filename=args.file, 
                           ligand_resname=args.ligand, 
                           output_prefix=args.prefix, 
                           target_pH=args.pH)


if __name__ == "__main__":
    app()