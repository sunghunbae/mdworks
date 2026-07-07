from pathlib import Path
from pdbfixer import PDBFixer  
from openmm.app import PDBFile
from pdb2pqr.main import main_driver, build_main_parser

import logging
from .utils import setup_logger


logger = logging.getLogger(__name__)


def get_receptor_ready(
        filename: str | Path | None = None, 
        pdb_id: str | None = None, 
        output_prefix: str | None = None, 
        target_pH: float = 7.4,
        quiet: bool = False) -> None:  
    """  
    Fix receptor structural issues and set protonation states.  
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

    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    fixed_pdb_path = f"{output_prefix}_fixed.pdb"  
    final_pqr_path = f"{output_prefix}_H.pqr"  
    final_pdb_path = f"{output_prefix}_ready.pdb"

    if filename:
        with open(filename, "r") as f:
            logger.info(f"PDBFixer reading a PDB file: {filename}")
            fixer = PDBFixer(pdbfile=f)
    elif pdb_id:
        logger.info(f"PDBFixer downloading the PDB {pdb_id} structure from RCSB Protein Data Bank")
        fixer = PDBFixer(pdbid=pdb_id)


    logger.info(f"--- Step 1: Fetching and Fixing {receptor_name} via PDBFixer ---")
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

    logger.info(f"\n--- Step 2: Predicting pKa and Protonating via PDB2PQR at pH {target_pH} ---")  
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
    logger.info(f"PDB2PQR protonated PQR file and saved to {final_pqr_path}")

    logger.info(f"\n--- Step 3: Generating clean, final PDB structure ---")
    # Generate companion structural pdb containing the newly calculated hydrogens
    pdb_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}",
        f"--pdb-output={final_pdb_path}",
        fixed_pdb_path,
        final_pqr_path]
    parsed_pdb_args = parser.parse_args(pdb_args)
    main_driver(parsed_pdb_args)
    logger.info(f"PDB2PQR saved final receptor to {final_pdb_path}")


def app():
    import argparse
    parser = argparse.ArgumentParser(description="Get receptor structure ready using PDBFixer and PDB2PQR.")
    parser.add_argument("--filename", type=str, help="Path to the input PDB/MMCIF file.")
    parser.add_argument("--pdb-id", type=str, help="PDB ID for the input structure.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output files.")
    parser.add_argument("--pH", type=float, default=7.4, help="Target pH for protonation (default: 7.4).")
    args = parser.parse_args()
    
    if args.pdb_id and args.filename:
        raise ValueError("Provide either --filename or --pdb-id, not both.")
    if args.pdb_id:
        get_receptor_ready(pdb_id=args.pdb_id, output_prefix=args.prefix, target_pH=args.pH)
    if args.filename:
        get_receptor_ready(filename=args.filename, output_prefix=args.prefix, target_pH=args.pH)


if __name__ == "__main__":
    app()