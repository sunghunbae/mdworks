from mdworks import __version__, ValidComplex
from rdkit import Chem
from pathlib import Path

import argparse
import shutil
import subprocess


def guess_smiles() -> None:
    parser = argparse.ArgumentParser(
        description="Guess SMILES string from a ligand PDB file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb_file", type=str, help="Path to the ligand PDB file.")
    parser.add_argument("--obabel", type=str, default=shutil.which("obabel"), help="Path to the obabel executable (if not in system PATH).")
    args = parser.parse_args()

    if not Path(args.pdb_file).exists():
        raise FileNotFoundError(f"{args.pdb_file} does not exist")
    # Check if the 'obabel' command is in the system path
    obabel_path = shutil.which("obabel")
    if args.obabel is None:
        raise NotImplementedError("Error: requires obabel executable.")
    try:
        result = subprocess.run([args.obabel, "-ipdb", str(args.pdb_file), "-osmi"], 
                                capture_output=True, 
                                text=True, 
                                check=True
                                )
        output = result.stdout.strip()
        if output:
            # ex. <SMILES> <Name>
            smiles, name = output.split(maxsplit=1)
            print(f"Ligand SMILES: {smiles}")
        else:
            raise ValueError(f"Could not guess SMILES from {args.pdb_file}.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error occurred while running obabel on {args.pdb_file}: {e}")
        

def app() -> None:
    parser = argparse.ArgumentParser(
        description="Get receptor structure ready using PDBFixer and PDB2PQR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to the input PDB/MMCIF file.")
    parser.add_argument("--smiles", type=str, default=None, help="Ligand SMILES string.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output files.")
    parser.add_argument("--ff-ligand", type=str, default="openff-2.2.1.offxml", help="Force field for ligand.")
    parser.add_argument("--ff-protein", type=str, default="amber/protein.ff14SB.xml", help="Force field for protein.")
    parser.add_argument("--ff-water", type=str, default="amber/tip3p_standard.xml", help="Force field for water.")
    parser.add_argument("--solvent", type=str, default="tip3p", help="Solvent model.")
    parser.add_argument("--box-padding", type=float, default=1.0, help="Box padding in Angstroms.")
    parser.add_argument("--salt-conc", type=float, default=0.15, help="Salt concentration in Molar.")
    parser.add_argument("--positive-ion", type=str, default="Na+", help="Positive ion type.")
    parser.add_argument("--negative-ion", type=str, default="Cl-", help="Negative ion type.")
    parser.add_argument("--h-mass-factor", type=float, default=3.0, help="Hydrogen mass factor.")
    parser.add_argument("--partial-charge-method", type=str, default="am1bcc", help="Partial charge method for ligand.")
    
    args = parser.parse_args()

    print(f"Using mdworks version: {__version__}")

    assert Path(args.infile).exists(), f"Input file {args.infile} does not exist."

    if args.smiles:
        assert Chem.MolFromSmiles(args.smiles) is not None, f"Invalid SMILES string: {args.smiles}"

    vc = ValidComplex(args.infile)
    
    if args.smiles:
        vc.fix_ligand(args.smiles)
    
    vc.assign_ligand_charges(partial_charge_method = args.partial_charge_method)
    vc.save_protein()
    vc.save_ligand()
    vc.build(
        ff_ligand = args.ff_ligand,
        ff_protein = args.ff_protein,
        ff_water = args.ff_water,
        solvent = args.solvent,
        box_padding = args.box_padding,
        salt_conc = args.salt_conc,
        positive_ion = args.positive_ion,
        negative_ion = args.negative_ion,
        h_mass_factor = args.h_mass_factor
    )


if __name__ == "__main__":
    app()