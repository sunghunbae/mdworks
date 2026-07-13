import argparse
import shutil
import subprocess
import typer

from pathlib import Path

from rdkit import Chem

from mdworks import __version__, ValidComplex
from mdworks.ready import get_receptor_ready
from mdworks.protocol import Equilibrium


app = typer.Typer(help='MMCIF tools to Get Protein Sequence or Convert to PDB')

def main():
    parser = argparse.ArgumentParser(description="mdworks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prefix", type=str, help="Prefix for output files.")
    parser.add_argument("--quiet", action="store_true", help="No stdout")

    subparsers = parser.add_subparsers(dest="command", 
                                       required=True, 
                                       help="Available commands")
    
    # Command: 'ready'
    parser_ready = subparsers.add_parser("ready", 
                                         help="Get system ready for MD", 
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_ready.add_argument("infile", type=str, help="Path to the input PDB/MMCIF file.")
    parser_ready.add_argument("--ligand", type=str, help="Residue name of the ligand to extract.")
    parser_ready.add_argument("--pH", type=float, default=7.4, help="Target pH for protonation.")
    parser_ready.add_argument("--info-only", action="store_true", help="Show info and exit.")

    # Command: 'guess'
    parser_guess = subparsers.add_parser("guess", 
                                         help="Guess SMILES from a ligand PDB file", 
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_guess.add_argument("infile", type=str, help="Path to the ligand PDB file.")
    parser_guess.add_argument("--obabel", type=str, default=shutil.which("obabel"), help="Path to the obabel executable (if not in system PATH).")

    # Command: 'build'
    parser_build = subparsers.add_parser("build", 
                                         help="Build MD system", 
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_build.add_argument("infile", type=str, help="Path to the input PDB/MMCIF file.")
    parser_build.add_argument("--smiles", type=str, default=None, help="Ligand SMILES string.")
    parser_build.add_argument("--ff-ligand", type=str, default="openff-2.2.1.offxml", help="Force field for ligand.")
    parser_build.add_argument("--ff-protein", type=str, default="amber/protein.ff14SB.xml", help="Force field for protein.")
    parser_build.add_argument("--ff-water", type=str, default="amber/tip3p_standard.xml", help="Force field for water.")
    parser_build.add_argument("--solvent", type=str, default="tip3p", help="Solvent model.")
    parser_build.add_argument("--box-padding", type=float, default=1.0, help="Box padding in Angstroms.")
    parser_build.add_argument("--salt-conc", type=float, default=0.15, help="Salt concentration in Molar.")
    parser_build.add_argument("--positive-ion", type=str, default="Na+", help="Positive ion type.")
    parser_build.add_argument("--negative-ion", type=str, default="Cl-", help="Negative ion type.")
    parser_build.add_argument("--h-mass-factor", type=float, default=3.0, help="Hydrogen mass factor.")
    parser_build.add_argument("--partial-charge-method", type=str, default="am1bcc", help="Partial charge method for ligand.")
    
    # Command: 'relax'
    parser_relax = subparsers.add_parser("relax",
                                         help="Run equilibrium MD simulation",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_relax.add_argument("infile", type=str, help="Path to the input PDB/MMCIF file.")
    parser_relax.add_argument("--temperature", type=float, default=300.0, help="Temperature for the simulation.")
    parser_relax.add_argument("--pressure", type=float, default=1.0, help="Pressure for the simulation.")
    parser_relax.add_argument("--workdir", type=str, default=".", help="Working directory for the simulation.")
    parser_relax.add_argument("--platform", type=str, default="CUDA", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU).")
    parser_relax.add_argument("--devices", type=str, default="1", help="GPU devices for the simulation (e.g., '0', '0,1').")

    args = parser.parse_args()

    print(f"mdworks version: {__version__}")
   
    if not Path(args.infile).exists():
        raise FileNotFoundError(f"{args.infile} does not exist")

    if args.command == "ready":
        get_receptor_ready(
            filename=args.infile, 
            ligand_resname=args.ligand, 
            output_prefix=args.prefix, 
            target_pH=args.pH,
            info_only=args.info)

    elif args.command == "guess":
        if args.obabel is None:
            raise NotImplementedError("Error: requires obabel executable.")
        try:
            result = subprocess.run([args.obabel, "-ipdb", str(args.infile), "-osmi"], 
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
                raise ValueError(f"Could not guess SMILES from {args.infile}.")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error occurred while running obabel on {args.infile}: {e}")

    elif args.command == "build":
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

    elif args.command == "relax":
        md = Equilibrium(args.infile,
                    temperature= args.temperature,
                    pressure= args.pressure, 
                    workdir=args.workdir, 
                    platform= args.platform, 
                    devices= args.devices)
        md.run()



if __name__ == "__main__":
    main()
