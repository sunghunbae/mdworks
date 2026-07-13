import shutil
import subprocess
import typer

from typing import Annotated, Optional
from pathlib import Path


app = typer.Typer(help='mdworks')


def version_callback(value: bool):
    if value:
        from mdworks import __version__
        print(f"mdworks version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True)):
    pass


@app.command()
def cif2info(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """Get information from a .cif file"""
    from mdworks.mmcif import get_info
    get_info(filename)


@app.command()
def cif2seq(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """Get missing residue(s) and sequence from a .cif file

    - Full sequence
    - Coordinate sequence with missing residues (`-`)
    """
    from mdworks.mmcif import (
        get_residue_poly_sequences,
        get_entity_poly_sequences,
        get_nonpoly_chains_and_residues,
    )
    from Bio import Align

    coor_seq = get_residue_poly_sequences(filename)
    auth_seq = get_entity_poly_sequences(filename)

    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1.0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = 0
    
    # Perform global alignment (simple, without scoring)
    alignments = aligner.align(auth_seq['1'], coor_seq['A'])
    alignment = alignments[0]
    print(alignment[0])
    print(alignment[1])
    print(f'aligner algorithm= {aligner.algorithm}')
    print(f'  length={alignment.length}')
    print(f'  aligned={alignment.aligned}')

    ligand = get_nonpoly_chains_and_residues(filename)
    print(ligand)



@app.command()
def cif2pdb(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """Convert .cif to .pdb"""
    from mdworks.mmcif import convert_mmcif_to_pdb

    print(f'converting {filename} to .pdb')
    convert_mmcif_to_pdb(filename)



@app.command(name="ready")
def ready(
    infile: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")],
    ligand: Annotated[str, typer.Option("--ligand", help="Ligand residue name")] = "",
    pH: Annotated[float, typer.Option("--pH", help="Target pH for protonation")] = 7.4):
    """Get protein complex system ready for MD"""
    if not Path(infile).exists():
        raise FileNotFoundError(f"{infile} does not exist")
    from mdworks.ready import get_receptor_ready
    get_receptor_ready(filename=infile, ligand_resname=ligand, target_pH=pH)


@app.command(name="guess")
def guess(
    infile: Annotated[str, typer.Argument(help="Input .pdb or .cif filename")],
    obabel: Annotated[str, typer.Option("--obabel", help="Path to the obabel executable")] = shutil.which("obabel")):
    """Guess SMILES from a ligand .pdb file"""
    if not Path(infile).exists():
        raise FileNotFoundError(f"{infile} does not exist")
    if obabel is None:
        raise NotImplementedError("Error: requires obabel executable.")
    try:
        result = subprocess.run([obabel, "-ipdb", str(infile), "-osmi"], 
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
            raise ValueError(f"Could not guess SMILES from {infile}.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error occurred while running obabel on {infile}: {e}")


@app.command(name="build")
def build(
    infile: Annotated[str, typer.Argument(help="Input .pdb or .cif filename")],
    smiles: Annotated[str, typer.Option("--smiles", help="Ligand SMILES string")] = "",
    ff_ligand: Annotated[str, typer.Option("--ff-ligand", help="Force field for ligand")] = "openff-2.2.1.offxml",
    ff_protein: Annotated[str, typer.Option("--ff-protein", help="Force field for protein")] ="amber/protein.ff14SB.xml",
    ff_water: Annotated[str, typer.Option("--ff-water", help="Force field for water.")] = "amber/tip3p_standard.xml",
    solvent: Annotated[str, typer.Option("--solvent", help="Solvent model (tip3p/gbn2/obc2/gbn1/obc1).")] = "tip3p",
    box_padding: Annotated[float, typer.Option("--box-padding", help="Box padding in Nanometer.")] = 1.0,
    salt_conc: Annotated[float, typer.Option("--salt-conc", help="Salt concentration in Molar.")] = 0.15,
    positive_ion: Annotated[str, typer.Option("--positive-ion", help="Positive ion type.")] = "Na+",
    negative_ion: Annotated[str, typer.Option("--negative-ion", help="Negative ion type.")] = "Cl-",
    h_mass_factor: Annotated[float, typer.Option("--h-mass-factor", help="Hydrogen mass factor.")] = 3.0,
    partial_charge_method: Annotated[str, typer.Option("--partial-charge-method", help="Partial charge method for ligand.")] = "am1bcc"):
    """Build MD system with implicit/explicit water box"""
    if not Path(infile).exists():
        raise FileNotFoundError(f"{infile} does not exist")

    from mdworks import ValidComplex
    from rdkit import Chem

    if smiles:
        assert Chem.MolFromSmiles(smiles) is not None, f"Invalid SMILES string: {smiles}"

    vc = ValidComplex(infile)
    
    if smiles:
        vc.fix_ligand(smiles)
    
    vc.assign_ligand_charges(partial_charge_method = partial_charge_method)
    vc.save_protein()
    vc.save_ligand()
    vc.build(
        ff_ligand = ff_ligand,
        ff_protein = ff_protein,
        ff_water = ff_water,
        solvent = solvent,
        box_padding = box_padding,
        salt_conc = salt_conc,
        positive_ion = positive_ion,
        negative_ion = negative_ion,
        h_mass_factor = h_mass_factor
    )


@app.command(name="relax")
def relax(
    infile: Annotated[str, typer.Argument(help="Input .pdb or .cif filename")],
    temperature: Annotated[float, typer.Option("--temperature", help="Temperature for the simulation")] = 300.0,
    pressure: Annotated[float, typer.Option("--temperature", help="Pressure for the simulation")] = 1.0,
    workdir: Annotated[str, typer.Option("--workdir", help="Working directory for the simulation")] = ".",
    platform: Annotated[str, typer.Option("--platform", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU)")] = "CUDA",
    devices: Annotated[str, typer.Option("--devices", help="GPU devices for the simulation (e.g., '0', '0,1')")] = "0"):
    """Relax/equilibrate MD system"""
    if not Path(infile).exists():
        raise FileNotFoundError(f"{infile} does not exist")
    from mdworks.protocol import Equilibrium
    md = Equilibrium(infile,
                temperature= temperature,
                pressure= pressure, 
                workdir= workdir, 
                platform= platform, 
                devices= devices)
    md.run()


if __name__ == '__main__':
    app()