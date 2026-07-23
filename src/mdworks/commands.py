import shutil
import typer
import json

from typing import Annotated, Optional
from pathlib import Path

import mdworks.mmcif as mmcif
import mdworks.ready as mdready

from mdworks import ValidComplex
from mdworks.protocol import Relax, Equilibrium, Desmond, Production

from rdkit import Chem
from Bio import Align


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
def mmcifinfo(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """(mmCIF) Get information"""
    mmcif.info(filename)


@app.command()
def mmcif2seq(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """(mmCIF) Get missing residue(s) and sequence

    - Full sequence
    - Coordinate sequence with missing residues (`-`)
    """
    coor_seq = mmcif.get_residue_poly_sequences(filename)
    auth_seq = mmcif.get_entity_poly_sequences(filename)

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

    ligand = mmcif.get_nonpoly_chains_and_residues(filename)
    print(ligand)



@app.command()
def mmcif2pdb(filename: Annotated[str, typer.Argument(help="Input .cif filename.")]):
    """(mmCIF) Convert to PDB"""
    print(f'converting {filename} to .pdb')
    mmcif.convert_to_pdb(filename)

@app.command()
def peek(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")]):
    """(Optional) Peek and show summary of model(s) and chain(s)"""
    mdready.peek(filename=filename)

@app.command()
def guess(filename: Annotated[str, typer.Argument(help="Input ligand .pdb")],
          obabel: Annotated[str, typer.Option("--obabel", help="Path to the obabel executable")] = shutil.which("obabel")):
    """(Optional) Guess SMILES from a ligand .pdb file"""
    if obabel is None:
        raise NotImplementedError("Error: requires obabel executable.")
    smiles = mdready.guess_smiles_from_pdb(filename, obabel)
    print(smiles)


@app.command()
def delete(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")],
           selection: Annotated[str, typer.Argument(help="Select chain:residues to remove: ex. A:1-30,A:200-230,B:1-10,C,D:1")] = "",
           tag: Annotated[str, typer.Argument(help="Output tag")] = 'del',
           ):
    """(Optional) Delete chain(s) and residue(s)"""
    mdready.delete(filename=filename, selection=selection, tag=tag)


@app.command()
def reorder(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")],
            tag: Annotated[str, typer.Argument(help="Output tag")] = 'ord',):
    """(Optional) Reorder chain(s) by chain id"""
    mdready.reorder(filename=filename, tag=tag)


@app.command()
def rename(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")],
           chain_map: Annotated[str, typer.Argument(help="Chain id map. Ex. A:B,B:C,D:C")],
           tag: Annotated[str, typer.Argument(help="Output tag")] = 'ren',
    ):
    """(Optional) Rename chain id(s)"""
    mdready.rename(filename=filename, chain_map=chain_map, tag=tag)


@app.command()
def split(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")]):
    """(Optional) Split and write individual model(s)"""
    mdready.split(filename=filename)


@app.command()
def test(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")]):
    """(Optional) Split and write individual model(s)"""
    mdready.test(filename=filename)


@app.command()
def ready(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename.")],
          ligand: Annotated[str, typer.Option("--ligand", help="Ligand residue name")] = "",
          waters: Annotated[bool, typer.Option("--waters/--no-waters", help="Keep waters or not")] = False,
          separate_hetgens: Annotated[bool, typer.Option("--separate-hetgens/--no-separate-hetgens", help="Separate chains for hetgens (Zn, ligand, ...)")] = True,
          zinc: Annotated[bool, typer.Option("--zinc/--no-zinc", help="Handle zinc related issue")] = True,
          terminals: Annotated[bool, typer.Option("--terminal/--no-terminal", help="Add missing residues at N/C-terminals or not")] = False,
          obabel: Annotated[str, typer.Option("--obabel", help="Path to the obabel executable")] = shutil.which("obabel"),
          pH: Annotated[float, typer.Option("--pH", help="Target pH for protonation")] = 7.4,
    ):
    """Get system ready for MD"""
    mdready.complex(filename=filename,
                    ligand_resname=ligand, 
                    waters=waters, 
                    separate_hetgens=separate_hetgens,
                    zinc= zinc,
                    terminals = terminals,
                    obabel=obabel, 
                    target_pH=pH)


@app.command()
def relax(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename")],
          smiles: Annotated[str, typer.Option("--smiles", help="Ligand SMILES")] = "",
          ligand: Annotated[str, typer.Option("--ligand", help="Ligand residue name")] = "",
          partial_charge_method: Annotated[str, typer.Option("--partial-charge-method", help="Partial charge method for ligand.")] = "am1bcc",
          ff_ligand: Annotated[str, typer.Option("--ff-ligand", help="Force field for ligand")] = "openff-2.2.1.offxml",
          ff_protein: Annotated[str, typer.Option("--ff-protein", help="Force field for protein")] ="amber/protein.ff14SB.xml",
          solvent: Annotated[str, typer.Option("--solvent", help="Solvent model (gbn2/obc2/gbn1/obc1).")] = "gbn2",
          maxiter: Annotated[int, typer.Option("--maxiter", help="Max. iteration")] = 5000,
          tolerance: Annotated[float, typer.Option("--tolerance", help="Tolerance")] = 0.1,
          workdir: Annotated[str, typer.Option("--workdir", help="Working directory for the simulation")] = ".",
          platform: Annotated[str, typer.Option("--platform", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU)")] = "CUDA",
          devices: Annotated[str, typer.Option("--devices", help="GPU devices for the simulation (e.g., '0', '0,1')")] = "0",
          ):
    """Build an implicit solvent system and run restrained energy minimization"""
    if ligand:
        smi_files = list(Path.cwd().glob(f"*{ligand}.smi"))
        assert len(smi_files) == 1, "more than one .smi files found. use --smiles instead"
        smi_file = smi_files[0]
        smiles = Path(smi_file).read_text().splitlines()[0].strip()
        print(f"reading SMILES from {smi_file}")

    if smiles:
        assert Chem.MolFromSmiles(smiles) is not None, f"Invalid SMILES string: {smiles}"

    vc = ValidComplex(filename)
    
    if smiles:
        vc.fix_ligand(smiles)
    
    vc.assign_ligand_charges(partial_charge_method = partial_charge_method)
    vc.build(ff_ligand = ff_ligand, ff_protein = ff_protein, solvent = solvent)
    md = Relax(vc, 
               maxiter = maxiter, 
               tolerance = tolerance, 
               workdir = workdir, 
               platform = platform, 
               devices = devices)
    md.run()


@app.command()
def build(filename: Annotated[str, typer.Argument(help="Input .pdb or .cif filename")],
          smiles: Annotated[str, typer.Option("--smiles", help="Ligand SMILES")] = "",
          ligand: Annotated[str, typer.Option("--ligand", help="Ligand residue name")] = "",
          ff_ligand: Annotated[str, typer.Option("--ff-ligand", help="Force field for ligand")] = "openff-2.2.1.offxml",
          ff_protein: Annotated[str, typer.Option("--ff-protein", help="Force field for protein")] ="amber/protein.ff14SB.xml",
          ff_water: Annotated[str, typer.Option("--ff-water", help="Force field for water.")] = "amber/tip3p_standard.xml",
          solvent: Annotated[str, typer.Option("--solvent", help="Solvent model (tip3p/gbn2/obc2/gbn1/obc1).")] = "tip3p",
          box_padding: Annotated[float, typer.Option("--box-padding", help="Box padding in Nanometer.")] = 1.0,
          salt_conc: Annotated[float, typer.Option("--salt-conc", help="Salt concentration in Molar.")] = 0.15,
          positive_ion: Annotated[str, typer.Option("--positive-ion", help="Positive ion type.")] = "Na+",
          negative_ion: Annotated[str, typer.Option("--negative-ion", help="Negative ion type.")] = "Cl-",
          h_mass_factor: Annotated[float, typer.Option("--h-mass-factor", help="Hydrogen mass factor.")] = 3.0,
          partial_charge_method: Annotated[str, typer.Option("--partial-charge-method", help="Partial charge method for ligand.")] = "am1bcc",
          ):
    """Build MD system with implicit/explicit water box"""
    if ligand:
        smi_files = list(Path.cwd().glob(f"*{ligand}.smi"))
        assert len(smi_files) == 1, "more than one .smi files found. use --smiles instead"
        smi_file = smi_files[0]
        smiles = Path(smi_files[0]).read_text().splitlines()[0].strip()
        print(f"reading SMILES from {smi_file}")

    if smiles:
        assert Chem.MolFromSmiles(smiles) is not None, f"Invalid SMILES string: {smiles}"

    vc = ValidComplex(filename)

    if smiles:
        vc.fix_ligand(smiles)

    # # if _ligand.sdf file exists, bypass recomuting AM1-BCC charges and fixing ligand
    # ligand_sdf_path = vc.workdir / f"{vc.prefix}_ligand.sdf"
    # if ligand_sdf_path.exists():
    #     vc.load_ligand_charges(filename= ligand_sdf_path.as_posix())

    vc.assign_ligand_charges(partial_charge_method = partial_charge_method)
    vc.save_protein()
    vc.build(ff_ligand = ff_ligand,
            ff_protein = ff_protein,
            ff_water = ff_water,
            solvent = solvent,
            box_padding = box_padding,
            salt_conc = salt_conc,
            positive_ion = positive_ion,
            negative_ion = negative_ion,
            h_mass_factor = h_mass_factor)


@app.command()
def equi(filename: Annotated[str, typer.Argument(help="Filename (.pdb, .pdb.gz, .cif, .cif.gz) for prefix")],
         desmond: Annotated[bool, typer.Option("--desmond", help="Use Desmond-like protocol")] = False,
         temperature: Annotated[float, typer.Option("--temperature", help="Temperature in Kelvin")] = 300.0,
         pressure: Annotated[float, typer.Option("--pressure", help="Pressure in Bar")] = 1.0,
         workdir: Annotated[str, typer.Option("--workdir", help="Working directory for the simulation")] = ".",
         platform: Annotated[str, typer.Option("--platform", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU)")] = "CUDA",
         devices: Annotated[str, typer.Option("--devices", help="GPU devices for the simulation (e.g., '0', '0,1')")] = "0",
         ):
    """Run multi-stage equilibration MD"""
    if desmond:
        md = Desmond(filename,
                     temperature= temperature,
                     pressure= pressure,
                     workdir= workdir,
                     platform= platform,
                     devices= devices
                     )
    else:
        md = Equilibrium(filename,
                         temperature= temperature,
                         pressure= pressure, 
                         workdir= workdir, 
                         platform= platform, 
                         devices= devices)
    md.run()


@app.command()
def prod(filename: Annotated[str, typer.Argument(help="Filename (.pdb, .pdb.gz, .cif, .cif.gz) for prefix")],
         temperature: Annotated[float, typer.Option("--temperature", help="Temperature in Kelvin")] = 300.0,
         pressure: Annotated[float, typer.Option("--pressure", help="Pressure in Bar")] = 1.0,
         time: Annotated[float, typer.Option("--time", help="Simulation time in ns")] = 10.0,
         timestep: Annotated[float, typer.Option("--timestep", help="Simulation timestep in fs")] = 2.0,
         hmr: Annotated[bool, typer.Option("--hmr/--no-hmr", help="Use HMR (Hydrogen Mass Repartitioning).")] = True,
         state_data_interval: Annotated[float, typer.Option("--state-data-interval", help="State data interval in ps")] = 100.0,
         trajectory_interval: Annotated[float, typer.Option("--state-data-interval", help="Trajectory interval in ps")] = 100.0,
         checkpoint_interval: Annotated[float, typer.Option("--state-data-interval", help="Checkpoint interval in ps")] = 100.0,
         workdir: Annotated[str, typer.Option("--workdir", help="Working directory for the simulation")] = ".",
         platform: Annotated[str, typer.Option("--platform", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU)")] = "CUDA",
         devices: Annotated[str, typer.Option("--devices", help="GPU devices for the simulation (e.g., '0', '0,1')")] = "0",
         ):
    """Run production MD"""
    md = Production(filename,
                    temperature= temperature,
                    pressure= pressure,
                    time= time,
                    timestep= timestep,
                    hmr= hmr,
                    state_data_interval= state_data_interval,
                    trajectory_interval= trajectory_interval,
                    checkpoint_interval= checkpoint_interval,
                    workdir= workdir, 
                    platform= platform, 
                    devices= devices)
    md.run()


if __name__ == '__main__':
    app()
