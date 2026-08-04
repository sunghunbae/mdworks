import shutil
import cyclopts
import mdworks.mmcif as mmcif

from mdworks import Editor, ReadyPipeline, ValidComplex 
from mdworks.protocol import Relax, Equilibrium, Desmond, Production


def _get_version() -> str:
    from mdworks import __version__
    return __version__


# Cyclopts ships a built-in --version handler, so there's no need for a
# manual version_callback / @app.callback() the way Typer required.

app = cyclopts.App(name="mdworks", help="mdworks", version=_get_version)

# def command(posistional_only, /, standard(both), *, key_words_only):

@app.command
def cifinfo(filename: str, /):
    """Get information from mmCIF

    Parameters
    ----------
    filename: str
        Input .cif filename.
    """
    mmcif.info(filename)


@app.command
def cif2seq(filename: str, /):
    """Get missing residue(s) and sequence from mmCIF

    - Full sequence
    - Coordinate sequence with missing residues (`-`)

    Parameters
    ----------
    filename: str
        Input .cif filename.
    """
    from Bio import Align

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


@app.command
def cif2pdb(filename: str, /):
    """Convert mmCIF to PDB

    Parameters
    ----------
    filename: str
        Input .cif filename.
    """
    print(f'converting {filename} to .pdb')
    Editor.load(filename).write(format='pdb')


@app.command
def pdb2cif(filename: str, /):
    """Convert PDB to mmCIF

    Parameters
    ----------
    filename: str
        Input .cif filename.
    """
    print(f'converting {filename} to .cif')
    Editor.load(filename).write(format='cif')


@app.command
def peek(filename: str, /):
    """Peek and show summary of model(s) and chain(s)

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    """
    Editor.load(filename).summary()


@app.command
def delete(filename: str, selection: str = "", /, *, tag: str = 'del', invert: bool = False):
    """Delete chain(s) and residue(s)

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    selection: str
        Select chain:residues to remove: ex. A:1-30,A:200-230,B:1-10,C,D:1
    tag: str
        Output tag
    invert: bool
        Invert selection
    """
    Editor.load(filename).select(expr=selection).delete(invert=invert).write(tag=tag,compress=True)


@app.command
def reorder(filename: str, /, *, tag: str = 'ord'):
    """Reorder chain(s) by chain id

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    tag: str
        Output tag
    """
    Editor.load(filename).reorder_chains().write(tag=tag)


@app.command
def rename(filename: str, subs: str, /, *, tag: str = 'ren'):
    """Rename chain id(s) and residue(s)

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    subs: str
        Rename chain id or residues seqid. ex. A/B,B/C (A->B, B->C), A:1/A:100 (A:1 -> A:100)
    tag: str
        Output tag
    """
    Editor.load(filename).rename(subs= subs).write(tag=tag)


@app.command
def split(filename: str, /):
    """Split and write individual model(s)

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    """
    Editor.load(filename).write(split=True)


@app.command
def ready(
    filename: str,
    /,
    *,
    ligand: str = "",
    keep_waters: bool = False,
    separate_hetgens: bool = False,
    zinc: bool = True,
    keep_terminals: bool = True,
    obabel: str | None = shutil.which("obabel"),
    ph: float = 7.4,
):
    """Ready the structure for relax

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename.
    ligand: str
        Ligand residue name
    keep_waters: bool
        Keep waters or not
    separate_hetgens: bool
        Separate chains for hetgens (Zn, ligand, ...)
    zinc: bool
        Handle zinc related issue
    keep_terminals: bool
        Keep input N/C-terminals or add missing residues at N/C-terminals
    obabel: str
        Path to the obabel executable
    ph: float
        Target pH for protonation
    """
    ReadyPipeline(
        in_file= filename,
        ligand_resname= ligand,
        keep_waters= keep_waters,
        separate_hetgens= separate_hetgens,
        zinc= zinc,
        keep_terminals= keep_terminals,
        obabel= obabel,
        target_ph= ph,
    ).run()


@app.command
def relax(
    filename: str,
    /,
    *,
    charge: str = "nagl",
    ff_ligand: str = "openff-2.2.1.offxml",
    ff_protein: str = "amber/protein.ff14SB.xml",
    solvent: str = "gbn2",
    posres: float = 1000.0,
    maxiter: int = 5000,
    tolerance: float = 0.1,
    workdir: str = ".",
    platform: str = "CUDA",
    devices: str = "0",
):
    """Relax the readied system by restrained energy minimization in implicit solvent

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename
    smiles: str
        Ligand SMILES
    ligand: str
        Ligand residue name
    charge: str
        Partial charge method for ligand [nagl/am1bcc/gasteiger/mmff94/mmff94s].
    ff_ligand: str
        Force field for ligand
    ff_protein: str
        Force field for protein
    solvent: str
        Solvent model (gbn2/obc2/gbn1/obc1).
    posres: float
        Force constant for positional restraints
    maxiter: int
        Max. iteration
    tolerance: float
        Tolerance
    workdir: str
        Working directory for the simulation
    platform: str
        Platform for the simulation (e.g., CUDA, OpenCL, CPU)
    devices: str
        GPU devices for the simulation (e.g., '0', '0,1')
    """
    vc = ValidComplex(filename)
    vc.assign_ligand_charges(partial_charge_method=charge)
    # if we use 'import' for ligand charges, ligand chain id and residue number are unknown and will be
    # set to 'UNK' and '0', respectively.
    vc.build(ff_ligand=ff_ligand, ff_protein=ff_protein, solvent=solvent, posres=posres)
    em = Relax(
        vc,
        maxiter=maxiter,
        tolerance=tolerance,
        workdir=workdir,
        platform=platform,
        devices=devices,
    )
    em.run()


@app.command
def build(
    filename: str,
    /,
    *,
    charge: str = "nagl",
    ff_ligand: str = "openff-2.2.1.offxml",
    ff_protein: str = "amber/protein.ff14SB.xml",
    ff_water: str = "amber/tip3p_standard.xml",
    solvent: str = "tip3p",
    box_padding: float = 1.0,
    salt_conc: float = 0.15,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    h_mass_factor: float = 3.0,
    
):
    """Build a MD system with explicit/implicit water box

    Parameters
    ----------
    filename: str
        Input .pdb or .cif filename
    charge: str
        Partial charge method for ligand [nagl/am1bcc/gasteiger/mmff94/mmff94s].
    ff_ligand: str
        Force field for ligand
    ff_protein: str
        Force field for protein
    ff_water: str
        Force field for water.
    solvent: str
        Solvent model (tip3p/gbn2/obc2/gbn1/obc1).
    box_padding: float
        Box padding in Nanometer.
    salt_conc: float
        Salt concentration in Molar.
    positive_ion: str
        Positive ion type.
    negative_ion: str
        Negative ion type.
    h_mass_factor: float
        Hydrogen mass factor.
    """    

    vc = ValidComplex(filename)
    vc.assign_ligand_charges(partial_charge_method= charge)
    vc.save_protein()
    vc.build(
        ff_ligand=ff_ligand,
        ff_protein=ff_protein,
        ff_water=ff_water,
        solvent=solvent,
        box_padding=box_padding,
        salt_conc=salt_conc,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        h_mass_factor=h_mass_factor,
    )


@app.command
def equi(
    filename: str,
    /,
    *,
    desmond: bool = False,
    temperature: float = 300.0,
    pressure: float = 1.0,
    workdir: str = ".",
    platform: str = "CUDA",
    devices: str = "0",
):
    """Run multi-stage equilibration MD

    Parameters
    ----------
    filename: str
        Filename (.pdb, .pdb.gz, .cif, .cif.gz) for prefix
    desmond: bool
        Use Desmond-like protocol
    temperature: float
        Temperature in Kelvin
    pressure: float
        Pressure in Bar
    workdir: str
        Working directory for the simulation
    platform: str
        Platform for the simulation (e.g., CUDA, OpenCL, CPU)
    devices: str
        GPU devices for the simulation (e.g., '0', '0,1')
    """
    if desmond:
        md = Desmond(
            filename,
            temperature=temperature,
            pressure=pressure,
            workdir=workdir,
            platform=platform,
            devices=devices,
        )
    else:
        md = Equilibrium(
            filename,
            temperature=temperature,
            pressure=pressure,
            workdir=workdir,
            platform=platform,
            devices=devices,
        )
    md.run()


@app.command
def prod(
    filename: str,
    /,
    *,
    temperature: float = 300.0,
    pressure: float = 1.0,
    time: float = 10.0,
    timestep: float = 2.0,
    hmr: bool = True,
    state_data_interval: float = 100.0,
    trajectory_interval: float = 100.0,
    checkpoint_interval: float = 100.0,
    workdir: str = ".",
    platform: str = "CUDA",
    devices: str = "0",
):
    """Run production MD

    Parameters
    ----------
    filename: str
        Filename (.pdb, .pdb.gz, .cif, .cif.gz) for prefix
    temperature: float
        Temperature in Kelvin
    pressure: float
        Pressure in Bar
    time: float
        Simulation time in ns
    timestep: float
        Simulation timestep in fs
    hmr: bool
        Use HMR (Hydrogen Mass Repartitioning).
    state_data_interval: float
        State data interval in ps
    trajectory_interval: float
        Trajectory interval in ps
    checkpoint_interval: float
        Checkpoint interval in ps
    workdir: str
        Working directory for the simulation
    platform: str
        Platform for the simulation (e.g., CUDA, OpenCL, CPU)
    devices: str
        GPU devices for the simulation (e.g., '0', '0,1')
    """
    md = Production(
        filename,
        temperature=temperature,
        pressure=pressure,
        time=time,
        timestep=timestep,
        hmr=hmr,
        state_data_interval=state_data_interval,
        trajectory_interval=trajectory_interval,
        checkpoint_interval=checkpoint_interval,
        workdir=workdir,
        platform=platform,
        devices=devices,
    )
    md.run()


if __name__ == '__main__':
    app()
