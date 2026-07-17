# Streamlines preparation and equilibration of molecular complex for MD simulations

Whether the atomic coordinates come from experimentally determined complex structures or from co-folding AI models, they often require adjustments such as correcting ligand stereochemistry or fixing other structural details needed for molecular dynamics. `mdworks` streamlines this process by making it easy to prepare a valid protein-ligand complex and setup up and run equilibrium MD simulations with OpenMM.

# Install

## pixi

```sh
# For more details, visit https://pixi.prefix.dev/latest/installation/
$ curl -fsSL https://pixi.sh/install.sh | sh
```

## mdworks

```sh
$ git clone https://github.com/sunghunbae/mdworks.git
$ cd mdworks
$ pixi install
```

## Jupyter Notebook

```sh
# Add `mdworks` environment to JupyterLab
$ python -m ipykernel install --user --name='mdworks'

# Start the Jupyter lab
$ jupyter lab
```

# Usage

## Command-line interface workflow

2RAP (The small G protein RAP2A in complex with GTP)

```bash
$ mdworks --help

$ mdworks ready --help

# fix input PDB using PDBFixer and PDB2PQR
$ mdworks ready input.pdb --ligand UNL

# guess SMILES from a ligand PDB
$ mdworks guess input_UNL.pdb

# cut input PDB to reduce system size for MD
$ mdworks cut input_complex.pdb A:1-46,A:288-298

$ mdworks relax --help

# run restrained energy minimization (with implicit solvent)
# partial charges are assigned with AM1-BCC by default
$ mdworks relax input_complex_cut.pdb --smiles `cat input_UNL.smi`

# build a MD system with explicit solvent
$ mdworks build input_complex_cut_relaxed.pdb.gz --smiles `cat input_UNL.smi`

# run multi-stage equilibration MD simulations
$ mdworks equi input_complex_cut_relaxed.pdb.gz

# run production MD simulations
$ mdworks prod --time 5.0 input_complex_cut_relaxed.pdb.gz

# extend production MD simulations
$ mdworks prod --time 10.0 input_complex_cut_relaxed.pdb.gz
# Since 5.0 ns production simulation has been already completed,
# only remaining 5.0 ns production simulation will be conducted.
# The trajectories will be appended to the production .dcd file.
```

## Python package

```py
from mdworks import ValidComplex
from mdworks.protocol import Equilibrium

vc = ValidComplex('protein_ligand_complex.cif')

# fix ligand stereochemistry
vc.fix_ligand(`target_SMILES`)

# am1bcc charges
vc.assign_ligand_charges()

# build openmm system
vc.build()

# run multi-stage equilibrium MD simulations
md = Equilibrium(vc)
md.run()
```

# Protocols

## Multi-stage Equilibrium / Production

| Stage               | Temperature (K) | Posres (kJ/mol/nm**2) | Friction (1/ps) | Time (ps) | Timestep (fs) | Tag | 
| :------------------ | :-------------- | :-------------------- | :-------------- | :-------- | :------------ | :-- |
| Energy Minimization |                 |                 1000  |                 |           |   | _0_min |
| NVT cold            | 10              |                 1000  |              5  |      100  | 1 | _1_nvt_cold |
| NVT warm            | 10 ⟶ 300       |                 1000  |              1  |      145  | 2 | _2_nvt_warm |
| NPT posres          | 300             |            1000 ⟶ 0  |              1  |      300  | 2 | _3_npt_posres |
| NPT free            | 300             |                    0  |              1  |      500  | 2 | _4_npt_free |
| NPT production      | 300             |                    0  |              1  |     user  | 2 or 4 (HMR) | _5_prod |

## Schrodinger Desmond-like Equilibrium

1. Energy Minimization
1. Brownian Dynamics NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 100ps, k=50
1. NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 12ps, k=50
1. NPT, T = 10 K, and restraints on solute heavy atoms, 12ps, k=50
1. NPT and restraints on solute heavy atoms, 12ps, k=50
1. NPT and no restraints, 24ps 

Notes: 

- 50 kcal/mol/A^2 is equal to 20,920 kJ/mol/nm^2 (1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2)
- scale to the typically used positional restraint force constant (1000 kJ/mol/nm^2)


| Stage               | Temperature (K) | Posres (kJ/mol/nm**2) | Friction (1/ps) | Time (ps) | Timestep (fs) | Tag |
| :------------------ | :-------------- | :-------------------- | :-------------- | :-------- | :------------ | :-- |
| Energy Minimization |                 |                 1000  |                 |           |   | _0_min |
| Brownian            | 10              |                 1000  |         **50**  |      100  | 1 | _1_brownian |
| NVT cold            | 10              |                 1000  |              1  |       12  | 2 | _2_nvt_cold |
| NPT cold            | 10              |                  200  |              1  |       12  | 2 | _3_npt_cold |
| NPT warm            | 10 ⟶ 300       |                   40  |              1  |       12  | 2 | _4_npt_warm |
| NPT free            | 300             |                    0  |              1  |       24  | 2 | _5_npt_free |
| NPT production      | 300             |                    0  |              1  |     user  | 2 or 4 (HMR) | _6_prod |


### Brownian MD

Brownian dynamics corresponds to:

- Motion dominated by friction + random force
- Inertia negligible
- Overdamped limit of Langevin dynamics
- Langevin dynamics with very high friction and small timestep
- Use with positional restraints is recommended
- When to use:
    - Initial solvent relaxation
    - Ion placement adjustment
    - Avoids solute distortion
    - Prevents pressure spikes later