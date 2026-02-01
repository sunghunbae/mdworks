# Install

## Pixi

```sh
# Install pixi

$ curl -fsSL https://pixi.sh/install.sh | sh
```

Please check out [Pixi installation](https://pixi.prefix.dev/latest/installation/) for more details.

## Mdworks

```sh
$ git clone https://github.com/sunghunbae/mdworks.git
$ cd mdworks
$ pixi install
```

# Jupyter Notebook

1. Add `mdworks` environment to JupyterLab

```sh
python -m ipykernel install --user --name='mdworks'
```

2. Start the Jupyter lab
```sh
jupyter lab
```

# Usage

```py
from openmd import ValidComplex
from openmd.protocol import Equilibrium

vc = ValidComplex('protein_ligand_complex.cif')

# fix ligand stereochemistry
vc.fix_ligand(`target_SMILES`)

# am1bcc charges
vc.assign_ligand_charges()

# build openmm system
vc.build_system()

# run unbiased md emulating desmond protocol
md = Equilibrium(vc)
md.run()
```

## Multi-stage Equilibrium Protocol

| Stage               | temperature (K) | posres (kJ/mol/nm**2) | friction (1/ps) | time (ps) | timestep (fs) | 
| ------------------- | --------------- | --------------------- | --------------- | --------- | ------------- |
| Energy Minimization |                 |                 1000  |                 |           |   |
| NVT cold            | 10              |                 1000  |              5  |      100  | 1 |
| NVT warm            | 10 -> 300       |                 1000  |              1  |      145  | 2 |
| NPT posres          | 300             |            1000 -> 0  |              1  |      300  | 2 |
| NPT free            | 300             |                    0  |              1  |      500  | 2 |
| NPT production      | 300             |                    0  |              1  |     user  | 2 or 4 (HMR) |

## Desmond-like Equilibrium Protocol

1. Energy Minimization
1. Brownian Dynamics NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 100ps, k=50
1. NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 12ps, k=50
1. NPT, T = 10 K, and restraints on solute heavy atoms, 12ps, k=50
1. NPT and restraints on solute heavy atoms, 12ps, k=50
1. NPT and no restraints, 24ps 

Notes: 

- 50 kcal/mol/A**2 is equal to 20,920 kJ/mol/nm**2 (1 kcal/mol/A**2 = 418.4 kJ/mol/nm**2)
- scale to the typically used positional restraint force constant (1000 kJ/mol/nm**2)


| Stage               | temperature (K) | posres (kJ/mol/nm**2) | friction (1/ps) | time (ps) | timestep (fs) | 
| ------------------- | --------------- | --------------------- | --------------- | --------- | ------------- |
| Energy Minimization |                 |                 1000  |                 |           |   |
| Brownian            | 10              |                 1000  |             50  |      100  | 1 |
| NVT cold            | 10              |                 1000  |              1  |       12  | 2 | 
| NPT cold            | 10              |                  200  |              1  |       12  | 2 |
| NPT warm            | 10 -> 300       |                   40  |              1  |       12  | 2 |
| NPT free            | 300             |                    0  |              1  |       24  | 2 |
| NPT production      | 300             |                    0  |              1  |     user  | 2 or 4 (HMR) |


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

