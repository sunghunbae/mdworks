## Install

```sh
$ mamba create -n openmd python=3.13 cuda-version=13.x openmmforcefields pdbfixer openmm
```

## Usage

```py
from openmd import ValidComplex
from openmd.protocol import UDesmond

vc = ValidComplex('protein_ligand_complex.cif')

# fix ligand stereochemistry
vc.fix_ligand(`target_SMILES`)

# am1bcc charges
vc.assign_ligand_charges()

# build openmm system
vc.build_system()

# run unbiased md emulating desmond protocol
md = UDesmond(vc)
md.run()
```

## Multi-Stage MD Simulation

| Stage               | temp (K) | posres (kcal/mol/A**2) | time (ps) | timestep (fs) | 
| ------------------- | -------- | ---------------------- | --------- | ------------- |
| Energy Minimization |        | 50 |     |  |
| NVT cold            | 10     | 50 | 100 | 1 |
| NVT heating         | 10-300 | 10 | 12 | 2 |
| NPT posres          | 300    | 2  | 12 | 2 |
| NPT unrestrained    | 300    | 0  | 24 | 2 |
| Production          | 300    | 0  | user | 2 |

## Schrodinger Desmond Protocol

1. Energy Minimization
1. Brownian Dynamics NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 100ps, k=50
1. NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 12ps, k=50
1. NPT, T = 10 K, and restraints on solute heavy atoms, 12ps, k=50
1. NPT and restraints on solute heavy atoms, 12ps, k=50
1. NPT and no restraints, 24ps 

| Stage               | temp (K) | posres (kcal/mol/A**2) | time (ps) | timestep (fs) | 
| ------------------- | -------- | ---------------------- | --------- | ------------- |
| Energy Minimization |          | 50 |      |   |
| Brownian            | 10       | 50 | 100  | 1 |
| NVT cold            | 10       | 50 | 12   | 2 | 
| NPT cold            | 10       | 10 | 12   | 2 |
| NPT annealing       | 10-300   | 2  | 12   | 2 |
| NPT unrestrained    | 300      | 0  | 24   | 2 |
| Production          | 300      | 0  | user | 2 |


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

