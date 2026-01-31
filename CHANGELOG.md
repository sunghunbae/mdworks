## (2026-01-09) Version 0.1.0
- ValidComplex class
    - Handles protein-ligand complex in .cif format
    - Protein only in .cif format
- emulating Schrodinger Desmond default protocol with OpenMM
    - Brownian dynamics after energy minimization
    - NPT at 10 K
    - NPT gradual heating from 10 K to 300 K
    - Positional restraints (50 -> 10 -> 2 -> 0)

### ToDo
- RunMD class (unbiased.py or std.py)
- energy minization protocol
- brownian md with 2 fs
- workdir 
- rename tags
    min -> _1_min
    brownian -> _2_brownian
    ...

- crambin test (no ligand case)
- select residues for simulation (HOH, MG, and etc.)