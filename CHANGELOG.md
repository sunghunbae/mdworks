## 0.12.0
- added CLI `equilibrate` to run equilibrium MD simulation
  
## 0.11.1
- added implicit solvent option (`gbn2`, `obc2`)
- Zn, Mg and other ions are not supported by implicit solvent model and requires explicit solvent model
- Hydrogen mass partioning is only for explicit solvent model
  
## 0.10.0
- added CLI `guess_smiles` to get SMILES string from a ligand PDB file

## 0.9.0
- added CLI `ready` to prepare receptor which isolates and merges ligand after processing

## 0.2.0
- added CustomMinimizationReporter

## 0.1.0
- ValidComplex class
    - Handles protein-ligand complex in .cif format
    - Protein only in .cif format
    - Issues with OpenFold3
        - OpenFold3 generates flat geometries instead of tetrahedral for undefined chiral center(s)
        - OpenFold3 does not generate hydrogen atoms
    - Issues with OpenEye Spruce
        - Spruce cannot assign either (S) nor (R) to these flat carbon atoms
        - Spruce arbitrarily assigns double bond when inferring structures from coordinates
    - Issue with OpenForceField `Molecule.from_pdb_and_smiles()`
        - not robust enough
- emulating Schrodinger Desmond default protocol with OpenMM
    - Brownian dynamics after energy minimization
    - NPT at 10 K
