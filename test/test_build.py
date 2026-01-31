from openmd import ValidComplex, UnbiasedMD
from openmd.protocol import UDesmond, UDefault

#in_file = 'MonteRosa_WO2025090727_6_seed_1011292330_sample_1_model.cif'
#target_smiles = "C[C@@H]1[C@@H](Nc2oc(C34CC(C4)(C(F)(F)F)C3)nn2)CN1c5cc(F)c([C@H]6CCC(NC6=O)=O)c(F)c5"
in_file = '2RAP.cif'
target_smiles = 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N' # GTP

vc = ValidComplex(in_file)
vc.fix_ligand(target_smiles)
vc.assign_ligand_charges()
vc.save_protein()
vc.save_ligand()
vc.build_system()
