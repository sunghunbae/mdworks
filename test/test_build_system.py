from mdworks import ValidComplex


vc = ValidComplex("2RAP.cif", workdir="small_G_protein")

# fix ligand with the target SMILES of GTP
smiles = 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N'
vc.fix_ligand(smiles)

# apply AM1BCC charges
vc.assign_ligand_charges()

vc.save_protein()
vc.save_ligand()
vc.build()
