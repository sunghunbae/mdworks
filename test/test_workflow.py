from mdworks import ValidComplex
from mdworks.protocol import Equilibrium, Production

vc = ValidComplex("2RAP.cif", workdir="small_G_protein")
# fix ligand with the target SMILES of GTP
smiles = 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N'
vc.fix_ligand(smiles)
# apply AM1BCC charges
vc.assign_ligand_charges()
vc.save_protein()
vc.save_ligand()
vc.build()

eq = Equilibrium(vc, temperature= 300.0, pressure= 1.0, workdir="small_G_protein")
eq.run()

# You can start equilibrium MD by defining `prefix` by providing input .cif filename
# eq = Equilibrium('2RAP.cif', temperature= 300.0, pressure= 1.0, workdir="small_G_protein")
# eq.run()

md = Production(eq, time= 1.0, devices= "1")
md.run()

# You can start production MD by defining `prefix` and `workdir` by providing a string
# md = Production('small_G_protein/2RAP', time= 1.0)
# md.run()