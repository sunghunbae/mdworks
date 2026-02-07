from mdworks.protocol import Equilibrium

md = Equilibrium('2RAP.cif',
                 temperature= 300.0,
                 pressure= 1.0, 
                 workdir="small_G_protein", 
                 platform= "CUDA", 
                 devices= "1")
md.run()