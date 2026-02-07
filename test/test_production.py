from mdworks.protocol import Production

md = Production('small_G_protein/2RAP',
                time= 1.0,
                hmr= True,
                platform= "CUDA",
                devices= "1")
md.run()