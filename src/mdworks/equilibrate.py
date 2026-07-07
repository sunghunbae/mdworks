import argparse
from pathlib import Path
from mdworks.protocol import Equilibrium


def app():
    parser = argparse.ArgumentParser(
        description="Run equilibrium MD simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to the input PDB/MMCIF file.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature for the simulation.")
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure for the simulation.")
    parser.add_argument("--workdir", type=str, default=".", help="Working directory for the simulation.")
    parser.add_argument("--platform", type=str, default="CUDA", help="Platform for the simulation (e.g., CUDA, OpenCL, CPU).")
    parser.add_argument("--devices", type=str, default="1", help="GPU devices for the simulation (e.g., '0', '0,1').")

    args = parser.parse_args()

    assert Path(args.infile).exists(), f"Input file {args.infile} does not exist."

    md = Equilibrium(args.infile,
                 temperature= args.temperature,
                 pressure= args.pressure, 
                 workdir=args.workdir, 
                 platform= args.platform, 
                 devices= args.devices)
    md.run()

    

if __name__ == "__main__":
    app()