__all__ = ['SimFileIO',]

import logging
import gzip

from pathlib import Path

try:
    from openmm import app, XmlSerializer
except ImportError:
    raise ImportError("install openmm from conda-forge.\n")

from .utils import setup_logger


logger = logging.getLogger(__name__)


class SimFileIO:
    def __init__(self):
        setup_logger(logger, self.workdir, self.prefix, quiet=self.quiet)


    def save_complex(self, compress: bool = True) -> None:
        filename = self.workdir / f'{self.prefix}_complex.pdb'
        if compress:
            with gzip.open(filename.with_suffix(".pdb.gz"), "wt") as f:
                app.PDBFile.writeFile(self.topology, self.positions, f, keepIds=True)
                logger.info(f"complex saved - {filename.with_suffix('.pdb.gz')}")
        else:
            with open(filename, "w") as f:
                app.PDBFile.writeFile(self.topology, self.positions, f, keepIds=True)
                logger.info(f"complex saved - {filename}")


    def load_complex(self) -> bool:
        filename = self.workdir / f'{self.prefix}_complex.pdb'
        if filename.exists():
            with open(filename, "r") as f:
                pdb = app.PDBFile(f)
                self.topology = pdb.topology
                self.positions = pdb.positions
                logger.info(f"complex loaded - {filename}")
        elif filename.with_suffix(".pdb.gz").exists():
            with gzip.open(filename.with_suffix(".pdb.gz"), "rt") as f:
                pdb = app.PDBFile(f)
                self.topology = pdb.topology
                self.positions = pdb.positions
                logger.info(f"complex loaded - {filename.with_suffix('.pdb.gz')}")
        else:
            logger.info(f"cannot load complex {filename} or {filename.with_suffix('.pdb.gz')}")
            return False
        return True
    

    def save_system(self, hmr: bool = False, compress: bool = True) -> None:
        # system contains positional restraints (CustomExternalForce)
        if hmr:
            filename = self.workdir / f"{self.prefix}_system_hmr.xml"
            if compress:
                with gzip.open(filename.with_suffix(".xml.gz"), "wt") as f:
                    f.write(XmlSerializer.serialize(self.system_hmr))
                    logger.info(f"system saved - {filename.with_suffix('.xml.gz')}")
            else:
                with open(filename, "w") as f:
                    f.write(XmlSerializer.serialize(self.system_hmr))
                    logger.info(f"system saved - {filename}")
        else:
            filename = self.workdir / f"{self.prefix}_system.xml"
            if compress:
                with gzip.open(filename.with_suffix(".xml.gz"), "wt") as f:
                    f.write(XmlSerializer.serialize(self.system))
                    logger.info(f"system saved - {filename.with_suffix('.xml.gz')}")
            else:    
                with open(filename, "w") as f:
                    f.write(XmlSerializer.serialize(self.system))
                    logger.info(f"system saved - {filename}")

    
    def load_system(self, hmr: bool = False) -> bool:
        if hmr:
            filename = self.workdir / f"{self.prefix}_system_hmr.xml"
        else:
            filename = self.workdir / f"{self.prefix}_system.xml"
        if filename.exists():
            with open(filename, "r") as f:
                self.system = XmlSerializer.deserialize(f.read())
                logger.info(f"system loaded - {filename}")
        elif filename.with_suffix(".xml.gz").exists():
            with gzip.open(filename.with_suffix(".xml.gz"), "rt") as f:
                self.system = XmlSerializer.deserialize(f.read())
                logger.info(f"system loaded - {filename.with_suffix(".xml.gz")}")
        else:
            logger.info(f"cannot load system {filename} or {filename.with_suffix(".xml.gz")}")
            return False
        return True


    def save_integrator(self, compress: bool = True) -> None:
        filename = self.workdir / f"{self.prefix}_integrator.xml"
        if compress:
            with gzip.open(filename.with_suffix(".xml.gz"), "wt") as f:
                f.write(XmlSerializer.serialize(self.integrator))
                logger.info(f"integrator saved - {filename.with_suffix('.xml.gz')}")
        else:
            with open(filename, "w") as f:
                f.write(XmlSerializer.serialize(self.integrator))
                logger.info(f"integrator saved - {filename}")


    def load_integrator(self) -> bool:
        filename = self.workdir / f"{self.prefix}_integrator.xml"
        if filename.exists():
            with open(filename, "r") as f:
                self.integrator = XmlSerializer.deserialize(f.read())
                logger.info(f"integrator loaded - {filename}")
        elif filename.with_suffix(".xml.gz").exists():
            with gzip.open(filename.with_suffix(".xml.gz"), "rt") as f:
                self.integrator = XmlSerializer.deserialize(f.read())
                logger.info(f"integrator loaded - {filename.with_suffix('.xml.gz')}")
        else:
            logger.info(f"cannot load integrator {filename} or {filename.with_suffix('.xml.gz')}")
            return False
        return True


    def save_state(self, tag: str, compress: bool = True) -> None:
        # Save the current state to an XML file (more portable than checkpoints)
        filename = self.workdir / f'{self.prefix}_{tag}.xml'
        if compress:
            with gzip.open(filename.with_suffix('.xml.gz'), 'wt') as f:
                self.simulation.saveState(f)
                logger.info(f"state saved - {filename.with_suffix('.xml.gz')}")
        else:
            with open(filename, 'w') as f:
                self.simulation.saveState(f)
                logger.info(f"state saved - {filename}")


    def load_state(self, tag: str) -> bool:
        filename = self.workdir / f'{self.prefix}_{tag}.xml'
        if filename.exists():
            with open(filename, 'r') as f:
                self.simulation.loadState(f.read())
                logger.info(f"state loaded - {filename}")
        elif filename.with_suffix('.xml.gz').exists():
            with gzip.open(filename.with_suffix('.xml.gz'), 'rt') as f:
                self.simulation.loadState(f.read())
                logger.info(f"state loaded - {filename.with_suffix('.xml.gz')}")
        else:
            logger.info(f"cannot load state {filename} or {filename.with_suffix('.xml.gz')}")
            return False        
        return True