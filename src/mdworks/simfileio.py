__all__ = ['SimFileIO',]

import logging
import gzip
import io

from pathlib import Path
from rdkit import Chem

try:
    from openmm import app, XmlSerializer
    from openff.toolkit.topology.molecule import Molecule
except ImportError:
    raise ImportError("install openmm and openff-toolkit from conda-forge.\n")

from .utils import setup_logger


logger = logging.getLogger(__name__)


class SimFileIO:
    def __init__(self):
        setup_logger(logger, self.workdir, self.prefix, quiet=self.quiet)


    def save_protein(self, filename: str = "") -> None:
        """Save protein to a PDB file.

        Returns:
            None
        """
        if not filename:
            filename = self.workdir / f'{self.prefix}_protein.pdb'

        filename = Path(filename)
        
        with open(filename, "w") as f:
            app.PDBFile.writeFile(
                self.protein_modeller.topology,
                self.protein_modeller.positions,
                f,
                keepIds=True
            )


    def save_ligand(self, filename: str = "") -> None:
        """Save the optimized (charged) ligand to an SDF file.

        Returns: 
            None
        """
        if not self.rdmolH:
            raise ValueError("we may need to fix the ligand first. use fix_ligand()")
        
        if not filename:
            filename = self.workdir / f'{self.prefix}_ligand.sdf'
        
        filename = Path(filename)

        if len(self.mem_ligand_charges.getvalue()) > 0:
            self.off_mol.to_file(filename, file_format='sdf')
        else:
            off_mol = Molecule.from_rdkit(self.rdmolH)
            off_mol.to_file(filename, file_format='sdf')


    def load_SMILES(self, filename: str = "") -> str:
        """Load SMILES from the ligand SDF file."""
        if filename:
            ligand_sdf_path = Path(filename)
        else:
            ligand_sdf_path = self.workdir / f'{self.prefix}_ligand.sdf'
        smiles = ""
        with Chem.SDMolSupplier(ligand_sdf_path.as_posix()) as supplier:
            for mol in enumerate(supplier):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    break
        return smiles


    def save_complex_string(self) -> str:
        with io.StringIO() as f:
            app.PDBFile.writeFile(self.topology, self.positions, f, keepIds=True)
            logger.info(f"complex saved to io.StringIO()")
            return f.getvalue()


    def load_complex_string(self, pdb_string: str) -> None:
        with io.StringIO(pdb_string) as f:
            pdb = app.PDBFile(f)
            self.topology = pdb.topology
            self.positions = pdb.positions
            logger.info(f"complex loaded from io.StringIO()")


    def save_complex(self, filename: str = "", compress: bool = True) -> None:
        if not filename:
            if self.solvent_implicit:
                filename = self.workdir / f'{self.prefix}_system_implicit.pdb'
            else:
                filename = self.workdir / f'{self.prefix}_system.pdb'
        
        filename = Path(filename)

        if compress:
            with gzip.open(Path(filename).with_suffix(".pdb.gz"), "wt") as f:
                app.PDBFile.writeFile(self.topology, self.positions, f, keepIds=True)
                logger.info(f"complex saved - {filename.with_suffix('.pdb.gz')}")
        else:
            with open(filename, "w") as f:
                app.PDBFile.writeFile(self.topology, self.positions, f, keepIds=True)
                logger.info(f"complex saved - {filename}")


    def load_complex(self, filename: str = "") -> bool:
        if not filename:
            if self.solvent_implicit:
                filename = self.workdir / f'{self.prefix}_system_implicit.pdb'
            else:
                filename = self.workdir / f'{self.prefix}_system.pdb'
        
        filename = Path(filename)
        
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
    

    def save_system(self, filename: str = "", hmr: bool = False, compress: bool = True) -> None:
        # system contains positional restraints (CustomExternalForce)
        if not filename:
            if hmr:
                filename = self.workdir / f"{self.prefix}_system_hmr.xml"
            else:
                if self.solvent_implicit:
                    filename = self.workdir / f"{self.prefix}_system_implicit.xml"
                else:
                    filename = self.workdir / f"{self.prefix}_system.xml"
        
        filename = Path(filename)
        
        if hmr:
            if compress:
                with gzip.open(filename.with_suffix(".xml.gz"), "wt") as f:
                    f.write(XmlSerializer.serialize(self.system_hmr))
                    logger.info(f"system saved - {filename.with_suffix('.xml.gz')}")
            else:
                with open(filename, "w") as f:
                    f.write(XmlSerializer.serialize(self.system_hmr))
                    logger.info(f"system saved - {filename}")
        else:
            if compress:
                with gzip.open(filename.with_suffix(".xml.gz"), "wt") as f:
                    f.write(XmlSerializer.serialize(self.system))
                    logger.info(f"system saved - {filename.with_suffix('.xml.gz')}")
            else:    
                with open(filename, "w") as f:
                    f.write(XmlSerializer.serialize(self.system))
                    logger.info(f"system saved - {filename}")

    
    def load_system(self, filename: str = "", hmr: bool = False) -> bool:
        if not filename:
            if hmr:
                filename = self.workdir / f"{self.prefix}_system_hmr.xml"
            else:
                if self.solvent_implicit:
                    filename = self.workdir / f"{self.prefix}_system_implicit.xml"
                else:
                    filename = self.workdir / f"{self.prefix}_system.xml"
        
        filename = Path(filename)

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