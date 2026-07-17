__all__ = ['Relax',]


"""Restrained Energy Minimization"""

from pathlib import Path
from functools import partial

from openmm import app, unit
from openmmtools.testsystems import TestSystem

from ..utils import setup_logger
from ..validcomplex import ValidComplex
from .multistage import MultiStage

import logging
import gzip


logger = logging.getLogger(__name__)


class Relax(MultiStage):
    def __init__(self, 
                 complex: TestSystem | ValidComplex, 
                 workdir: Path | str | None = None,
                 platform: str = 'CUDA', 
                 devices: str = '0',
                 maxiter: int = 5000,
                 tolerance: float = 0.1,
                 interval: int = 10,
                 quiet: bool = False) -> None:
        """Restrained Energy Minimization

        Args:
            complex (TestSystem | ValidComplex): molecular system to run equilibrium MD.
            workdir (Path | str | None, optional): output path. Defaults to None.
            platform (str, optional): openmm platform. Defaults to 'CUDA'.
            devices (str, optional): CUDA devices. Defaults to '0'.
            temperature (float, optional): temperate in kelvin. Defaults to 300.0.
            pressure (float, optional): pressure in bar. Defaults to 1.0.
            quiet (bool, optional): whether to show logging info. Defaults to False.
        """
        super().__init__(complex, workdir, platform, devices, quiet)

        self.stages = [
            {
                'tag': 'relax',
                'description': 'Restrained energy minimization', 
                'maxiter': maxiter, # or 0 for convergence
                'tolerance': tolerance, # default 10
                'interval': interval },
        ]
        
        self.stage_partials = [
            partial(self._stage_energy_minimization, stage=0),
        ]

        assert len(self.stage_partials) == len(self.stages)
        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)
        

    def run(self) -> None:
        """Run Restrained Energy Minimization."""
        # check the last stage checkpoint
        # if no checkpoint exists, stage_cpt_idx ends up with -1
        stage_cpt_idx = len(self.stage_partials)-1
        while stage_cpt_idx >= 0 and not self.load_checkpoint(stage_cpt_idx):
            stage_cpt_idx -= 1
        
        # restart from the last stage checkpoint
        for i, stage_sim in enumerate(self.stage_partials):
            if i > stage_cpt_idx:
                stage_sim(**self.stages[i])
        
        # final update
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()

        self.save_complex(filename = f"{self.prefix}_relaxed.pdb", compress=True)
        self.save_ligand(filename= f"{self.prefix}_relaxed.sdf")

        # create the empty file to mark completion
        (self.workdir / f"{self.prefix}_RELAXED").touch(exist_ok=True)

        logger.info(f"Restrained energy minimization complete!")