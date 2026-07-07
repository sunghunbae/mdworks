__all__ = ['Equilibrium',]


"""Multistage Equilibrium Molecular Dynamics"""

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


class Equilibrium(MultiStage):
    def __init__(self, 
                 complex: TestSystem | ValidComplex, 
                 workdir: Path | str | None = None,
                 platform: str = 'CUDA', 
                 devices: str = '0',
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 quiet: bool = False) -> None:
        """_summary_

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

        self.pressure = pressure * unit.bar
        self.stages = [
            {
                'tag': '0_min',
                'description': 'Energy minimization', 
                'maxiter': 5000, # or 0 for convergence
                'tolerance': 0.1, # default 10
                'interval': 10 },
            {
                'tag': '1_nvt_cold',
                'description': 'NVT with positional restraints and high friction coefficient',
                't': (100., 1.0),
                'T': 10,
                'k': 1000,
                'friction': 5, 
                'interval': 1000 },
            {
                'tag': '2_nvt_warm',
                'description': 'NVT with positional restraints and gradual heating',
                't': (145., 1.0),
                'T': (10., temperature, 10.),
                'k': 1000.,
                'friction': 1,
                'interval': 1000 },
            {
                'tag': '3_npt_posres',
                'description': 'NPT with gradual releasing of positional restraints',
                't': (300., 1.0),
                'T': temperature,
                'k': (1000., 0, -20),
                'friction': 1,
                'frequency': 50,
                'interval': 1000 },
            {
                'tag': '4_npt_free',
                'description': 'NPT without positional restraint',
                't': (500., 2.0),
                'T': temperature,
                'k': 0.0,
                'friction': 1,
                'frequency': 50,
                'interval': 1000 },
        ]
        
        self.stage_partials = [
            partial(self._stage_energy_minimization, stage=0),
            partial(self._stage_NVT_cold, stage=1),
            partial(self._stage_NVT_warm, stage=2),
            partial(self._stage_NPT_posres, stage=3),
            partial(self._stage_NPT_free, stage=4),
        ]

        assert len(self.stage_partials) == len(self.stages)
        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)


    def run(self) -> None:
        """Run multi-stage equilibrium MD simulation."""
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

        outfile = f"{self.prefix}_relaxed.pdb.gz"
        with gzip.open(self.workdir / outfile, "wt") as f:
            app.PDBFile.writeFile(self.topology, self.positions, f)

        # create the empty file to mark completion
        (self.workdir / f"{self.prefix}_RELAXED").touch(exist_ok=True)

        logger.info(f"Equilibration complete!")
        logger.info(f"Structure saved to {outfile}")