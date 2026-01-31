__all__ = ['Equilibrium',]


"""Multistage Equilibrium Molecular Dynamics"""

from pathlib import Path
from functools import partial
from openmm import app, unit
from openmmtools.testsystems import TestSystem

from mdworks.multistage import MultiStage
from mdworks.validcomplex import ValidComplex
from mdworks.utils import setup_logger

import logging

logger = logging.getLogger(__name__)


class Equilibrium(MultiStage):

    def __init__(self, 
                 complex: TestSystem | ValidComplex, 
                 workdir: Path | str | None = None,
                 platform: str = 'CUDA', 
                 devices: str = '0',
                 hmr : bool = True,
                 time : float = 10.0,
                 timestep : float = 2.0,
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 state_data_interval : float = 100.,
                 trajectory_interval : float = 100.,
                 checkpoint_interval : float = 100.,
                 quiet: bool = False) -> None:
        """Initialize equilibrium MD.

        Args:
            complex (TestSystem | ValidComplex): _description_
            hmr (bool, optional): whether to use HMR and use 4 fs timestep. Defaults to True.
            time (float, optional): time in nanosecond. Defaults to 10.0.
            timestep (float, optional): timestep in femtosecond. Defaults to 2.0.
            temperature (float, optional): temperature in kelvin. Defaults to 300.0.
            pressure (float, optional): pressure in bar. Defaults to 1.0.
            state_data_interval (float, optional): state data interval time in ps. Defaults to 100.
            trajectory_interval (float, optional): trajectory interval time in ps. Defaults to 100.
            checkpoint_interval (float, optional): checkpoint interval time in ps. Defaults to 100.
        """
        super().__init__(complex, workdir, platform, devices, quiet)

        self.time = time * unit.nanoseconds
        self.hmr = hmr
        if hmr:
            timestep = 4.0
        self.timestep = timestep * unit.femtoseconds
        self.temperature = temperature * unit.kelvin
        self.pressure = pressure * unit.bar
        
        self.stages = [
            {'tag': '0_min', 'maxiter': 5000, 'tolerance': 0.1, 'interval': 10},
            {
                'tag': '1_nvt_cold',
                'description': 'NVT 100 ps at 10 K with positional restraints and a high friction coefficient',
                't': (100., 1.0),
                'T': 10,
                'k': 1000,
                'friction': 5, 
                'interval': 1000 },
            {
                'tag': '2_nvt_warm',
                'description': f'NVT 145 ps with heating gradually while maintaining positional restraints',
                't': (145., 1.0),
                'T': (10., temperature, 10.),
                'k': 1000.,
                'friction': 1,
                'interval': 1000 },
            {
                'tag': '3_npt_posres',
                'description': f'NPT 300 ps at {temperature} K with releasing positional restraints gradually',
                't': (300., 1.0),
                'T': temperature,
                'k': (1000., 0, -20),
                'friction': 1,
                'frequency': 50,
                'interval': 1000 },
            {
                'tag': '4_npt_free',
                'description': f'NPT 500 ps at {temperature} K without positional restraints',
                't': (500., 2.0),
                'T': temperature,
                'k': 0.0,
                'friction': 1,
                'frequency': 50,
                'interval': 1000 },
            {
                'tag': '5_prod',
                'description': f'NPT {time} ns at {temperature} K (production)',
                't': (time * 1000., timestep),
                'T': temperature,
                'k': 0.0,
                'friction': 1,
                'frequency': 50,
                'state_data_interval': state_data_interval,
                'checkpoint_interval': checkpoint_interval,
                'trajectory_interval': trajectory_interval,
             },
        ]
        
        self.stage_partials = [
            partial(self._stage_energy_minimization, stage=0),
            partial(self._stage_NVT_cold, stage=1),
            partial(self._stage_NVT_warm, stage=2),
            partial(self._stage_NPT_posres, stage=3),
            partial(self._stage_NPT_free, stage=4),
            partial(self._stage_NPT_prod, stage=5),
        ]

        assert len(self.stage_partials) == len(self.stages)

        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)


    def run(self):
        """Run multi-stage MD simulation.

        Note:
            all arguments apply to the production stage.
            workdir (str | Path | None, optional): output directory. Defaults to None (same directory as input).
        """
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

        with open(self.workdir / f"{self.prefix}_FINAL.pdb", "w") as f:
            app.PDBFile.writeFile(self.topology, self.positions, f)

        # create the empty file to mark completion
        (self.workdir / f"{self.prefix}_DONE").touch(exist_ok=True)

        logger.info(f"Simulation complete!")
        logger.info(f"Trajectory saved to {self.prefix}.dcd")
        logger.info(f"The final structure has been written to {self.prefix}_FINAL.pdb")
