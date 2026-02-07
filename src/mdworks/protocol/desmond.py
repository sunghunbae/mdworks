__all__ = ['Desmond',]


"""Unbiased Classical Molecular Dynamics"""

from pathlib import Path
from openmm import app, unit

from openmmtools.testsystems import TestSystem

from .multistage import MultiStage
from ..validcomplex import ValidComplex
from ..utils import setup_logger

import logging

logger = logging.getLogger(__name__)


class Desmond(MultiStage):

    def __init__(self, complex: TestSystem | ValidComplex, quiet: bool = False):
        super().__init__(complex, quiet=quiet)
        setup_logger(logger, self.parent, self.prefix, quiet=quiet)


    def _stage_3(self, stage: int = 3) -> None:
        if self.load_checkpoint(stage):
            return
        if not self.quiet:
            print(f'({stage}) NPT (T= 10 K, posres_k=200 kJ/mol/nm**2, dt=2 fs, t=12 ps)')
        self._change_temperature(temp=10)
        self._change_integrator(temp=10, friction=1, timestep=2)
        self._change_posres(k=200)
        self._add_barostat(frequency=25)
        steps = self._get_steps(ps=12)
        # add reporter
        self.simulation.reporters.append(
            app.StateDataReporter(
                (self.workdir / f'{self.prefix}_{self.tags[stage]}.ene').as_posix(), 
                1000,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True))
        # run simulation
        self.simulation.step(steps)
        # remove reporter
        self.simulation.reporters.pop()
        self.save_checkpoint(stage)


    def _stage_4(self, stage: int = 4) -> None:
        if self.load_checkpoint(stage):
            return
        # NVT warm up
        if not self.quiet:
            print(f'({stage}) NPT heating (T= 10-300 K, posres_k=40 kJ/mol/nm**2, dt=2 fs, t=12 ps)')
        self._change_temperature(temp=10)
        self._change_integrator(temp=10, friction=1, timestep=2)
        self._change_posres(k=40)
        steps = self._get_steps(ps=12)
        # add reporter
        self.simulation.reporters.append(
            app.StateDataReporter(
                (self.workdir / f'{self.prefix}_{self.tags[stage]}.ene').as_posix(), 
                1000,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True))
        # gradually heat up the system
        self._ramp_temperature(steps, 
                               start=10, 
                               end=(self.temperature / unit.kelvin), 
                               tempstep=5)
        # remove reporter
        self.simulation.reporters.pop()
        self.save_checkpoint(stage)

    
    def _stage_5(self, stage: int = 5) -> None:
        if self.load_checkpoint(stage):
            return
        if not self.quiet:
            print(f'({stage}) NPT unrestrained (T= 300 K, posres_k=0 kJ/mol/nm**2, dt=2 fs, t=24 ps)')
        self._change_temperature(temp= (self.temperature / unit.kelvin))
        self._change_integrator(temp= (self.temperature / unit.kelvin), friction=1, timestep=2)
        self._change_posres(k=0)
        steps = self._get_steps(ps=24)
        # add reporter
        self.simulation.reporters.append(
            app.StateDataReporter(
                (self.workdir / f'{self.prefix}_{self.tags[stage]}.ene').as_posix(), 
                1000,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True))
        # run simulation
        self.simulation.step(steps)
        # remove reporter
        self.simulation.reporters.pop()
        self.save_checkpoint(stage)



    def _stage_6(self, stage: int = 6) -> None:
        if not self.quiet:
            print(f"NPT production ({self.time.in_units_of(unit.nanosecond)})")
            print(f'({stage}) NPT production (T= {self.temperature}, dt={self.timestep}, t={self.time})')
            print(f"trajectory {self.trajectory_path}")
            print(
                f"{'trajectory interval':<20}= "
                f"{str(self.trajectory_interval.in_units_of(unit.picosecond)):>12} | "
                f"{self.trajectory_steps:>9} steps")
            print(
                f"{'checkpoint interval':<20}= {str(self.checkpoint_interval):>12} | "
                f"{self.checkpoint_steps:>9} steps"
            )
            print(
                f"{'trajectory frames':<20}= {str(self.trajectory_frames):>9}"
            )
        if not self._resume_prod_simulation():
            self._change_temperature(temp= (self.temperature / unit.kelvin))
            self._change_integrator(temp= (self.temperature / unit.kelvin), 
                                    friction=1.0, 
                                    timestep= (self.timestep / unit.femtosecond))
            self._change_posres(k=0)
            # add reporters
            self.simulation.reporters.append(
                app.CheckpointReporter(
                    self.checkpoint_path.as_posix(), 
                    self.checkpoint_steps))
            self.simulation.reporters.append(
                app.DCDReporter(
                    self.trajectory_path.as_posix(),
                    self.trajectory_steps))
            self.simulation.reporters.append(
                app.StateDataReporter(
                    self.state_data_path.as_posix(), 
                    self.state_data_steps,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True))
            # run simulation
            self.simulation.step(self.steps)


    def run(self,
            temperature: float = 300.0,
            pressure: float = 1.0, 
            timestep : float = 2.0,
            time : float = 10.0,
            hmr: bool = True,
            state_data_interval : float = 100,
            trajectory_interval : float = 100,
            checkpoint_interval : float = 100,
            workdir : str| Path | None = None):
        """Run multi-stage MD simulation.

        Note:
            All arguments apply to the production stage.

        Args:
            temperature (float, optional): temperature in K. Defaults to 300.0.
            pressure (float, optional): pressure in bar. Defaults to 1.0.
            timestep (float, optional): timestep in femtosecond. Defaults to 2.0.
            time (float, optional): time in ns. Defaults to 10.0.
            report_interval (float, optional): report interval time in ps. Defaults to 100.
            trajectory_interval (float, optional): trajectory interval time in ps. Defaults to 100.
            checkpoint_interval (float, optional): checkpoint interval time in ps. Defaults to 100.
            workdir (str | Path | None, optional): output directory. Defaults to None (same directory as input).
        """
        if workdir is None:
            self.workdir = self.parent
        elif isinstance(workdir, str):
            self.workdir = Path(workdir)

        # production stage settings
        self.temperature = temperature * unit.kelvin
        self.pressure = pressure * unit.bar
        self.timestep = timestep * unit.femtoseconds
        self.time = time * unit.nanoseconds
        self.hmr = hmr
        if hmr:
            self.timestep = 4.0 * unit.femtoseconds
        else:
            self.timestep = timestep * unit.femtoseconds
        self.steps = int(self.time / self.timestep + 0.5)
        
        # state data
        self.state_data_path = self.workdir / f'{self.prefix}_{self.tags[-1]}.ene'
        self.state_data_interval = state_data_interval * unit.picoseconds
        self.state_data_steps = int(self.state_data_interval / self.timestep + 0.5)

        # checkpoint
        self.checkpoint_path = self.workdir / f'{self.prefix}_{self.tags[-1]}.cpt'
        self.checkpoint_interval = checkpoint_interval * unit.picoseconds
        self.checkpoint_steps = int(self.checkpoint_interval / self.timestep + 0.5)

        # trajectory
        self.trajectory_path = self.workdir / f'{self.prefix}_{self.tags[-1]}.dcd'
        self.trajectory_interval = trajectory_interval * unit.picoseconds
        self.trajectory_steps = int(self.trajectory_interval / self.timestep + 0.5)
        self.trajectory_frames = int(self.time / self.trajectory_interval + 0.5)

        # log
        self.log_path = self.workdir / f'{self.prefix}.log'

        self.tags = [
            '0_min', 
            '1_brownian', 
            '2_nvt_cold', 
            '3_npt_cold', 
            '4_npt_warm', 
            '5_npt_free', 
            '6_prod',
            ]
        
        stage_settings = {
            1 : {
                'description': 'NVT Brownian Dynamics (T= 10 K, posres_k=1000 kJ/mol/nm**2, dt=1 fs, t=100 ps',
                'T': 10.0,
                'k': 1000.0,
                'dt': 1.0,
                'ps': 100.0,
                'friction': 50.0,
            },
            2 : {
                'description': 'NVT (T= 10 K, posres_k=1000 kJ/mol/nm**2, dt=2 fs, t=12 ps)',
                'T': 10.0,
                'k': 1000.0,
                'dt': 2.0,
                'ps': 12.0,
                'friction': 1.0,
            },
            3: {
                'description': 'NPT (T= 10 K, posres_k=200 kJ/mol/nm**2, dt=2 fs, t=12 ps)',
                'T': 10.0,
                'k': 200.0,
                'dt': 2.0,
                'ps': 12.0,
                'friction': 1.0,
            },
            4: {
                'description': 'NPT heating (T= 10-300 K, posres_k=40 kJ/mol/nm**2, dt=2 fs, t=12 ps)',
                'T': 10.0,
                'k': 40.0,
                'dt': 2.0,
                'ps': 12.0,
                'friction': 1.0,
            }, 

        }
        

        if self.load_checkpoint(stage=5):
            self._stage_6()

        elif self.load_checkpoint(stage=4):
            self._stage_5()
            self._stage_6()

        elif self.load_checkpoint(stage=3):
            self._stage_4()
            self._stage_5()
            self._stage_6()

        elif self.load_checkpoint(stage=2):
            self._stage_3()
            self._stage_4()
            self._stage_5()
            self._stage_6()
        
        elif self.load_checkpoint(stage=1):
            self._stage_NVT_cold()
            self._stage_3()
            self._stage_4()
            self._stage_5()
            self._stage_6()
        
        elif self.load_checkpoint(stage=0):
            self._stage_NVT_cold(stage=1, **stage_settings[1])
            self._stage_NVT_cold(stage=2, **stage_settings[2])
            self._stage_3()
            self._stage_4()
            self._stage_5()
            self._stage_6()
        
        else:
            self._stage_energy_minimization(stage=0)
            self._stage_NVT_cold(stage=1, **stage_settings[1])
            self._stage_NVT_cold(stage=2, **stage_settings[2])
            self._stage_NVT_cold(stage=3, **stage_settings[3])
            self._stage_NPT_warm(stage=4, **stage_settings[4])
            # 
            self._stage_4()
            # NPT unrestrained (T= 300 K, posres_k=0 kJ/mol/nm**2, dt=2 fs, t=24 ps)
            self._stage_5()
            self._stage_6()
        
        # Save final state
        if not self.quiet:
            print(
                f"\nSimulation complete!"
                f"\nTrajectory saved to {self.prefix}.dcd"
                f"\nThe last structure saved to {self.prefix}_final.pdb"
                )
        
        with open(workdir / f"{self.prefix}_final.pdb", "w") as f:
            positions = self.simulation.context.getState(getPositions=True).getPositions()
            app.PDBFile.writeFile(self.topology, positions, f)
