__all__ = ['Production',]

from pathlib import Path
from openmm import app, unit, LangevinIntegrator

from ..utils import setup_logger
from .multistage import MultiStage
from .equilibrium import Equilibrium

import logging
import gzip

logger = logging.getLogger(__name__)


class Production(MultiStage):
    def __init__(self, 
                 complex: Equilibrium | Path | str,
                 workdir: Path | str | None = None,
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 platform: str = 'CUDA', 
                 devices: str = '0',
                 hmr : bool = True,
                 time : float = 10.0,
                 timestep : float = 2.0,
                 state_data_interval : float = 100.,
                 trajectory_interval : float = 100.,
                 checkpoint_interval : float = 100.,
                 quiet: bool = False) -> None:
        """Initialize Production Molecular Dynamics.

        Args:
            complex (Equilibrium): multi-stage complex Equilibrium object.
            workdir (Path | str | None, optional): output path. Defaults to None.
            platform (str, optional): openmm platform. Defaults to 'CUDA'.
            devices (str, optional): CUDA devices. Defaults to '0'.
            hmr (bool, optional): whether to use HMR and override timestep and use 4 fs. Defaults to True.
            time (float, optional): time in nanosecond. Defaults to 10.0.
            timestep (float, optional): timestep in femtosecond. Defaults to 2.0.
            state_data_interval (float, optional): state data interval time in ps. Defaults to 100.
            trajectory_interval (float, optional): trajectory interval time in ps. Defaults to 100.
            checkpoint_interval (float, optional): checkpoint interval time in ps. Defaults to 100.
        """
        if isinstance(workdir, Path) or isinstance(workdir, str):
            self.workdir = Path(workdir)
        else:
            if isinstance(complex, Equilibrium):
                self.workdir = complex.workdir
            elif isinstance(complex, Path) or isinstance(complex, str):
                self.workdir = Path(complex).parent
        
        if isinstance(complex, Equilibrium):
            self.prefix = complex.prefix
        elif isinstance(complex, Path) or isinstance(complex, str):
            p = Path(complex)
            self.prefix = p.name.removesuffix("".join(p.suffixes))

        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)

        self.platform = None
        self.properties = {}
        self._set_platform(platform, devices)

        self.topology = None
        self.positions = None
        self.system = None
        self.integrator = None
        self.simulation = None

        if isinstance(complex, Equilibrium):
            self.topology = complex.topology
            self.positions = complex.positions
            self.system = complex.system
            self.integrator = complex.integrator
            self.simulation = complex.simulation
    
        elif isinstance(complex, Path) or isinstance(complex, str):
            # Loading a checkpoint requires identical System/Platform/OpenMM
            if not self.simulation:
                # loads complex, system, integrator and create simulation
                self.load_sim_env()
            # load the latest checkpoint (e.g., xxx_4_npt_free.cpt)
            try:
                filename = sorted(self.workdir.glob(f"{self.prefix}_?_*.cpt"))[-1]
                logger.info(f"loading checkpoint {filename} ..")
                with open(filename, 'rb') as f:
                    self.simulation.context.loadCheckpoint(f.read())
                    # checkpoint does not remember positional restraints
            except:
                logger.error(f"cannot load checkpoint")
                raise

        if isinstance(complex, Equilibrium):
            self.temperature = complex.temperature # unit.Quantity
            self.pressure = complex.pressure # unit.Quantity
            self.friction = complex.friction # unit.Quantity
            self.frequency = complex.frequency
            self.barostat = complex.barostat
        else:
            self.temperature = temperature * unit.kelvin
            self.pressure = pressure * unit.bar
            self.friction = 1.0 / unit.picosecond
            self.frequency = 50.0
            self.barostat = self._add_barostat(frequency= self.frequency)

        self.hmr = hmr
        if hmr:
            timestep = 4.0
        self.timestep = timestep * unit.femtoseconds
        self.time = time * unit.nanoseconds
        self.steps = int(self.time / self.timestep + 0.5)

        self.state_data_interval = state_data_interval * unit.picosecond
        self.checkpoint_interval = checkpoint_interval * unit.picosecond
        self.trajectory_interval = trajectory_interval * unit.picosecond
        self.trajectory_frames = int(self.time / self.trajectory_interval + 0.5)


    def _set_hmr(self) -> None:
        logger.info(f"Switching to HMR system for timestep of 4 fs")

        # self.positions has not been updated since __init__()
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()

        # self.system will be overwritten
        if not self.load_system(hmr=True):
            raise FileNotFoundError(".._system_hmr.xml file is required.")

        self.integrator = LangevinIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
            )

        self.simulation = app.Simulation(
            self.topology,
            self.system,
            self.integrator,
            self.platform,
            self.properties,
            )

        # initialize Simulation from positions ONLY
        self.simulation.context.setPositions(self.positions)

        logger.info(
            f"Simulation re-created with HMR "
            f"Atoms: {self.topology.getNumAtoms()}, "
            f"Residues: {self.topology.getNumResidues()}, "
            f"Chains: {self.topology.getNumChains()}, "
            f"Constraints: {self.simulation.system.getNumConstraints()}"
            )
        self.list_forces()

        # short minimization (mandatory)
        self.simulation.minimizeEnergy(maxIterations=2000)
        logger.info(f"Energy minimized")

        # reinitialize velocities (mandatory)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        logger.info(f"Velocities set from a Maxwell-Boltzmann distribution at {self.temperature}")

        # barostat
        self.barostat = self._add_barostat(frequency= self.frequency)

        # save checkpoint
        filename = (self.workdir / f'{self.prefix}_hmr.cpt').as_posix()
        self.simulation.saveCheckpoint(filename)
        logger.info(f"Checkpoint saved to {filename}")


    def _is_stable(self) -> bool:
        T  = (self.temperature / unit.kelvin)
        dt = (self.timestep / unit.femtosecond)
        self._change_temperature(temperature=T)
        self._change_integrator(temperature=T, friction=1.0, timestep=dt)
        self._change_posres(k=0.0)
        try:
            self.simulation.step(1000)
        except:
            return False
        return True


    def _continue_pre_production(self, steps: int = 1000) -> None:
        logger.info(f"cont. pre-produciton equilibration {steps} steps")
        T = (self.temperature / unit.kelvin)
        self._change_temperature(temperature=T)
        self._change_integrator(temperature=T, friction=1.0, timestep=1.0)
        self._change_posres(k=0.0)
        self.simulation.step(steps)


    def _resume_prod_simulation(self, tag: str) -> bool:
        # similar to load_checkpoint()
        filename = self.workdir / f'{self.prefix}_{tag}.cpt'
        if not filename.exists():
            return False
        
        # restarting from a checkpoint requires identical System/Platform/OpenMM
        if not self.simulation:
            self.load_sim_env()
        
        with open(filename, 'rb') as f:
            self.simulation.context.loadCheckpoint(f.read())
            steps_done = self.simulation.currentStep
            remaining_steps = self.steps - steps_done
            logger.info(f"loaded checkpoint: {steps_done} done {remaining_steps} remaining")
            if remaining_steps > 0:
                # checkpoint does not remember posres.
                # but, production simulation does not use posres
                # so, posres is not added here.
                # checkpoint does not remember reporters.
                self.simulation.reporters.append(
                    app.CheckpointReporter(
                        (self.workdir / f'{self.prefix}_{tag}.cpt').as_posix(),
                        int(self.checkpoint_interval / self.timestep + 0.5),
                        ))
                self.simulation.reporters.append(
                    app.DCDReporter(
                        (self.workdir / f'{self.prefix}_{tag}.dcd').as_posix(),
                        int(self.trajectory_interval / self.timestep + 0.5),
                        append=True))
                self.simulation.reporters.append(
                    app.StateDataReporter(
                        (self.workdir / f'{self.prefix}_{tag}.ene').as_posix(),
                        int(self.state_data_interval / self.timestep + 0.5),
                        step=True,
                        potentialEnergy=True,
                        temperature=True,
                        volume=True,
                        density=True,
                        progress=True,
                        remainingTime=True,
                        speed=True,
                        totalSteps=self.steps,
                        append=True))
                # run simulations
                self.simulation.step(remaining_steps)
        return True
    

    def run(self, tag: str = '5_prod') -> None:
        """Run production MD simulation."""
        logger.info(f"NPT production (T= {self.temperature})")
        logger.info(f'    t= {self.time}, dt= {self.timestep}, steps= {self.steps/1000}K')
        logger.info(f"    state data interval {self.state_data_interval}")
        logger.info(f"    checkpoint interval {self.checkpoint_interval}")
        logger.info(f"    trajectory interval {self.trajectory_interval}")
        logger.info(f"    trajectory frames {self.trajectory_frames}")
        # note: accurate steps and timestep should be calculated
        # before _resume_prod_simulation()
        if not self._resume_prod_simulation(tag=tag):
            if self.hmr:
                self._set_hmr()
            else:
                T = self.temperature.value_in_unit(unit.kelvin)
                f = self.friction.value_in_unit(1.0 / unit.picosecond)
                dt = self.timestep.value_in_unit(unit.femtosecond)
                self._change_temperature(temperature=T)
                self._change_integrator(temperature=T, friction=f, timestep=dt)
                self._change_posres(k=0.0)
                while not self._is_stable():
                    self._continue_pre_production()
            
            # add reporters
            self.simulation.reporters.append(
                app.CheckpointReporter(
                    (self.workdir / f'{self.prefix}_{tag}.cpt').as_posix(),
                    int(self.checkpoint_interval/self.timestep + 0.5),
                    ))
            self.simulation.reporters.append(
                app.DCDReporter(
                    (self.workdir / f'{self.prefix}_{tag}.dcd').as_posix(),
                    int(self.trajectory_interval/self.timestep + 0.5),
                    ))
            self.simulation.reporters.append(
                app.StateDataReporter(
                    (self.workdir / f'{self.prefix}_{tag}.ene').as_posix(), 
                    int(self.state_data_interval/self.timestep + 0.5),
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    totalSteps=self.steps))
            
            # run simulation
            self.simulation.step(self.steps)

        # final update
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()

        with gzip.open(self.workdir / f"{self.prefix}_LAST.pdb.gz", "wt") as f:
            app.PDBFile.writeFile(self.topology, self.positions, f)

        # create the empty file to mark completion
        (self.workdir / f"{self.prefix}_DONE").touch(exist_ok=True)

        logger.info(f"Production simulation complete!")
        logger.info(f"Structure saved to {self.prefix}_LAST.pdb.gz")
        logger.info(f"Trajectory saved to {self.prefix}.dcd")