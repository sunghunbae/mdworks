__all__ = ['MultiStage',]


"""Unbiased Classical Molecular Dynamics"""

import logging
import numpy as np

from pathlib import Path
from importlib.metadata import version
from copy import deepcopy

from openmm import (
    app, 
    unit,
    Platform,
    LangevinIntegrator,
    MonteCarloBarostat,
    MonteCarloAnisotropicBarostat,
    MonteCarloMembraneBarostat,
    CustomExternalForce,
    XmlSerializer,
    MinimizationReporter,
)

from openmmtools.testsystems import TestSystem

from .validcomplex import ValidComplex
from .simfileio import SimFileIO
from .utils import setup_logger

logger = logging.getLogger(__name__)

def is_cuda_available() -> bool:
    # Checks if the CUDA platform is available
    for i in range(Platform.getNumPlatforms()):
        if Platform.getPlatform(i).getName() == 'CUDA':
            return True
    return False


def is_opencl_available() -> bool:
    # Checks if the OpenCL platform is available
    for i in range(Platform.getNumPlatforms()):
        if Platform.getPlatform(i).getName() == 'OpenCL':
            return True
    return False


def getpar(kwargs: dict , key: str, default: float | tuple | None = None) -> float | tuple:
    value = kwargs.get(key)
    if value is None:
        return default
    elif isinstance(value, (tuple, list)):
        return tuple(value)
    elif isinstance(value, (float, int)):
        return value


def apply_hmr(system, topology, h_mass_factor: float = 3.0) -> None:
    """Apply HMR to a system.
    Assumes all bonds to H are constrained.
    Total mass is preserved.
    """
    h_mass_orig = 1.008 * unit.amu
    # Calculate mass to transfer
    delta = (h_mass_factor - 1.0) * 1.008 * unit.amu
    for bond in topology.bonds():
        # Identify hydrogen (H) and its bonded atom (non-H, X)
        a1, a2 = bond[0], bond[1]
        # Determine which is hydrogen
        if a1.element == app.element.hydrogen:
            h, x = a1, a2
        elif a2.element == app.element.hydrogen:
            h, x = a2, a1
        else:
            continue
        x_mass_orig = system.getParticleMass(x.index)
        # Apply new masses
        system.setParticleMass(h.index, h_mass_orig + delta)
        system.setParticleMass(x.index, x_mass_orig - delta)


class CustomMinimizationReporter(MinimizationReporter):
    def __init__(self, file: str | Path, reportInterval: int):
        super().__init__()
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._iterations_since_last_report = 0
        # Write a header
        self._out.write("Iteration, Potential Energy (kJ/mol)\n")

    def report(self, iteration, x, grad, args):
        # This method is called after each L-BFGS iteration
        self._iterations_since_last_report += 1
        if self._iterations_since_last_report % self._reportInterval == 0:
            # The objective function is not exactly potential energy due to constraints
            # For basic reporting, potential energy might be an approximation or
            # you can focus on the 'objective' value
            potential_energy = args['potentialEnergy'] # This might not be directly available, check 'objective' in args
            # A better way is to get the state from the context if needed, but 'args' has stats
            self._out.write(f"{iteration}, {potential_energy}\n")
            self._out.flush() # Ensure it writes to the file immediately

        # Return False to continue minimization (True to stop early)
        return False

    def __del__(self):
        self._out.close()


class MultiStage(SimFileIO):
    def __init__(self, 
                 complex: TestSystem | ValidComplex | Path | str,
                 workdir: Path | str | None = None,
                 platform: str = 'CUDA', 
                 devices: str = '0',
                 quiet:  bool = False,
                 ) -> None:
        self.platform = None
        self.properties = {}
        
        self.topology = None
        self.positions = None
        self.system = None
        self.integrator = None
        self.simulation = None

        self._set_platform(platform, devices)

        # setup workdir
        self.workdir : Path | None = None
        if isinstance(workdir, str):
            self.workdir = Path(workdir)
            self.workdir.mkdir(exist_ok=True)
        elif isinstance(workdir, Path):
            self.workdir = workdir
            self.workdir.mkdir(exist_ok=True)
        else:
            if isinstance(complex, ValidComplex):
                self.workdir = complex.parent
            elif isinstance(complex, Path) or isinstance(complex, str):
                self.workdir = Path(complex).parent
            else: # including TestSystem
                self.workdir = Path('.')
        
        # set up prefix
        self.prefix : str | None = None
        if isinstance(complex, ValidComplex):
            self.prefix = complex.prefix
        elif isinstance(complex, Path) or isinstance(complex, str):
            self.prefix = Path(complex).stem
        elif isinstance(complex, TestSystem):
            self.prefix = complex.name # the name of the test system

        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)

        logger.info(f"mdworks {version('mdworks')}")
        logger.info(f"openff-toolkit {version('openff-toolkit')}")
        logger.info(f"OpenMM platform= {platform} devices= {devices}")
        logger.info(f"workdir= {self.workdir}")
        logger.info(f"prefix= {self.prefix}")

        if isinstance(complex, ValidComplex):
            self.topology = complex.topology
            self.positions = complex.positions
            self.system = complex.system
            self._create_integrator()
            self._create_simulation()
            self.save_integrator()

        elif isinstance(complex, Path) or isinstance(complex, str):
            if not self.load_system():
                raise FileNotFoundError(".._system.xml file is required.")            
            if not self.load_complex():
                raise FileNotFoundError(".._complex.pdb file is required.")
            if not self.load_integrator():
                self._create_integrator()
                self.save_integrator()
            self._create_simulation()
        
        elif isinstance(complex, TestSystem):
            self.topology = complex.topology
            self.system = complex.system
            self.positions = complex.positions
            # positional restraints
            self._add_posres(k=1000.0)
            self._create_integrator()
            self._create_simulation()
            self.save_integrator()
            # hydrogen mass repartitioning (HMR)
            self.system_hmr = deepcopy(complex.system)
            # Apply to the copied system
            apply_hmr(self.system_hmr, complex.topology)
            filename = self.workdir / f"{self.prefix}_system_hmr.xml"
            with open(filename, "w") as f:
                f.write(XmlSerializer.serialize(self.system_hmr))
        
        

    def _add_posres(self, k: float = 1000.0) -> None:
        # create a positional restraint force
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        restrained_non_protein_residues = set()
        restrained = []

        for atom in self.topology.atoms():
            res = atom.residue.name
            if atom.element.symbol == 'H' or \
                res in ValidComplex.std_solvent_residues or \
                res in ValidComplex.std_divalent_ion_residues :
                continue
            i = atom.index
            restrained.append(i)
            pos = self.positions[i]
            force.addParticle(i, pos.value_in_unit(unit.nanometer))
            # just for reporting non-protein residues
            if res not in ValidComplex.std_protein_residues:
                restrained_non_protein_residues.add(res)
            
        self.system.addForce(force)


    def _set_platform(self, platform: str, devices: str) -> None:
        # platforms: OpenCL, CUDA, CPU, or Reference
        if platform == 'CUDA' and is_cuda_available():
            self.platform = Platform.getPlatform('CUDA') 
            self.properties = {
                "DeviceIndex" : devices,
                "Precision" : "mixed", # single, mixed, double
                "DeterministicForces" : "true",
                }
        elif platform == 'OpenCL' and is_opencl_available():
            self.platform = Platform.getPlatform('OpenCL') 
            self.properties = {
                "DeviceIndex" : devices,
                "Precision" : "mixed", # single, mixed, double
                "DeterministicForces" : "true",
                }
        else:
            self.platform = Platform.getPlatform('CPU') 

        logger.info(f"OpenMM {version('openmm')} with {self.platform.getName()}")



    def load_sim_env(self) -> bool:
        if not self.load_complex():
            return False
        if not self.load_system():
            return False
        if not self.load_integrator():
            return False
        self._create_simulation()
        return True


    def save_checkpoint(self, stage: int) -> None:
        tag = self.stages[stage]['tag']
        # Save an immediate binary checkpoint
        filename = (self.workdir / f'{self.prefix}_{tag}.cpt').as_posix()
        self.simulation.saveCheckpoint(filename)


    def load_checkpoint(self, stage: int) -> bool:
        tag = self.stages[stage]['tag']
        filename = self.workdir / f'{self.prefix}_{tag}.cpt'
        if not filename.exists():
            return False
        if not self.simulation:
            self.load_sim_env()
        # For loading a checkpoint (requires identical System/Platform/OpenMM)
        with open(filename, 'rb') as f:
            self.simulation.context.loadCheckpoint(f.read())
            # checkpoint does not remember positional restraints 
        return True
    

    def _create_integrator(self) -> None:
        """
        In OpenMM, the integrator friction factor (often denoted as gamma or frictionCoeff) 
        is a parameter (typically in 1/ps) used in Langevin or Brownian dynamics 
        to control how strongly the system couples to a heat bath. It determines the rate 
        of energy exchange, balancing damping (friction) and random noise to maintain a target temperature. 
        
        About OpenMM Friction Factors:
            - Purpose: It models the effect of an implicit solvent or external thermal bath, 
                affecting how quickly particles lose memory of their initial velocities.
            - Units: Measured in inverse picoseconds (1/ps).
            - Langevin Dynamics: A typical value is 1.0 or 0.1 (1/ps)
            - Effect: Higher values result in stronger coupling and faster thermalization, 
                while very low values approximate Hamiltonian (NVE) dynamics.
        """
        self.integrator = LangevinIntegrator(
            10 * unit.kelvin, 
            1 / unit.picosecond, 
            1 * unit.femtosecond)
        self.integrator.setConstraintTolerance(1e-5)  # default is 1e-5
            

    def _create_simulation(self) -> None:
        """
        The CUDA Platform supports parallelizing a simulation across multiple GPUs. 
        To do that, set the DeviceIndex property to a comma separated list of values. 
        For example,
            properties["DeviceIndex"] = "0,1";
        This tells it to use both devices 0 and 1, splitting the work between them.
        """
        # check if posres is already in the system
        if not self._is_posres_found():
            raise ValueError('posres info missing in the system')
            # posres should be added to system before creating simulation       

        self.simulation = app.Simulation(
                self.topology, 
                self.system, 
                self.integrator,
                self.platform,
                self.properties,
                )

        self.simulation.context.setPositions(self.positions)
        
        logger.info(
            f"Simulation created "
            f"Atoms: {self.topology.getNumAtoms()}, "
            f"Residues: {self.topology.getNumResidues()}, "
            f"Chains: {self.topology.getNumChains()}, "
            f"Constraints: {self.simulation.system.getNumConstraints()}"
            )
        params = ','.join(list(self.simulation.context.getParameters().keys()))
        if params:
            logger.info(f"Parameters: {params}")
        self.list_forces()


    def _is_posres_found(self) -> bool:
        for force in self.system.getForces():
            if isinstance(force, CustomExternalForce) and \
                "k*periodicdistance" in force.getEnergyFunction():
                return True
        return False


    def _is_prod_stable(self) -> bool:
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


    def _continue_pre_prod_equilibration(self, steps: int = 1000) -> None:
        logger.info(f"cont. pre-produciton equilibration {steps} steps")
        T = (self.temperature / unit.kelvin)
        self._change_temperature(temperature=T)
        self._change_integrator(temperature=T, friction=1.0, timestep=1.0)
        self._change_posres(k=0.0)
        self.simulation.step(steps)


    def list_forces(self) -> None:
        for i, force in enumerate(self.system.getForces()):
            if isinstance(force, CustomExternalForce) and \
                "k*periodicdistance" in force.getEnergyFunction():
                params = []
                try:
                    for j in range(force.getNumGlobalParameters()):
                        k = force.getGlobalParameterName(j)
                        v = force.getGlobalParameterDefaultValue(j)
                        params.append(f"{k}= {v:6.1f} KJ/mol/nm^2")
                except AttributeError:
                    continue
                logger.info(f"Force {i}: {force.__class__.__name__} posres ({' ,'.join(params)})")
            else:
                logger.info(f"Force {i}: {force.__class__.__name__}")


    def _change_posres(self, k: float) -> None:
        self.simulation.context.setParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
        # This directly updates the parameter value in the Context, 
        # which is where the simulation state is held. 
        # No further update is needed.
        logger.info(f"posres k= {self._get_posres_k():6.1f} KJ/mol/nm^2")
        


    def _get_posres_k(self) -> float:
        return self.simulation.context.getParameter("k")
    

    def _change_temperature(self, temperature: float, reset: bool = False) -> None:
        self.integrator.setTemperature(temperature * unit.kelvin)
        # setTemperature() tells the thermostat the new target
        # Changes the thermostat target
        # Affects future dynamics
        # Does not modify current particle velocities

        if reset:
            self.simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
            # setVelocitiesToTemperature() would erase physical velocity 
            # correlations and prevent a long relaxation period
            # Reassigns velocities from a Maxwell–Boltzmann distribution
            # Affects the instantaneous kinetic temperature
            # Has no memory: next step depends on integrator & forces
            logger.info(f"velocities set from a Maxwell-Boltzmann distribution at {temperature} K")


    def _change_integrator(self, temperature: float, friction: float, timestep: float) -> None:
        self.integrator.setTemperature(temperature * unit.kelvin)
        self.integrator.setFriction(friction / unit.picosecond)
        self.integrator.setStepSize(timestep * unit.femtosecond)


    def _get_integrator_steps(self, ps: float) -> int:
        dt = self.integrator.getStepSize() # Quantity with units
        return int(ps * unit.picoseconds / dt + 0.5)
    

    def _add_barostat(self, frequency: int = 25) -> None:
        """Add barostat only if one does not already exist.

        In OpenMM, the barostat frequency is the interval, measured in simulation time steps, 
        at which the MonteCarloBarostat attempts to change the volume of the simulation box 
        to maintain a target pressure (typically used in NPT ensembles). It defines how often 
        volume-scaling moves are attempted. 

        Default Value: The default is commonly set to every 25 steps.
        Function: It determines the frequency of Monte Carlo volume changes.
        Disabling: Setting the frequency to 0 disables the barostat.
 
        A standard practice is to allow the system to reach equilibrium, which is why a frequency 
        like 25 steps is common to maintain stable pressure without excessive, 
        costly volume calculations.

        Args:
            frequency (int, optional): _description_. Defaults to 25.
        """
        barostat_exists = False
        for force in self.system.getForces():
            if isinstance(force, (MonteCarloBarostat, 
                                  MonteCarloAnisotropicBarostat,
                                  MonteCarloMembraneBarostat)):
                barostat_exists = True
                logger.info(f"Barostat ({type(force).__name__}) exists: not adding a new one.")
                break

        if not barostat_exists:
            self.barostat = self.system.addForce(
                MonteCarloBarostat(self.pressure, self.temperature,  frequency))
            self.simulation.context.reinitialize(preserveState=True)
            logger.info(f"Barostat (MonteCarloBarostat, frequency= {frequency}) added")


    def _ramp_temperature(self, steps: int, start: float, end: float, dT: float = 10.0):
        num_ramps = int((end - start)/dT + 0.5) 
        steps_ramp = steps // num_ramps
        for i in range(num_ramps): # 0..59
            T = min(start + (i+1)*dT, end)
            self._change_temperature(T)
            self.simulation.step(steps_ramp)

    
    def _ramp_posres(self, steps: int, start: float, end: float, dk: float = -1.0):
        num_ramps = int((end - start)/dk + 0.5)
        steps_ramp = steps // num_ramps
        for i in range(num_ramps): # 0..19
            k = max(start + (i+1)*dk, end)
            self._change_posres(k)
            self.simulation.step(steps_ramp)


    def _resume_prod_simulation(self) -> bool:
        tag = self.stages[-1]['tag']
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


    def _get_runtime_precision(self) -> str:
        platform = self.simulation.context.getPlatform()
        return platform.getPropertyValue(self.simulation.context, "Precision")


    def _set_hmr(self) -> None:
        logger.info(f"Switching to HMR system for timestep of 4 fs")

        # self.positions has not been updated since __init__()
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()

        # self.system will be overwritten
        if not self.load_system(hmr=True):
            raise FileNotFoundError(".._system_hmr.xml file is required.")

        self.integrator = LangevinIntegrator(
            self.temperature,
            1 / unit.picosecond,
            4 * unit.femtosecond,
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
        self.barostat = self._add_barostat(frequency=50.)

        # save checkpoint
        filename = (self.workdir / f'{self.prefix}_hmr.cpt').as_posix()
        self.simulation.saveCheckpoint(filename)
        logger.info(f"Checkpoint saved to {filename}")


    def _add_state_data_reporter(self, stage: int, steps: int, interval: int, eta: bool = False) -> None:
        # add reporter
        tag = self.stages[stage]['tag']
        filename = self.workdir / f'{self.prefix}_{tag}.ene'
        if eta:
            progress = True
            remainingTime= True
            speed= True
            totalSteps= steps
        else:
            progress = False
            remainingTime = False
            speed = False
            totalSteps= steps
        
        self.simulation.reporters.append(
            app.StateDataReporter(
                filename.as_posix(), 
                interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                progress=progress,
                remainingTime=remainingTime,
                speed=speed,
                totalSteps=totalSteps))
        

    def _del_reporter(self) -> None:
        self.simulation.reporters.pop()


    def _stage_energy_minimization(self, stage: int, **kwargs) -> None:
        if self.load_checkpoint(stage):
            return
        maxiter = kwargs.get('maxiter', 5000) # openmm default 0 (until convergence is achived)
        tolerance = kwargs.get('tolerance', 0.1) # openmm default 10.0 KJ/mol
        interval = kwargs.get('interval', 10)
        logger.info(f"({stage}) Energy Minimization")
        # StateDataReporter does not work with simulataion.minimizeEnergy()
        # so a customized MinimizatinReporter is attached here
        # ... set up system, forcefield, simulation ...
        tag = self.stages[stage]['tag']
        filename = self.workdir / f'{self.prefix}_{tag}.ene'
        reporter = CustomMinimizationReporter(filename, reportInterval=interval)
        # Once the minimization process (simulation.minimizeEnergy()) is complete, 
        # the MinimizationReporter is no longer active.
        # The Simulation object's main reporters list, which is used during molecular dynamics (MD) steps 
        # (e.g., PDBReporter, StateDataReporter), is separate.
        self.simulation.minimizeEnergy(
            tolerance = tolerance * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations= maxiter,
            reporter= reporter,
            )
        self.save_checkpoint(stage)


    def _stage_NVT_cold(self, stage: int, **kwargs) -> None:
        if self.load_checkpoint(stage):
            return
        T = getpar(kwargs, 'T', 10.0)
        k = getpar(kwargs, 'k', 1000.0)
        (ps, dt) = getpar(kwargs, 't')
        steps = int(1000 * ps / dt + 0.5)
        friction = kwargs.get('friction', 5.0) # use 50 for Brownian dynamics
        interval = kwargs.get('interval', 1000) # report interval
        logger.info(f'({stage}) NVT (T= {T} K, posres_k= {k} kJ/mol/nm**2, friction= {friction} 1/ps)')
        logger.info(f'    t= {ps} ps, dt= {dt} fs, steps= {steps/1000}K')
        # T=10-50K friction=1-5 dt=1 fs t=50-200 ps
        self._change_temperature(temperature=T, reset=True)
        self._change_integrator(temperature=T, friction=friction, timestep=dt)
        self._change_posres(k=k)
        self._add_state_data_reporter(stage, steps, interval=interval)
        self.simulation.step(steps)
        self._del_reporter()
        self.save_checkpoint(stage)


    def _stage_NVT_warm(self, stage: int, **kwargs) -> None:
        if self.load_checkpoint(stage):
            return
        (T0, T1, dT) = getpar(kwargs, 'T', (10., 300., 10))
        (ps, dt) = getpar(kwargs, 't')
        steps = int(1000 * ps / dt + 0.5)
        k = getpar(kwargs, 'k', 1000.0)
        friction = kwargs.get('friction', 1.0)
        interval = kwargs.get('interval', 1000)
        logger.info(f'({stage}) NVT (T= {T0} -> {T1} K, posres_k= {k} kJ/mol/nm**2, friction= {friction} 1/ps)')
        logger.info(f'    t= {ps} ps, dt= {dt} fs, steps= {steps/1000}K')
        self._change_temperature(temperature=T0, reset=True)
        self._change_integrator(temperature=T0, friction=friction, timestep=dt)
        self._change_posres(k=k)
        self._add_state_data_reporter(stage, steps, interval=interval)
        # gradually heat up the system (it runs simulation.step())
        self._ramp_temperature(steps, start=T0, end=T1, dT=dT)
        self._del_reporter()
        self.save_checkpoint(stage)


    def _stage_NPT_warm(self, stage: int, **kwargs) -> None:
        if self.load_checkpoint(stage):
            return
        (T0, T1, dT) = getpar(kwargs, 'T', (10., 300., 10))
        (ps, dt) = getpar(kwargs, 't')
        steps = int(1000 * ps / dt + 0.5)
        k = getpar(kwargs, 'k', 40.0)
        friction = kwargs.get('friction', 1.0)
        interval = kwargs.get('interval', 1000)
        frequency = kwargs.get('frequency', 50)
        logger.info(f'({stage}) NPT (T= {T0} -> {T1} K, posres_k= {k} kJ/mol/nm**2, friction= {friction} 1/ps)')
        logger.info(f'    t= {ps} ps, dt= {dt} fs, steps= {steps/1000}K')
        # dt=2 fs t=500-1000 ps
        self._add_barostat(frequency=frequency)
        # reinitialize velocities
        self._change_temperature(temperature=T0, reset=True)
        self._change_integrator(temperature=T0, friction=friction, timestep=dt)
        self._change_posres(k=k)
        self._add_state_data_reporter(stage, steps, interval=interval)
        # gradually decrease posres (it runs simulation.step())
        # gradually heat up the system (it runs simulation.step())
        self._ramp_temperature(steps, start=T0, end=T1, dT=dT)
        self._del_reporter()
        self.save_checkpoint(stage)


    def _stage_NPT_posres(self, stage: int, **kwargs) -> None:
        if self.load_checkpoint(stage):
            return
        T = getpar(kwargs, 'T', 300)
        (ps, dt) = getpar(kwargs, 't')
        steps = int(1000 * ps / dt + 0.5)
        (k_start, k_end, dk) = getpar(kwargs, 'k', (1000., 0., -20))
        friction = kwargs.get('friction', 1.0)
        frequency = kwargs.get('frequency', 50)
        interval = kwargs.get('interval', 1000)
        logger.info(f'({stage}) NPT (T= {T} K, posres_k= {k_start} -> {k_end} kJ/mol/nm**2, friction= {friction} 1/ps)')
        logger.info(f'    t= {ps} ps, dt= {dt} fs, steps= {steps/1000}K')
        # add barostat only if one does not already exist.
        self._add_barostat(frequency=frequency)
        # reinitialize velocities
        self._change_temperature(temperature=T, reset=True)
        self._change_integrator(temperature=T, friction=friction, timestep=dt)
        self._change_posres(k=k_start)
        self._add_state_data_reporter(stage, steps, interval=interval)
        # gradually decrease posres (it runs simulation.step())
        self._ramp_posres(steps, start=k_start, end=k_end, dk=dk)
        self._del_reporter()
        self.save_checkpoint(stage)


    def _stage_NPT_free(self, stage: int, **kwargs)-> None:
        if self.load_checkpoint(stage):
            return
        T = getpar(kwargs, 'T', 300.0)
        (ps, dt) = getpar(kwargs, 't')
        steps = int(1000 * ps / dt + 0.5)
        friction = kwargs.get('friction', 1.0)
        frequency = kwargs.get('frequency', 50)
        interval = kwargs.get('interval', 1000)
        logger.info(f'({stage}) NPT (T= {T} K, posres_k= 0 kJ/mol/nm**2, friction= {friction} 1/ps)')
        logger.info(f'    t= {ps} ps, dt= {dt} fs, steps= {steps/1000}K')
        # dt=2 fs t=1000-5000 ps
        # add barostat only if one does not already exist.
        self._add_barostat(frequency=frequency)
        self._change_temperature(temperature=T)
        self._change_integrator(temperature=T, friction=friction, timestep=dt)
        self._change_posres(k=0.0)
        self._add_state_data_reporter(stage, steps, interval=interval, eta=True)
        self.simulation.step(steps)
        self._del_reporter()
        self.save_checkpoint(stage)


    def _stage_NPT_prod(self, stage: int, **kwargs) -> None:
        T = getpar(kwargs, 'T', 300.0)
        (ps, dt) = getpar(kwargs, 't')
        friction = kwargs.get('friction', 1.0)
        self.time = ps * unit.picosecond
        if self.hmr: # override
            dt = 4.0
        self.timestep = dt * unit.femtosecond
        self.steps = int(self.time / self.timestep + 0.5)
        state_data_interval = kwargs.get('state_data_interval', 100.0)
        checkpoint_interval = kwargs.get('checkpoint_interval', 100.0)
        trajectory_interval = kwargs.get('trajectory_interval', 100.0)
        self.state_data_interval = state_data_interval * unit.picosecond
        self.checkpoint_interval = checkpoint_interval * unit.picosecond
        self.trajectory_interval = trajectory_interval * unit.picosecond
        self.trajectory_frames = int(self.time / self.trajectory_interval + 0.5)
        logger.info(f"({stage}) NPT production (T= {T} K)")
        logger.info(f'    t= {self.time}, dt= {self.timestep}, steps= {self.steps/1000}K')
        logger.info(f"    state data interval {self.state_data_interval}")
        logger.info(f"    checkpoint interval {self.checkpoint_interval}")
        logger.info(f"    trajectory interval {self.trajectory_interval}")
        logger.info(f"    trajectory frames {self.trajectory_frames}")
        tag = self.stages[stage]['tag']
        # note: accurate steps and timestep should be calculated
        # before _resume_prod_simulation()
        if not self._resume_prod_simulation():
            if self.hmr:
                self._set_hmr()
            else:
                self._change_temperature(temperature=T)
                self._change_integrator(temperature=T, friction=friction, timestep=dt)
                self._change_posres(k=0.0)
                while not self._is_prod_stable():
                    self._continue_pre_prod_equilibration()
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
            self._add_state_data_reporter(
                stage, 
                steps= self.steps, 
                interval= int(self.state_data_interval/self.timestep + 0.5),
                eta= True)
            # run simulation
            self.simulation.step(self.steps)


    def diagnose(self) -> None:
        state = self.simulation.context.getState(
            getPositions=True, 
            getVelocities=True, 
            getForces=True, 
            getEnergy=True)

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    
        diagnostics = {
            'nan_positions': [],
            'nan_velocities': [],
            'nan_forces': [],
            'extreme_forces': [],
            'extreme_velocities': [],
            'close_contacts': [],
            'hydrogen_issues': []
        }
        
        # Check for NaN values
        for i in range(len(positions)):
            atom = list(self.topology.atoms())[i]
            
            if np.any(np.isnan(positions[i])):
                diagnostics['nan_positions'].append({
                    'index': i,
                    'atom': atom.name,
                    'residue': f"{atom.residue.name}{atom.residue.id}",
                    'element': atom.element.symbol if atom.element else 'Unknown'
                })
            
            if np.any(np.isnan(velocities[i])):
                diagnostics['nan_velocities'].append({
                    'index': i,
                    'atom': atom.name,
                    'residue': f"{atom.residue.name}{atom.residue.id}",
                    'element': atom.element.symbol if atom.element else 'Unknown'
                })
            
            if np.any(np.isnan(forces[i])):
                diagnostics['nan_forces'].append({
                    'index': i,
                    'atom': atom.name,
                    'residue': f"{atom.residue.name}{atom.residue.id}",
                    'element': atom.element.symbol if atom.element else 'Unknown'
                })
        
        # Check for extreme forces (> 10000 kJ/mol/nm)
        force_magnitudes = np.linalg.norm(forces, axis=1)
        extreme_force_idx = np.where(force_magnitudes > 10000)[0]
        
        for i in extreme_force_idx:
            atom = list(self.topology.atoms())[i]
            diagnostics['extreme_forces'].append({
                'index': i,
                'atom': atom.name,
                'residue': f"{atom.residue.name}{atom.residue.id}",
                'force_magnitude': force_magnitudes[i],
                'element': atom.element.symbol if atom.element else 'Unknown'
            })
        
        # Check for extreme velocities (> 100 nm/ps)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        extreme_vel_idx = np.where(velocity_magnitudes > 100)[0]
        
        for i in extreme_vel_idx:
            atom = list(self.topology.atoms())[i]
            diagnostics['extreme_velocities'].append({
                'index': i,
                'atom': atom.name,
                'residue': f"{atom.residue.name}{atom.residue.id}",
                'velocity_magnitude': velocity_magnitudes[i],
                'element': atom.element.symbol if atom.element else 'Unknown'
            })
        
        # Check for close contacts (< 0.05 nm)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if not np.any(np.isnan(positions[i])) and not np.any(np.isnan(positions[j])):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 0.05:
                        atom_i = list(self.topology.atoms())[i]
                        atom_j = list(self.topology.atoms())[j]
                        diagnostics['close_contacts'].append({
                            'atom1_idx': i,
                            'atom2_idx': j,
                            'atom1': f"{atom_i.name}({atom_i.residue.name}{atom_i.residue.id})",
                            'atom2': f"{atom_j.name}({atom_j.residue.name}{atom_j.residue.id})",
                            'distance': dist
                        })
        
        # Check hydrogen-specific issues (relevant for HMR)
        system = self.simulation.system
        for i in range(system.getNumParticles()):
            atom = list(self.topology.atoms())[i]
            if atom.element and atom.element.symbol == 'H':
                mass = system.getParticleMass(i).value_in_unit(unit.dalton)
                if mass > 3.0:  # HMR typically uses ~3-4 Da for hydrogens
                    if i in extreme_force_idx or i < len(diagnostics['nan_forces']) and diagnostics['nan_forces']:
                        diagnostics['hydrogen_issues'].append({
                            'index': i,
                            'atom': atom.name,
                            'residue': f"{atom.residue.name}{atom.residue.id}",
                            'mass': mass,
                            'issue': 'Heavy hydrogen with extreme force/NaN'
                        })
        
        # Print summary
        print("=" * 60)
        print("OpenMM NaN Diagnostic Report")
        print("=" * 60)
        
        if diagnostics['nan_positions']:
            print(f"\n⚠️  Found {len(diagnostics['nan_positions'])} atoms with NaN positions:")
            for item in diagnostics['nan_positions'][:10]:
                print(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['nan_velocities']:
            print(f"\n⚠️  Found {len(diagnostics['nan_velocities'])} atoms with NaN velocities:")
            for item in diagnostics['nan_velocities'][:10]:
                print(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['nan_forces']:
            print(f"\n⚠️  Found {len(diagnostics['nan_forces'])} atoms with NaN forces:")
            for item in diagnostics['nan_forces'][:10]:
                print(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['extreme_forces']:
            print(f"\n⚠️  Found {len(diagnostics['extreme_forces'])} atoms with extreme forces (>10000 kJ/mol/nm):")
            for item in diagnostics['extreme_forces'][:10]:
                print(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}, F={item['force_magnitude']:.1f}")
        
        if diagnostics['extreme_velocities']:
            print(f"\n⚠️  Found {len(diagnostics['extreme_velocities'])} atoms with extreme velocities (>100 nm/ps):")
            for item in diagnostics['extreme_velocities'][:10]:
                print(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}, v={item['velocity_magnitude']:.1f}")
        
        if diagnostics['close_contacts']:
            print(f"\n⚠️  Found {len(diagnostics['close_contacts'])} close contacts (<0.05 nm):")
            for item in diagnostics['close_contacts'][:10]:
                print(f"   Atoms {item['atom1_idx']}-{item['atom2_idx']}: {item['atom1']} <-> {item['atom2']}, d={item['distance']:.4f} nm")
        
        if diagnostics['hydrogen_issues']:
            print(f"\n⚠️  Found {len(diagnostics['hydrogen_issues'])} problematic hydrogens with HMR:")
            for item in diagnostics['hydrogen_issues']:
                print(f"   Atom {item['index']}: {item['atom']} in {item['residue']}, mass={item['mass']:.2f} Da")
        
        print("\n" + "=" * 60)
        print("Recommendations:")
        print("=" * 60)
        
        if diagnostics['close_contacts']:
            print("• Close contacts detected - consider energy minimization")
        if diagnostics['extreme_forces']:
            print("• Extreme forces detected - reduce timestep or increase constraint tolerance")
        if diagnostics['hydrogen_issues']:
            print("• Heavy hydrogen issues - verify HMR parameters or reduce timestep to 2-3 fs")
        if diagnostics['extreme_velocities']:
            print("• Extreme velocities detected - system may be unstable, consider restarting")
        
        print("\n")