__all__ = ['Diagnosis',]

from openmm import unit
import numpy as np
import logging


logger = logging.getLogger(__name__)


class Diagnosis:
    def __init__(self, simulation, topology, positions):
        self.simulation = simulation
        self.topology = topology
        self.positions = positions


    def examine(self) -> None:
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
        for i in range(len(self.positions)):
            for j in range(i+1, len(self.positions)):
                if not np.any(np.isnan(self.positions[i])) and not np.any(np.isnan(self.positions[j])):
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
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
        
        # logger.info summary
        logger.info("=" * 60)
        logger.info("OpenMM NaN Diagnostic Report")
        logger.info("=" * 60)
        
        if diagnostics['nan_positions']:
            logger.info(f"\n⚠️  Found {len(diagnostics['nan_positions'])} atoms with NaN positions:")
            for item in diagnostics['nan_positions'][:10]:
                logger.info(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['nan_velocities']:
            logger.info(f"\n⚠️  Found {len(diagnostics['nan_velocities'])} atoms with NaN velocities:")
            for item in diagnostics['nan_velocities'][:10]:
                logger.info(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['nan_forces']:
            logger.info(f"\n⚠️  Found {len(diagnostics['nan_forces'])} atoms with NaN forces:")
            for item in diagnostics['nan_forces'][:10]:
                logger.info(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}")
        
        if diagnostics['extreme_forces']:
            logger.info(f"\n⚠️  Found {len(diagnostics['extreme_forces'])} atoms with extreme forces (>10000 kJ/mol/nm):")
            for item in diagnostics['extreme_forces'][:10]:
                logger.info(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}, F={item['force_magnitude']:.1f}")
        
        if diagnostics['extreme_velocities']:
            logger.info(f"\n⚠️  Found {len(diagnostics['extreme_velocities'])} atoms with extreme velocities (>100 nm/ps):")
            for item in diagnostics['extreme_velocities'][:10]:
                logger.info(f"   Atom {item['index']}: {item['atom']} ({item['element']}) in {item['residue']}, v={item['velocity_magnitude']:.1f}")
        
        if diagnostics['close_contacts']:
            logger.info(f"\n⚠️  Found {len(diagnostics['close_contacts'])} close contacts (<0.05 nm):")
            for item in diagnostics['close_contacts'][:10]:
                logger.info(f"   Atoms {item['atom1_idx']}-{item['atom2_idx']}: {item['atom1']} <-> {item['atom2']}, d={item['distance']:.4f} nm")
        
        if diagnostics['hydrogen_issues']:
            logger.info(f"\n⚠️  Found {len(diagnostics['hydrogen_issues'])} problematic hydrogens with HMR:")
            for item in diagnostics['hydrogen_issues']:
                logger.info(f"   Atom {item['index']}: {item['atom']} in {item['residue']}, mass={item['mass']:.2f} Da")
        
        logger.info("\n" + "=" * 60)
        logger.info("Recommendations:")
        logger.info("=" * 60)
        
        if diagnostics['close_contacts']:
            logger.info("• Close contacts detected - consider energy minimization")
        if diagnostics['extreme_forces']:
            logger.info("• Extreme forces detected - reduce timestep or increase constraint tolerance")
        if diagnostics['hydrogen_issues']:
            logger.info("• Heavy hydrogen issues - verify HMR parameters or reduce timestep to 2-3 fs")
        if diagnostics['extreme_velocities']:
            logger.info("• Extreme velocities detected - system may be unstable, consider restarting")