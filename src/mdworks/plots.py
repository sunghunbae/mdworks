import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSF

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from collections import defaultdict

import logging


logger = logging.getLogger(__name__)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class CpptrajWrapper:
    def __init__(self):

        pass

class MDAnalyzer:
    # https://github.com/OmarArias-Gaguancela/MD_quick_plot/blob/main/md_quick_plot.py
    def __init__(self, 
                 topology : str | Path, 
                 trajectory : str | Path, 
                 receptors : dict | None = None, 
                 ligands : dict | None = None,
                 superimpose: str | None = None,
                 selection: str = 'backbone',
                 ref_frame: int = 0,
                 interval: float = 100.0,
                ):
        """
        Initialize MD Analyzer

        Parameters:
        -----------
        topology : str
            Path to PDB or GRO file
        trajectory : str
            Path to trajectory file (XTC, TRR, DCD, etc.)
        receptors : dict
            { key : MDAnalysis selection string for receptor, ...}
        ligands : dict
            { key : MDAnalysis selection string for ligand, ...}
        """
        self.receptors = receptors
        self.ligands = ligands
        self.complex = None
        
        self.topology = topology
        self.trajectory = trajectory

        self.labels : list[str] = []
        self.groupselections : list[str] = []
        self.selection : str = ''
        self.superimpose : str = ''
        self.ref_frame : int = ref_frame

        self.figsize: tuple[float, float] = (6, 4)
        self.axis_label_fontsize: int = 12
        self.title_fontsize: int = 14
        
        logger.info("Loading trajectory... This may take a moment.")
        self.u = mda.Universe(self.topology, self.trajectory)

        # # Select everything except water and ions
        # no_solvent = "not (resname SOL or resname WAT or resname HOH or resname NA or resname CL)"
        self.set_receptors(receptors)
        self.set_ligands(ligands)
        self.set_complex()

        self.rmsd_values : np.ndarray | None = None
        # select_rmsd, group1_rmsd, ...
        self.pl_distances  : np.ndarray | None = None

        self.n_frames = len(self.u.trajectory) # total number of trajectory frames
        self.interval = interval
        self.time = np.arange(0.0, interval * self.n_frames, interval)

        # Notes: below values are not accurate
        # self.duration = self.u.trajectory[0].time / 1000.0
        # self.interval = self.u.trajectory.dt
        
        logger.info(f"✓ Loaded {self.n_frames} frames")
        logger.info(f"✓ Interval(input): {self.interval} (ps)")
        logger.info(f"✓ Duration: {self.interval * self.n_frames} (ps)")
        logger.info(f"✓ Segment(s):")
        for segment in self.u.segments:
            logger.info(f"  segid: {segment.segid} residues: {len(segment.residues)} atoms: {len(segment.atoms)}")
        
        if selection.lower() == 'ca':
            selection = 'name CA'
        elif selection.lower() == 'backbone':
            selection = 'backbone'
        
        if isinstance(superimpose, str):
            assert self.receptors, "please define receptor(s) using .set_receptors()"
            assert superimpose in list(self.receptors.keys()), "ref should be one of the receptor keys"
            self.superimpose = superimpose
            self.selection = f'{self.receptors[superimpose]} and {selection}'
            atomgroup = self.u.select_atoms(self.selection)
            assert len(atomgroup) > 5, "5 or more atoms must be selected to superimpose"
            logger.info(f"✓ superimposing {self.selection} ({len(atomgroup)} atoms)")
            
        
    def set_receptors(self, receptors: dict | None) -> None:
        if isinstance(receptors, dict):
            self.labels += list(receptors.keys())
            self.groupselections += list(receptors.values())
            logger.info(f"✓ Receptor(s):")
            for k, e in self.receptors.items():
                logger.info(f"✓   key: {k} atoms: {len(self.u.select_atoms(e))}")

    
    def set_ligands(self, ligands: dict | None) -> None:
        if isinstance(ligands, dict):
            self.labels += list(ligands.keys())
            self.groupselections += list(ligands.values())
            logger.info(f"✓ Ligand(s):")
            for k, e in self.ligands.items():
                logger.info(f"✓   key: {k} atoms: {len(self.u.select_atoms(e))}")

    
    def set_complex(self) -> None:
        if self.groupselections:
            complex_expr = ' or '.join(self.groupselections)
            self.complex = self.u.select_atoms(complex_expr)
            logger.info(f"✓ Complex (Receptor(s)+Ligand(s)): {len(self.complex)} atoms")


    def set_figure(self, 
                   figsize: tuple[float, float] = (6, 4),
                   axis_label_fontsize: int = 12,
                   title_fontsize: int = 14,
                   ) -> None:
        self.figsize = figsize
        self.axis_label_fontsize = axis_label_fontsize
        self.title_fontsize = title_fontsize


    def calculate_rmsd(self) -> None:
        """Calculate RMSD over trajectory

        class MDAnalysis.analysis.rms.RMSD(atomgroup, reference=None, 
            select='all', groupselections=None, weights=None, weights_groupselections=False, 
            tol_mass=0.1, ref_frame=0, **kwargs)
            
        The RMSD will be computed for two groups of atoms and all frames in the trajectory belonging to atomgroup.
        The groups of atoms are obtained by applying the selection selection select to the changing atomgroup and 
        the fixed reference.

        If you use trajectory data from simulations performed under periodic boundary conditions then you must 
        make your molecules whole before performing RMSD calculations so that the centers of mass of the selected 
        and reference structure are properly superimposed.
        
        """
        logger.info("Calculating RMSD...")
        R = rms.RMSD(atomgroup= self.u, # mobile
                     reference= None, # fixed reference
                     select= self.selection, # applied to mobile and the fixed reference for superimposing 
                     groupselections= self.groupselections, # additional RMSDs to be computed
                     ref_frame= self.ref_frame,
                     center= False,
                     superposition= True,
                    verbose=True) # frame index to select frame from reference
        R.run()

        # R.results.rmsd columns [Frame, Time, select_rmsd, group1_rmsd, ...]
        self.rmsd_values = R.results.rmsd[:, 2:] # select_rmsd, group1_rmsd, ...

        
    def plot_rmsd(self, fig_path: str | Path | None = None, csv_path: str | Path | None = None) -> None:
        """Plot RMSD vs Time"""
        if self.rmsd_values is None:
            self.calculate_rmsd()

        fig, ax = plt.subplots(figsize = self.figsize)
        select_rmsd = self.rmsd_values[:, 0]
        data = [np.median(select_rmsd), np.std(select_rmsd)]
        for i in range(len(self.labels)):
            group_rmsd = self.rmsd_values[:, i+1]
            data += [np.median(group_rmsd), np.std(group_rmsd)]
            ax.plot(self.time, group_rmsd, linewidth=1.5, label=self.labels[i])
        
        ax.set_xlabel('Time (ps)', fontsize= self.axis_label_fontsize, fontweight='bold')
        ax.set_ylabel('RMSD (Å)', fontsize= self.axis_label_fontsize, fontweight='bold')
        ax.set_title('Backbone RMSD over Time', fontsize= self.title_fontsize, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add statistics - only mean line
        mean_rmsd = np.mean(self.rmsd_values[:, 0])
        std_rmsd = np.std(self.rmsd_values[:, 0])
        ax.axhline(mean_rmsd, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_rmsd:.2f} Å')
        ax.legend()

        plt.tight_layout()
        if isinstance(fig_path, str) or isinstance(fig_path, Path):
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ RMSD plot saved to {fig_path}")
        plt.show()

        # append to a given csv_path
        if isinstance(csv_path, str) or isinstance(csv_path, Path):
            csv_path = Path(csv_path)
            if csv_path.exists():
                header = None
            else:
                header = [self.selection,]
                for l in self.labels:
                    header.append(f'{l}(rmsd)')
                    header.append(f'{l}(stdev)')
                header.append('trajectory')
            with open(csv_path, '+a') as f:
                if header:
                    f.write(','.join(header) + '\n')
                f.write(','.join([f'{v:.2f}' for v in data]) + f',{self.trajectory}\n')



    def calculate_pl_distance(self, protein_sel: str = "name CA", ligand_sel: str = "not type H*") -> None:
        """Calculate minimum distance between protein and ligand"""
        logger.info("Calculating Protein-Ligand Distances...")
        distances = defaultdict(list)
        for lk, le in self.ligands.items():
            ligand = self.u.select_atoms(f'{le} and {ligand_sel}')
            for pk, pe in self.receptors.items():
                protein = self.u.select_atoms(f'{pe} and {protein_sel}')
                column = f'{lk}-{pk}'
                for ts in self.u.trajectory:
                    dist_array = mda.lib.distances.distance_array(
                        protein.positions,
                        ligand.positions,
                        self.u.dimensions,
                    )
                    min_dist = np.min(dist_array)
                    distances[column].append(min_dist)
        self.pl_distances = distances


    def plot_pl_distance(self, fig_path : str | Path | None = None, csv_path: str | Path | None = None) -> None:
        """Plot protein-ligand minimum distance"""
        if not self.pl_distances:
            self.calculate_pl_distance()

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(self.pl_distances.keys())
        data = []
        for k, v in self.pl_distances.items():
            ax.plot(self.time, v, linewidth=1.5, label=k)
            median_dist = np.median(v)
            data.append(median_dist)
            ax.axhline(median_dist, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_dist:.2f} Å')
        ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Minimum Distance (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Protein-Ligand Minimum Distance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Highlight contact threshold (typical: 4Å)
        ax.axhline(4.0, color='green', linestyle=':', alpha=0.5, label='Contact Threshold (4 Å)')
        ax.legend()

        plt.tight_layout()
        if isinstance(fig_path, str) or isinstance(fig_path, Path):
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Distance plot saved to {fig_path}")
        plt.show()

        # append to a given csv_path
        if isinstance(csv_path, str) or isinstance(csv_path, Path):
            csv_path = Path(csv_path)
            if csv_path.exists():
                header = None
            else:
                header = labels + ['trajectory',]
            with open(csv_path, '+a') as f:
                if header:
                    f.write(','.join(header) + '\n')
                f.write(','.join([f'{v:.2f}' for v in data]) + f',{self.trajectory}\n')


    def calculate_rmsf(self, selection="ca"):
        """Calculate RMSF per residue"""
        logger.info("Calculating RMSF...")

        # Use C-alpha atoms for RMSF calculation
        ca_atoms = self.protein.select_atoms("name CA")

        # Align trajectory using backbone
        backbone = self.protein.select_atoms("backbone")

        # Align all frames to first frame
        align.AlignTraj(self.u, self.u, select="backbone", in_memory=True).run()

        # Calculate RMSF
        rmsf_obj = RMSF(ca_atoms).run()
        rmsf_values = rmsf_obj.results.rmsf

        # Get residue numbers
        residue_numbers = ca_atoms.resnums

        return residue_numbers, rmsf_values

    def plot_rmsf(self, save_path="rmsf.png"):
        """Plot RMSF vs Residue"""
        residues, rmsf_values = self.calculate_rmsf()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(residues, rmsf_values, linewidth=1.5, color='#A23B72')
        ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSF (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Root Mean Square Fluctuation per Residue', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Highlight high fluctuation regions
        threshold = np.mean(rmsf_values) + np.std(rmsf_values)
        high_fluct = rmsf_values > threshold
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold: {threshold:.2f} Å')
        ax.scatter(residues[high_fluct], rmsf_values[high_fluct],
                  color='red', s=30, zorder=5, label='High Fluctuation')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ RMSF plot saved to {save_path}")
        plt.show()

        return residues, rmsf_values

    def calculate_rg(self, selection="protein"):
        """Calculate radius of gyration"""
        logger.info("Calculating Radius of Gyration...")

        if selection == "protein":
            sel = self.protein
        else:
            sel = self.u.select_atoms(selection)

        rg_values = []
        time_ns = []

        for ts in self.u.trajectory:
            rg_values.append(sel.radius_of_gyration())
            time_ns.append(ts.time / 1000.0)  # Convert to ns

        return np.array(time_ns), np.array(rg_values)

    def plot_rg(self, save_path="rg.png"):
        """Plot Radius of Gyration vs Time"""
        time_ns, rg_values = self.calculate_rg()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, rg_values, linewidth=1.5, color='#F18F01')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Gyration (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Protein Compactness over Time', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add statistics - only mean line
        mean_rg = np.mean(rg_values)
        std_rg = np.std(rg_values)
        ax.axhline(mean_rg, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_rg:.2f} Å')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Rg plot saved to {save_path}")
        plt.show()

        return time_ns, rg_values

    def calculate_free_energy_landscape(self, rmsd_values=None, rg_values=None, bins=50, temperature=300):
        """
        Calculate Free Energy Landscape using RMSD and Rg

        Parameters:
        -----------
        rmsd_values : array, optional
            RMSD values (if None, will calculate)
        rg_values : array, optional
            Rg values (if None, will calculate)
        bins : int
            Number of bins for 2D histogram
        temperature : float
            Temperature in Kelvin
        """
        logger.info("Calculating Free Energy Landscape...")

        if rmsd_values is None:
            _, rmsd_values = self.calculate_rmsd()
        if rg_values is None:
            _, rg_values = self.calculate_rg()

        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(rmsd_values, rg_values, bins=bins)

        # Convert to probability
        hist = hist / np.sum(hist)

        # Avoid log(0)
        hist[hist == 0] = np.min(hist[hist > 0]) * 0.01

        # Calculate free energy: ΔG = -RT ln(P)
        kB = 0.001987  # kcal/(mol·K)
        RT = kB * temperature
        free_energy = -RT * np.log(hist)

        # Normalize to minimum
        free_energy = free_energy - np.min(free_energy)

        return free_energy, xedges, yedges

    def plot_free_energy_landscape(self, save_path="fel.png", temperature=300):
        """Plot Free Energy Landscape and identify global minimum frame"""
        _, rmsd_values = self.calculate_rmsd()
        _, rg_values = self.calculate_rg()

        fel, xedges, yedges = self.calculate_free_energy_landscape(
            rmsd_values, rg_values, temperature=temperature
        )

        # Smooth the FEL
        fel_smooth = gaussian_filter(fel, sigma=1.0)

        # Find global minimum in the FEL
        min_idx = np.unravel_index(np.argmin(fel_smooth), fel_smooth.shape)
        min_rmsd = xedges[min_idx[0]]
        min_rg = yedges[min_idx[1]]

        # Find the frame closest to the global minimum
        # Calculate distance from each point to the minimum
        distances_to_min = np.sqrt((rmsd_values - min_rmsd)**2 + (rg_values - min_rg)**2)
        min_frame_idx = np.argmin(distances_to_min)

        # Get time information
        min_time_ns = self.u.trajectory[min_frame_idx].time / 1000.0
        min_frame_number = min_frame_idx

        # Get RMSD and Rg at that frame
        actual_rmsd = rmsd_values[min_frame_idx]
        actual_rg = rg_values[min_frame_idx]

        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot contour
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        levels = np.linspace(0, np.percentile(fel_smooth, 95), 20)

        contour = ax.contourf(X, Y, fel_smooth.T, levels=levels, cmap='viridis')
        contour_lines = ax.contour(X, Y, fel_smooth.T, levels=levels, colors='white',
                                   linewidths=0.5, alpha=0.3)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Free Energy (kcal/mol)', fontsize=12, fontweight='bold')

        ax.set_xlabel('RMSD (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Gyration (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Free Energy Landscape', fontsize=14, fontweight='bold')

        # Mark global minimum on FEL
        ax.plot(xedges[min_idx[0]], yedges[min_idx[1]], 'r*', markersize=20,
               label='Global Minimum', markeredgecolor='white', markeredgewidth=1)

        # Mark the actual trajectory frame closest to global minimum
        ax.plot(actual_rmsd, actual_rg, 'yo', markersize=12,
               label=f'Frame {min_frame_number} ({min_time_ns:.2f} ns)',
               markeredgecolor='black', markeredgewidth=1.5)

        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ FEL plot saved to {save_path}")

        # logger.info detailed information about global minimum
        logger.info("\n" + "="*60)
        logger.info("GLOBAL MINIMUM INFORMATION")
        logger.info("="*60)
        logger.info(f"FEL Global Minimum Position:")
        logger.info(f"  RMSD: {min_rmsd:.3f} Å")
        logger.info(f"  Rg:   {min_rg:.3f} Å")
        logger.info(f"\nClosest Trajectory Frame:")
        logger.info(f"  Frame Number: {min_frame_number}")
        logger.info(f"  Time:         {min_time_ns:.3f} ns")
        logger.info(f"  RMSD:         {actual_rmsd:.3f} Å")
        logger.info(f"  Rg:           {actual_rg:.3f} Å")
        logger.info(f"\nTo extract this frame from your trajectory:")
        logger.info(f"  Frame index (0-based): {min_frame_idx}")
        logger.info(f"  Time in trajectory:    {min_time_ns:.3f} ns")
        logger.info("="*60 + "\n")

        plt.show()

        return fel, xedges, yedges, min_frame_idx, min_time_ns

    def calculate_binding_energy_mm(self):
        """
        Calculate MM binding energy (Electrostatic + VdW) in kJ/mol
        Uses simplified Coulomb and Lennard-Jones potentials
        """
        logger.info("Calculating MM Binding Energy...")

        if len(self.ligand) == 0:
            logger.info("Warning: No ligand found. Please check ligand selection.")
            return None, None

        energies = []
        time_ns = []

        # Constants
        epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        epsilon_r = 1.0  # Relative permittivity (vacuum approximation)
        e = 1.602176634e-19  # Elementary charge (C)
        Na = 6.02214076e23  # Avogadro's number

        # Coulomb constant in kJ/(mol·nm)
        # k_e = (1 / (4π * ε₀ * ε_r)) * e² * Na / 1000
        k_e = 138.935458  # kJ·nm/(mol·e²) - standard value for MD simulations

        # Lennard-Jones parameters (typical protein-ligand values)
        epsilon_lj = 0.5  # kJ/mol - depth of potential well
        sigma = 0.35  # nm - distance at which potential is zero

        for ts in self.u.trajectory:
            # Get protein-ligand distance matrix in Angstroms
            distances_angstrom = mda.lib.distances.distance_array(
                self.protein.positions,
                self.ligand.positions
            )

            # Convert to nm for proper energy calculation
            distances_nm = distances_angstrom / 10.0

            # Avoid division by zero
            distances_nm = np.maximum(distances_nm, 0.01)  # Minimum distance 0.01 nm

            # Electrostatic energy (Coulomb's law)
            # E_elec = k_e * q1 * q2 / r
            # Assuming partial charges of ±0.5e for simplified calculation
            q_protein = 0.5  # Partial charge (elementary charge units)
            q_ligand = 0.5
            elec_energy = k_e * q_protein * q_ligand / distances_nm
            total_elec = np.sum(elec_energy)

            # Van der Waals energy (Lennard-Jones 6-12 potential)
            # E_vdw = 4ε[(σ/r)^12 - (σ/r)^6]
            sigma_r = sigma / distances_nm
            vdw_energy = 4 * epsilon_lj * (sigma_r**12 - sigma_r**6)
            total_vdw = np.sum(vdw_energy)

            # Total interaction energy
            total_energy = total_elec + total_vdw

            energies.append(total_energy)
            time_ns.append(ts.time / 1000.0)

        return np.array(time_ns), np.array(energies)

    def plot_binding_energy(self, save_path="binding_energy.png"):
        """Plot binding energy over time in kJ/mol"""
        time_ns, energies = self.calculate_binding_energy_mm()

        if energies is None:
            return None, None

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, energies, linewidth=1.5, color='#6A4C93')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Interaction Energy (kJ/mol)', fontsize=12, fontweight='bold')
        ax.set_title('Protein-Ligand Interaction Energy', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        ax.axhline(mean_energy, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_energy:.2f} kJ/mol')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Binding energy plot saved to {save_path}")
        logger.info(f"  Mean Energy: {mean_energy:.2f} ± {std_energy:.2f} kJ/mol")
        plt.show()

        return time_ns, energies


    def run_complete_analysis(self, output_prefix="analysis"):
        """Run all analyses and save plots"""
        logger.info("\n" + "="*60)
        logger.info("Running Complete MD Analysis")
        logger.info("="*60 + "\n")

        # RMSD
        self.plot_rmsd(f"{output_prefix}_rmsd.png")

        # RMSF
        self.plot_rmsf(f"{output_prefix}_rmsf.png")

        # Rg
        self.plot_rg(f"{output_prefix}_rg.png")

        # FEL
        self.plot_free_energy_landscape(f"{output_prefix}_fel.png")

        # Binding Energy
        if len(self.ligand) > 0:
            self.plot_binding_energy(f"{output_prefix}_binding_energy.png")
            self.plot_protein_ligand_distance(f"{output_prefix}_pl_distance.png")
        else:
            logger.info("\n⚠ Skipping binding energy and distance calculations (no ligand detected)")

        logger.info("\n" + "="*60)
        logger.info("✓ Analysis Complete!")
        logger.info("="*60)