__all__ = ['ValidComplex',]

import io
import logging
import numpy as np

from pathlib import Path
from importlib.metadata import version

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from pdbfixer import PDBFixer
    from openff.toolkit.topology.molecule import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openmm import (
        app, 
        unit, 
        CustomExternalForce, 
        XmlSerializer,
        )

except ImportError:
    raise ImportError("install openmm, openmmforcefields, pdbfixer, and openff-toolkit from conda-forge.\n")

from .utils import setup_logger


logger = logging.getLogger(__name__)


class ValidComplex:
    """Class for preparing valid protein/ligand complex structure.
    
    Issues with OpenFold3 and OpenEye Spruce
        - OpenFold3 generates flat geometries instead of tetrahedral for undefined chiral center(s)
        - OpenFold3 does not generate hydrogen atoms
        - Spruce cannot assign either (S) nor (R) to these flat carbon atoms
        - Spruce arbitrarily assigns double bond when inferring structures from coordinates

    Issue with OpenForceField `Molecule.from_pdb_and_smiles()`
        - not robust enough
    """

    std_protein_residues = {
        "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY",
        "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
        "THR","TRP","TYR","VAL",
        }
        
    std_solvent_residues = {
        "HOH","NA", "K", "CL",
        }
    
    std_divalent_ion_residues = {
        "MG" , "ZN", "CA", "MN", "FE", "CU", "CO", "CD", "NI", "SR", "BA", 
        }
    
    def __init__(self, 
                 in_file: str | Path,
                 remove_solvent: bool = True,
                 pH: float = 7.0, 
                 max_displacement: float = 0.5,
                 k: float = 1000.0,
                 max_iter: int = 500,
                 quiet: bool = False,                
                 ):
        """Initialize class object.

        Args:
            in_file (str | Path): input complex structure in mmcif format.
            pH (float, optional): pH to be considered when adding hydrogens. Defaults to 7.0.
            max_displacement (float, optional): max displacement (in A) during restrained optimization. Defaults to 0.5.
            k (float, optional): force constant during restrained optimization. Defaults to 1000.0.
            max_iter (int, optional): max number of iteration in restrained optimization. Defaults to 500.
        """
        if isinstance(in_file, Path):
            in_file = in_file.as_posix()

        self.in_file : str = in_file
        self.parent : Path = Path(in_file).parent
        self.prefix : str = Path(in_file).stem
        self.mem_protein : io.StringIO = io.StringIO()
        self.mem_ligand : io.StringIO = io.StringIO()
        self.mem_ligand_charges: io.StringIO = io.StringIO()
        self.remove_solvent : bool = remove_solvent
        self.pH : float = pH
        
        # ligand
        self.smiles : str = ''
        self.rdmol : Chem.Mol = Chem.Mol()
        self.rdmolH : Chem.Mol = Chem.Mol() # with hydrogens and 3D coords
        self.atom_map : dict = {} # mapping from self.rdmol to source molecule
        self.source : Chem.Mol = Chem.Mol()
        self.off_mol : Molecule = Molecule()

        # optimization
        self.max_displacement: float = max_displacement
        self.k: float = k
        self.max_iter: int = max_iter

        self.fixer = PDBFixer(in_file)
        self.protein_modeller = None
        self.ligand_modeller = None
        self.modeller = None
        self.restrained = []
        self.system = None
        
        setup_logger(logger, self.parent, self.prefix, quiet=quiet)

        logger.info(f"mdworks {version('mdworks')}")
        logger.info(f"pdbfixer {version('pdbfixer')}")
        logger.info(f"openmmforcefields {version('openmmforcefields')}")
        logger.info(f"openff-toolkit {version('openff-toolkit')}")
        logger.info(f"rdkit {version('rdkit')}")
        logger.info(f"scipy {version('scipy')}")

        self._add_missing_atoms()
        self._sort_protein_and_ligand_residues()


    def _add_missing_atoms(self) -> None:
        """Add missing atoms such as OXT."""
        # fixer.findMissingResidues()
        self.fixer.missingResidues = {}  # bypass logic
        self.fixer.findMissingAtoms()
        self.fixer.addMissingAtoms()
        self.fixer.addMissingHydrogens(pH=self.pH)


    def _sort_protein_and_ligand_residues(self) -> None:
        """Sort protein and ligand."""
        groups: list[str] = ['protein', 'solvent', 'divalent', 'ligand']
        residue_group = {k: set() for k in groups}
        for res in self.fixer.topology.residues():
            if res.name in ValidComplex.std_protein_residues:
                residue_group['protein'].add(res.name)
            elif res.name in ValidComplex.std_solvent_residues:
                residue_group['solvent'].add(res.name)
            elif res.name in ValidComplex.std_divalent_ion_residues:
                residue_group['divalent'].add(res.name)
            else:
                residue_group['ligand'].add(res.name)
                logger.info(f"ligand residue found {res.name}")

        # protein (including structural divalent ions)
        logger.info("protein preparation:")
        self.protein_modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
        residues_to_delete = []
        for res in self.protein_modeller.topology.residues():
            if (self.remove_solvent and (res.name in residue_group['solvent'])) or \
                (res.name in residue_group['ligand']):
                logger.info(f"  deleting: {res}")
                residues_to_delete.append(res)
        self.protein_modeller.delete(residues_to_delete)
        # non-standard residues        
        for res in self.protein_modeller.topology.residues():
            if res.name not in ValidComplex.std_protein_residues:
                logger.info(f"  including non-protein: {res}")
        logger.info(f"  number of residues: {self.protein_modeller.topology.getNumResidues()}")

        self._check_clashes(
            self.protein_modeller.topology, 
            self.protein_modeller.positions)

        # ligand
        logger.info("ligand preparation:")
        self.ligand_modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
        residues_to_delete = []
        for res in self.ligand_modeller.topology.residues():
            if res.name in residue_group['protein'] or \
               res.name in residue_group['solvent'] or \
               res.name in residue_group['divalent']:
                residues_to_delete.append(res)
            else:
                logger.info(f"  keeping: {res}")
        self.ligand_modeller.delete(residues_to_delete)
        logger.info(f"  number of residues: {self.ligand_modeller.topology.getNumResidues()}")

        self._check_clashes(
            self.ligand_modeller.topology, 
            self.ligand_modeller.positions)

        app.PDBFile.writeFile(
            self.protein_modeller.topology,
            self.protein_modeller.positions,
            self.mem_protein,
            keepIds=True
        )

        app.PDBFile.writeFile(
            self.ligand_modeller.topology,
            self.ligand_modeller.positions,
            self.mem_ligand,
            keepIds=False
        )
        # keepIds (optional): A boolean value (default is False). 
        # If True, the residue and chain IDs specified in the Topology are used; 
        # otherwise, new ones are generated. The caller is responsible 
        # for ensuring the IDs are PDB-compliant if this is set to True


    def _add_posres(self, k: float = 1000.0) -> None:
        # create a positional restraint force
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        restrained_non_protein_residues = set()

        for atom in self.topology.atoms():
            res = atom.residue.name
            if atom.element.symbol == 'H' or \
                res in ValidComplex.std_solvent_residues or \
                res in ValidComplex.std_divalent_ion_residues :
                continue
            i = atom.index
            self.restrained.append(i)
            pos = self.positions[i]
            force.addParticle(i, pos.value_in_unit(unit.nanometer))
            # just for reporting non-protein residues
            if res not in ValidComplex.std_protein_residues:
                restrained_non_protein_residues.add(res)
            
        self.system.addForce(force)
        logger.info("positional restraints added to the system:")
        logger.info(f"  on {len(self.restrained)} heavy atoms")
        logger.info(f"  non-protein residue(s): {', '.join(restrained_non_protein_residues)}")
        logger.info(f"  k= {k} kJ/mol/nm**2")

        
    def _create_atom_map(self) -> dict:
        """Create atom map between self.rdmol and source molecule based on connectivity only."""

        target = Chem.RWMol(self.rdmol) # copy
        
        # Use SMARTS with any bonds (~)
        for b in target.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        
        smarts = Chem.MolToSmarts(target).replace("-", "~")
        query = Chem.MolFromSmarts(smarts)
        
        match = self.source.GetSubstructMatch(query)
        if not match:
            raise ValueError("No connectivity match found")
            
        return dict(enumerate(match))


    def _import_coord(self) -> Chem.Mol:
        """Import 3D coordinates from source molecule."""

        target = Chem.RWMol(self.rdmol) # copy
        
        if not self.atom_map:
            self.create_atom_map(self.source)
            
        # Check if molecules have conformers
        if self.source.GetNumConformers() == 0:
            raise ValueError("Source molecule needs coordinates (conformers) first.")
    
        # Ensure the destination molecule has a writable conformer (add one if necessary)
        # The default behavior when setting positions is to add a conformer if none exists
        conf = target.GetConformer(0) if target.GetNumConformers() > 0 else Chem.Conformer(target.GetNumAtoms())
        
        # Iterate over the map numbers and copy positions
        for target_idx, source_idx in self.atom_map.items():
            # Get the position from the source conformer
            pos = self.source.GetConformer(0).GetAtomPosition(source_idx)
            # Set the position in the destination conformer
            conf.SetAtomPosition(target_idx, pos)
                
        # Add the conformer back to the molecule if a new one was created
        if target.GetNumConformers() == 0:
            target.AddConformer(conf, assignId=True)
    
        return target
        


    def _optimize(self) -> Chem.Mol:
        """Optimize the molecule using MMFF94 with positional restraints."""

        mol = Chem.AddHs(self.rdmol, addCoords=True)
        
        conf = mol.GetConformer()
        original_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

        # Get MMFF properties
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties for molecule")
    
        # Create force field
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
        if ff is None:
            raise ValueError("Could not create MMFF force field")

        # Add positional restraints
        restraint_count = 0
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() > 1:
                ff.MMFFAddPositionConstraint(i, self.max_displacement, self.k)
                restraint_count += 1
    
        # Optimize
        initial_energy = ff.CalcEnergy()
        converged = ff.Minimize(maxIts=self.max_iter)
        final_energy = ff.CalcEnergy()
    
        # Calculate RMSD
        optimized_coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        rmsd = np.sqrt(np.mean(np.sum((original_coords - optimized_coords)**2, axis=1)))
        logger.info(f"ligand optimized with MMFF:")
        logger.info(f"  positional restraints on {restraint_count} atoms with k= {self.k} kJ/mol/A^2")
        logger.info(f"  energy initial: {initial_energy:.2f} kcal/mol")
        logger.info(f"  energy final: {final_energy:.2f} kcal/mol")
        logger.info(f"  rmsd from original: {rmsd:.3f} Å")
        
        return mol


    def fix_ligand(self, smiles: str) -> None:
        """Fix stereochemistry of ligand with restrained geometry optimization.

        Args:
            smiles (str): target SMILES.

        Returns:
            None
        """
        logger.info(f"ligand fixed with SMILES:")
        logger.info(f"  {smiles}")
        self.smiles = smiles
        self.rdmol = Chem.MolFromSmiles(smiles)
        self.source = Chem.MolFromPDBBlock(self.mem_ligand.getvalue(), removeHs=True, sanitize=False)
        self.atom_map = self._create_atom_map()
        self.rdmol = self._import_coord()
        self.rdmolH = self._optimize()


    def assign_ligand_charges(self, partial_charge_method: str ='am1bcc') -> None:
        """Assign ligand charges.

        Args:
            partial_charge_method (str, optional): charge assignment method. Defaults to 'am1bcc'.
        
        Returns:
            None
        """
        logger.info(f"partial charges assigned with {partial_charge_method}")
        self.off_mol = Molecule.from_rdkit(self.rdmolH)
        self.off_mol.assign_partial_charges(partial_charge_method=partial_charge_method)
        self.off_mol.to_file(self.mem_ligand_charges, file_format='sdf')

        
    def _add_solvent(self) -> None:
        """Solvate simulation box using addSolvent()."""
        logger.info(f"system solvated with:")
        logger.info(f"  solvent model: {self.solvent}")
        logger.info(f"  box padding: {self.box_padding} nm")
        logger.info(f"  salt concentration: {self.salt_conc} M")
        logger.info(f"  positive ion: {self.positive_ion}")
        logger.info(f"  negative ion: {self.negative_ion}")
        self.modeller.addSolvent(
            self.forcefield,
            model= self.solvent,
            padding= self.box_padding * unit.nanometer,
            ionicStrength= self.salt_conc * unit.molar,
            positiveIon= self.positive_ion,
            negativeIon= self.negative_ion
        )


    def build_system(self,
                    ff_ligand: str = "openff-2.2.1.offxml", # Sage
                    ff_protein: str = "amber/protein.ff14SB.xml",
                    ff_water: str = "amber/tip3p_standard.xml",
                    solvent: str = 'tip3p',
                    box_padding: float = 1.0,
                    salt_conc: float = 0.15, # 0.15 M
                    positive_ion: str = 'Na+',
                    negative_ion: str = 'Cl-',
                    h_mass_factor: float = 3.0,
                    ) -> None:
        """Build Openmm System object.

        Notes:
            System contains:
                - Particles (masses)
                - Forces (bonded, nonbonded, constraints, barostat, etc.)
                - Periodic box vectors
            But, it does NOT contain:
                - Atomic coordinates
                - Velocities
                - Integrator state
                - Random number seeds
                - Thermostat/barostat internal state
            So, saving only the System lets you rebuild a simulation, but not continue it.

        Args:
            ff_ligand (str, optional): forcefield for ligand. Defaults to "openff-2.2.1.offxml".
            ff_protein (str, optional): forcefield for protein. Defaults to "amber/protein.ff14SB.xml".
            ff_water (str, optional): forcefield for water. Defaults to "amber/tip3p_standard.xml".
            solvent (str, optional): solvent model. Defaults to 'tip3p'.
            box_padding (float, optional): simulation box solvation padding. Defaults to 1.0 nm.
            salt_conc (float, optional): salt concentration in M. Defaults to 0.15.
            positive_ion (str, optional): positive ion. Defaults to 'Na+'.
            negative_ion (str, optional): negative ion. Defaults to 'Cl-'.
            h_mass_factor (float, optional): hydrogen mass repartitioning factor. Defaults to 3.0.

        Returns:
            openmm.System: OpenMM SyStem object.
        """
        self.modeller = app.Modeller(
            self.protein_modeller.topology, 
            self.protein_modeller.positions,
            )

        # Add ligand
        self.modeller.add(
            self.off_mol.to_topology().to_openmm(), 
            self.off_mol.conformers[0].to_openmm(),
            )
        
        self._check_clashes(
            self.modeller.topology,
            self.modeller.positions)
        
        # force field
        self.forcefield = app.ForceField(ff_protein, ff_water)
        
        # load OpenFF ligand FF
        smirnoff = SMIRNOFFTemplateGenerator(molecules=[self.off_mol], forcefield=ff_ligand)
        
        self.forcefield.registerTemplateGenerator(smirnoff.generator)

        self.solvent = solvent
        self.box_padding = box_padding
        self.salt_conc = salt_conc
        self.positive_ion = positive_ion
        self.negative_ion = negative_ion
        self._add_solvent()

        self.system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod= app.PME,
            nonbondedCutoff= 1.0 * unit.nanometer,
            constraints= app.HBonds,
            rigidWater= True, # fix water geometry
        )
        logger.info(f"system built with:")
        logger.info(f"  {ff_protein}")
        logger.info(f"  {ff_ligand}")
        logger.info(f"  {ff_water}")

        self.topology = self.modeller.topology
        self.positions = self.modeller.positions
        self.save_complex() # topology & positions
        
        self._add_posres() # posres should be added to system before creating simulation
        self.save_system() # system has the positional restraints info.
        logger.info(f"system saved - {self.parent / f'{self.prefix}_system.xml'}")

        # hydrogen mass repartitioning (HMR)
        self.system_hmr = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod= app.PME,
            nonbondedCutoff= 1.0 * unit.nanometer,
            constraints= app.HBonds,
            rigidWater= True, # fix water geometry
            hydrogenMass= h_mass_factor * 1.008 * unit.amu,
        )
        logger.info(f"hydrogen mass repartitioning applied with factor {h_mass_factor}")
        # HMR stage does not require posres
        self.save_system(hmr=True) # system has the positional restraints info.
        logger.info(f"system saved - {self.parent / f'{self.prefix}_system_hmr.xml'}")


    def save_protein(self) -> None:
        """Save the fixed protein to a PDB file.

        Returns:
            None
        """
        filename = self.parent / f'{self.prefix}_protein.pdb'
        with open(filename, "w") as f:
            app.PDBFile.writeFile(
                self.protein_modeller.topology,
                self.protein_modeller.positions,
                f,
                keepIds=True
            )


    def save_ligand(self) -> None:
        """Save the optimized (charged) ligand to an SDF file.

        Returns: 
            None
        """
        if not self.rdmolH:
            raise ValueError("we may need to fix the ligand first. use fix_ligand()")
        
        filename = self.parent / f'{self.prefix}_ligand.sdf'

        if len(self.mem_ligand_charges.getvalue()) > 0:
            self.off_mol.to_file(filename, file_format='sdf')
        else:
            off_mol = Molecule.from_rdkit(self.rdmolH)
            off_mol.to_file(filename, file_format='sdf')

    
    def save_complex(self) -> None:
        """Save the fixed protein to a PDB file.

        Returns:
            None
        """
        filename = self.parent / f'{self.prefix}_complex.pdb'
        with open(filename, "w") as f:
            app.PDBFile.writeFile(
                self.topology,
                self.positions,
                f,
                keepIds=True
            )


    def load_complex(self) -> bool:
        filename = self.parent / f'{self.prefix}_complex.pdb'
        if not filename.exists():
            return False
        
        pdb = app.PDBFile(filename.as_posix())

        self.topology = pdb.topology
        self.positions = pdb.positions

        return True
    

    def save_system(self, hmr: bool = False) -> None:
        # system contains positional restraints (CustomExternalForce)
        if hmr:
            filename = self.parent / f"{self.prefix}_system_hmr.xml"
            with open(filename, "w") as f:
                f.write(XmlSerializer.serialize(self.system_hmr))
        else:
            filename = self.parent / f"{self.prefix}_system.xml"
            with open(filename, "w") as f:
                f.write(XmlSerializer.serialize(self.system))

    
    def load_system(self, hmr: bool = False) -> bool:
        if hmr:
            filename = self.parent / f"{self.prefix}_system_hmr.xml"
        else:
            filename = self.parent / f"{self.prefix}_system.xml"
        if not filename.exists():
            return False
        with open(filename, "r") as f:
            self.system = XmlSerializer.deserialize(f.read())
        return True
    

    def _get_bonded_atom_pairs(self, topology) -> list:
        bonded_12 = [tuple(sorted([bond.atom1.index, bond.atom2.index])) for bond in topology.bonds()]
        return bonded_12
    

    def _get_13_14_atom_pairs(self, topology) -> tuple:
        # Step 1: Build an adjacency list (1-2 bonds)
        adj = {}
        for bond in topology.bonds():
            i, j = bond.atom1.index, bond.atom2.index
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)

        # Step 2: Find 1-3 pairs (neighbors of neighbors)
        pairs_13 = set()
        for b in adj:
            for middle_atom in adj[b]:
                for c in adj[middle_atom]:
                    if c != b: # Exclude the starting atom (1-1)
                        # Use a sorted tuple to avoid (i,j) and (j,i) duplicates
                        pair = tuple(sorted((b, c)))
                        # Ensure it's not a 1-2 bond (e.g., in rings)
                        if pair[1] not in adj[pair[0]]:
                            pairs_13.add(pair)
        pairs_14 = set()
        for a in adj:
            for b in adj[a]:
                for c in adj[b]:
                    if c == a: continue
                    for d in adj[c]:
                        # Exclude self (1-1), 1-2 (d==b), and 1-3 (d==a)
                        if d != a and d != b:
                            # Standardize order to avoid duplicates (i,j) vs (j,i)
                            pair = tuple(sorted((a, d)))
                            # Verify the distance is exactly 3 bonds
                            # (Relevant for rings where 1-4 might also be 1-2 or 1-3)
                            is_shorter = (d in adj[a]) or any(d in adj[n] for n in adj[a])
                            if not is_shorter:
                                pairs_14.add(pair)
                                
        return list(pairs_13), list(pairs_14)


    def _check_clashes(self, topology, positions, threshold: float = 1.5) -> float:
        """Check for steric clashes in the system.
        Notes:
            Van der Waals radii (in Angstroms): H: 1.20, C: 1.70, N: 1.55, O: 1.52
            Bond lengths (in Angstroms): C-H: 1.09, C-C: 1.54, C-N: 1.47, C-O: 1.43, N-H: 1.01, O-H: 0.96

        Args:
            threshold (float, optional): distance threshold to consider as close contact. Defaults to 1.5 A.
        """
        from scipy.spatial.distance import pdist, squareform # SciPy is efficient

        positions_A = positions.value_in_unit(unit.angstrom)
        # Calculate all pairwise distances (efficiently using broadcasting)
        # For specific residues, you might need to slice the array first
        # distances_matrix = np.linalg.norm(positions_np[indices1][:, np.newaxis, :] - positions_np[indices2], axis=2)
        
        distances_flat = pdist(positions_A)
        distances_matrix = squareform(distances_flat)

        # Assign the new value to the lower triangle
        lower_indices = np.tril_indices_from(distances_matrix, k=0) # diagonal included 
        distances_matrix[lower_indices] = 100.0

        # indices where distance < cutoff
        close_atom_pair_indices = np.where(distances_matrix < threshold)

        bonded = [tuple(sorted([bond.atom1.index, bond.atom2.index])) for bond in topology.bonds()]
        bonded_13, bonded_14 = self._get_13_14_atom_pairs(topology)

        # close atom pair indices excluding bonded (1-2), 1-3, and 1-4
        clashing_atoms = [(i, j) for i, j in zip(*close_atom_pair_indices) if not ((i,j) in bonded or (i,j) in bonded_13 or (i,j) in bonded_14)]

        logger.info(f"clashing atom pairs (distance < {threshold} A): {len(clashing_atoms)}")
        for i, j in clashing_atoms:
            a1 = list(topology.atoms())[i]
            a2 = list(topology.atoms())[j]
            a1_id = f'{a1.residue.name:<4} {a1.residue.id:<4} {a1.name:<4}'
            a2_id = f'{a2.residue.name:<4} {a2.residue.id:<4} {a2.name:<4}'
            d = distances_matrix[i, j] # Get actual distance for reporting
            logger.info(f"  {a1_id} & {a2_id}: {d:.3f} A")