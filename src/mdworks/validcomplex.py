__all__ = ['ValidComplex',]

import io
import gzip
import logging
import numpy as np

from pathlib import Path
from typing import Iterable
from importlib.metadata import version

try:
    from openff.toolkit.topology.molecule import Molecule
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openmm import app, unit, CustomExternalForce, NonbondedForce
except ImportError:
    raise ImportError("install openmm, openmmforcefields, and openff-toolkit from conda-forge.\n")

from .simfileio import SimFileIO
from .utils import setup_logger


logger = logging.getLogger(__name__)


class ValidComplex(SimFileIO):
    """Class for preparing valid protein/ligand complex structure.
    
    Issues with OpenFold3
        - OpenFold3 generates flat geometries instead of tetrahedral for undefined chiral center(s)
        - OpenFold3 does not generate hydrogen atoms

    Issues with OpenEye Spruce
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
    
    def __init__(self, in_file: str | Path, workdir: Path | str | None = None, quiet: bool = False):
        """Initialize ValidComplex class object.

        Args:
            in_file (str | Path): input complex structure in PDB or MMCIF format.
            workdir (Path | str | None, optional): working directory. Defaults to None.
            quiet (bool, optional): whether to suppress logging. Defaults to False.
        """
        assert isinstance(in_file, str) or isinstance(in_file, Path)
        in_path = Path(in_file)
        assert in_path.exists()

        # setup prefix and workdir
        # remove all extensions and get the true stem: ex. x.pdb.gz -> x
        self.prefix : str = in_path.name.removesuffix("".join(in_path.suffixes))
        if isinstance(workdir, str) or isinstance(workdir, Path):
            self.workdir = Path(workdir)
            self.workdir.mkdir(exist_ok=True)
        else:
            self.workdir = in_path.parent

        # self.mem_protein : io.StringIO = io.StringIO()
        # self.mem_ligand : io.StringIO = io.StringIO()
        # self.mem_ligand_charges: io.StringIO = io.StringIO()
        
        # ligand
        self.off_mol : Molecule = Molecule()
        self.off_mol_list : list[Molecule] = [] # for structure with multiple ligands
        
        # solvent
        self.solvent : str = 'tip3p'
        self.solvent_implicit : bool = False

        # self.protein_modeller = None
        # self.ligand_modeller = None
        self.modeller = None
        self.restrained = []
        self.system = None

        extension = "".join(in_path.suffixes).lower()

        if extension == ".cif":
            st = app.PDBxFile(str(in_path))
            self.modeller = app.Modeller(st.getTopology(), st.getPositions())

        elif extension == ".pdb":
            st = app.PDBFile(str(in_path))
            self.modeller = app.Modeller(st.getTopology(), st.getPositions())

        elif extension == ".cif.gz":
            with gzip.open(in_file, "rt") as f:
                st = app.PDBxFile(f)
                self.modeller = app.Modeller(st.getTopology(), st.getPositions())
                # pdbxfile: file-like object from which the PDBx/mmCIF file is to be read
        
        elif extension == ".pdb.gz":
            with gzip.open(in_file, "rt") as f:
                st = app.PDBFile(f)
                self.modeller = app.Modeller(st.getTopology(), st.getPositions())
        
        setup_logger(logger, self.workdir, self.prefix, quiet=quiet)

        logger.info(f"mdworks {version('mdworks')}")
        logger.info(f"pdbfixer {version('pdbfixer')}")
        logger.info(f"openmmforcefields {version('openmmforcefields')}")
        logger.info(f"openff-toolkit {version('openff-toolkit')}")
        logger.info(f"rdkit {version('rdkit')}")
        logger.info(f"scipy {version('scipy')}")
        logger.info(f"workdir= {self.workdir}")
        logger.info(f"prefix= {self.prefix}")
        
        # self._sort_protein_and_ligand_residues()

        # check prepared receptor and ligand files
        upstream_prefix = self.prefix.replace('_complex', '')

        # look for sdf
        self.ligand_sdf = list(self.workdir.glob(f'{upstream_prefix}_*.sdf'))[0]
        self.ligand_resname  = self.ligand_sdf.name.replace('.sdf','').replace(f'{upstream_prefix}_', '')
        self.off_mol_list = Molecule.from_file(self.ligand_sdf, file_format="sdf")
        if len(self.off_mol_list) > 1:
            logger.info(f"  multiple molecules found in ligand: {self.ligand_resname} {len(self.off_mol_list)}")


    # def _sort_protein_and_ligand_residues(self) -> None:
    #     """Sort protein and ligand."""
    #     groups: list[str] = ['protein', 'solvent', 'divalent', 'ligand']
    #     residue_group = {k: set() for k in groups}
    #     for res in self.fixer.topology.residues():
    #         if res.name in ValidComplex.std_protein_residues:
    #             residue_group['protein'].add(res.name)
    #         elif res.name in ValidComplex.std_solvent_residues:
    #             residue_group['solvent'].add(res.name)
    #         elif res.name in ValidComplex.std_divalent_ion_residues:
    #             residue_group['divalent'].add(res.name)
    #         else:
    #             residue_group['ligand'].add(res.name)
    #             logger.info(f"ligand residue found {res.name}")

    #     # protein (including structural divalent ions)
    #     logger.info("protein preparation:")
    #     self.protein_modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
    #     residues_to_delete = []
    #     for res in self.protein_modeller.topology.residues():
    #         if (self.remove_solvent and (res.name in residue_group['solvent'])) or \
    #             (res.name in residue_group['ligand']):
    #             logger.info(f"  deleting: {res}")
    #             residues_to_delete.append(res)
    #     self.protein_modeller.delete(residues_to_delete)

    #     # non-standard residues        
    #     for res in self.protein_modeller.topology.residues():
    #         if res.name not in ValidComplex.std_protein_residues:
    #             logger.info(f"  including non-protein: {res}")
    #     logger.info(f"  number of residues: {self.protein_modeller.topology.getNumResidues()}")
    #     self._check_clashes(
    #         self.protein_modeller.topology, 
    #         self.protein_modeller.positions)

    #     # ligand
    #     logger.info("ligand preparation:")
    #     self.ligand_modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
    #     residues_to_delete = []
    #     for res in self.ligand_modeller.topology.residues():
    #         if res.name in residue_group['protein'] or \
    #            res.name in residue_group['solvent'] or \
    #            res.name in residue_group['divalent']:
    #             residues_to_delete.append(res)
    #         else:
    #             logger.info(f"  keeping: {res}")
    #             self.off_mol_info.append({'name': res.name, 'id': res.id, 'chain': res.chain.id})

    #     self.ligand_modeller.delete(residues_to_delete)
    #     logger.info(f"  number of residues: {self.ligand_modeller.topology.getNumResidues()}")

    #     app.PDBFile.writeFile(
    #         self.ligand_modeller.topology,
    #         self.ligand_modeller.positions,
    #         self.mem_ligand,
    #         keepIds=False
    #     )
    #     # keepIds (optional): A boolean value (default is False). 
    #     # If True, the residue and chain IDs specified in the Topology are used; 
    #     # otherwise, new ones are generated. The caller is responsible 
    #     # for ensuring the IDs are PDB-compliant if this is set to True


    def _add_posres(self, k: float = 1000.0, exclude: Iterable | None = None) -> None:
        # create a positional restraint force
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        restrained_non_protein_residues = set()

        # exclude N-/C-terminal residues
        # terminals = []
        # for chain in self.topology.chains():
        #     residues = list(chain.residues())
        #     if not residues:
        #         continue
        #     is_protein = any(atom.name == 'CA' for atom in residues[0].atoms())
        #     if not is_protein:
        #         continue
        #     terminals.append(residues[0]) # N-ter
        #     terminals.append(residues[-1]) # C-ter

        for atom in self.topology.atoms():
            chainid = atom.residue.chain.id
            resseq = int(atom.residue.id)
            resname = atom.residue.name
            res_id = (chainid, resname, resseq)
            if atom.element.symbol == 'H' or \
                resname in ValidComplex.std_solvent_residues or \
                resname in ValidComplex.std_divalent_ion_residues or \
                (exclude is not None and res_id in exclude):
                # atom.residue in terminals:
                continue
            i = atom.index
            self.restrained.append(i)
            pos = self.positions[i]
            force.addParticle(i, pos.value_in_unit(unit.nanometer))
            # just for reporting non-protein residues
            if resname not in ValidComplex.std_protein_residues:
                restrained_non_protein_residues.add(resname)
            
        self.system.addForce(force)
        logger.info("positional restraints added to the system:")
        logger.info(f"  on {len(self.restrained)} heavy atoms")
        logger.info(f"  non-protein residue(s): {', '.join(restrained_non_protein_residues)}")
        logger.info(f"  k= {k} kJ/mol/nm**2")

        

    def assign_ligand_charges(self, partial_charge_method: str ='nagl') -> None:
        """Assign ligand charges.

        Args:
            partial_charge_method (str, optional): charge assignment method. Defaults to 'am1bcc'.
                set 'import' to load saved charges in the sdf file.
        """
        self.off_mol = self.off_mol_list[0]

        logger.info(f"partial charges assigned with {partial_charge_method}")

        if partial_charge_method in ['am1bcc', 'gasteiger', 'mmff94', 'mmff94s']:        
            self.off_mol.assign_partial_charges(partial_charge_method= partial_charge_method)
        elif partial_charge_method == 'nagl':
            nagl_wrapper = NAGLToolkitWrapper()
            self.off_mol.assign_partial_charges(
                partial_charge_method="openff-gnn-am1bcc-1.0.0.pt",
                toolkit_registry=nagl_wrapper
            )
        elif partial_charge_method == 'import':
            pass

        if self.off_mol.partial_charges is None or len(self.off_mol.partial_charges) == 0:
            raise ValueError("partial charges are not assigned")
        
        logger.info(f"copying partial charges to {len(self.off_mol_list)} ligand molecule(s)..")

        for mol in self.off_mol_list:
            mol.partial_charges = self.off_mol.partial_charges

        # if multiple molecules are present for ligand,
        # save them all in a single SDF file, which can be read by OpenFF Toolkit.
        # partial charges are cloned from the first molecule to all other molecules in the SDF file.
        # Get the partial charges (includes units, e.g., elementary_charge)

        with open(self.ligand_sdf, "w") as f:
            for mol in self.off_mol_list:
                mol.to_file(f, file_format='sdf')
        

    def _add_explicit_solvent(self) -> None:
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

    
    def build(self,
              ff_ligand: str = "openff-2.2.1.offxml", # Sage
              ff_protein: str = "amber/protein.ff14SB.xml",
              ff_water: str = "amber/tip3p_standard.xml",
              solvent: str = "tip3p",
              box_padding: float = 1.0, # 1.0 nm
              salt_conc: float = 0.15, # 0.15 M
              positive_ion: str = 'Na+',
              negative_ion: str = 'Cl-',
              h_mass_factor: float = 3.0,
              posres: float = 1000.0,
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
        off_mol_res = []
        off_mol_res_info = []
        # self.modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
        for res in self.modeller.topology.residues():
            if res.name == self.ligand_resname:
                off_mol_res_info.append({'name': res.name, 'id': res.id, 'chain': res.chain.id})
                off_mol_res.append(res)
        self.modeller.delete(off_mol_res)

        if self.off_mol_list:
            logger.info(f"adding ligand(s) to system..")
            # Rebuild the complex so ligand connectivity is preserved — 
            # e.g. load the ligand from an SDF/MOL2 with RDKit 
            # or OpenFF (Molecule.from_file(...), which carries bond info natively), 
            # convert to an OpenMM topology (molecule.to_topology().to_openmm()), 
            # and merge it with the protein topology via Modeller.add(ligand_topology, ligand_positions)
            for i, mol in enumerate(self.off_mol_list):
                if mol.partial_charges is None or len(mol.partial_charges) == 0:
                    raise ValueError("Ligand molecule must have partial charges assigned before adding to the system.")
                ligand_topology = mol.to_topology().to_openmm()
                ligand_positions = mol.conformers[0].to_openmm()
                for chain in ligand_topology.chains():
                    chain.id = off_mol_res_info[i]['chain'] # preserve original chain id
                for residue in ligand_topology.residues():
                    residue.name = off_mol_res_info[i]['name'] # preserve original residue name
                    residue.id = off_mol_res_info[i]['id']     # preserve original residue id  
                # preserve chain id and residue name/id before adding to the modeller
                # for chain, origin in zip(ligand_topology.chains(), self.ligand_modeller.topology.chains()):
                #     chain.id = origin.id
                # for residue, origin in zip(ligand_topology.residues(), self.ligand_modeller.topology.residues()):
                #     residue.name = origin.name # 3-4 letter code, whatever convention you use
                #     residue.id = origin.id     # residue number/seqid, as a string
                self.modeller.add(ligand_topology, ligand_positions)
                logger.info(f"  {off_mol_res_info[i]['name']} {off_mol_res_info[i]['id']} in chain {off_mol_res_info[i]['chain']}")

        self._check_clashes(
            self.modeller.topology,
            self.modeller.positions)
        
        self.solvent = solvent

        if solvent in ['gbn2', 'obc2', 'gbn1', 'obc1']:
            self.solvent_implicit = True
        else:
            self.solvent_implicit = False

        # force field
        if self.solvent_implicit:
            # implicit solvent model
            # Zn and other divalent ions are not supported in implicit solvent model and requires explicit solvent model.
            self.forcefield = app.ForceField(ff_protein, ff_water, f"implicit/{solvent}.xml")
        else:
            self.forcefield = app.ForceField(ff_protein, ff_water)
        
        # load OpenFF ligand FF
        smirnoff = SMIRNOFFTemplateGenerator(molecules=[self.off_mol], forcefield=ff_ligand)
        self.forcefield.registerTemplateGenerator(smirnoff.generator)

        if self.solvent_implicit:
            # implicit solvent model
            self.system = self.forcefield.createSystem(
                self.modeller.topology,
                nonbondedMethod= app.CutoffNonPeriodic,
                nonbondedCutoff= 1.0 * unit.nanometer,
                constraints= app.HBonds,
                soluteDielectric= 1.0, # default interior dielectric constant for protein
                solventDielectric= 78.5, # default exterior dielectric constant for bulk water
                # implicitSolventSaltConc = salt_conc * unit.molar,
            )
            logger.info(f"system built with:")
            logger.info(f"  {ff_protein}")
            logger.info(f"  {ff_ligand}")
            logger.info(f"  implicit/{self.solvent}")

        else:
            # explicit solvent model
            self.box_padding = box_padding
            self.salt_conc = salt_conc
            self.positive_ion = positive_ion
            self.negative_ion = negative_ion
            self._add_explicit_solvent()

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

        # 3. Manually reassign the Chain ID and Residue Number
        for chain in self.topology.chains():
            for res in chain.residues():
                # Find the ligand (OpenMM will name it UNK if it reset it, or keep UNL)
                if res.name in ['UNK', 'UNL']: 
                    chain.id = 'L'       # Assign your desired Chain ID string
                    res.id = '1'         # Assign your desired Residue Number string
                    res.name = 'UNL'     # Explicitly correct the name if it became UNK

        self.save_complex() # topology & positions
        
        self._add_posres(k= posres) # posres should be added to system before creating simulation
        self.save_system() # system has the positional restraints info.

        logger.info(f"system saved - {self.workdir / f'{self.prefix}_system.xml'}")

        # hydrogen mass repartitioning (HMR)
        # ONLY for explicit solvent model.
        if not self.solvent_implicit:
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
            logger.info(f"system saved - {self.workdir / f'{self.prefix}_system_hmr.xml'}")
                

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


    def _get_excluded_pairs(self) -> set[tuple[int, int]]:
        """
        When the ligand was parameterized (GAFF/OpenFF/etc.), the force field computed 
        the correct 1-2/1-3/1-4 exclusion list as NonbondedForce exceptions 
        (typically chargeProd = 0, epsilon = 0, or scaled for 1-4). 
        """
        nonbonded = next(f for f in self.system.getForces() if isinstance(f, NonbondedForce))
        excluded = set()
        for idx in range(nonbonded.getNumExceptions()):
            i, j, chargeProd, sigma, epsilon = nonbonded.getExceptionParameters(idx)
            excluded.add(tuple(sorted((i, j))))
        return excluded

    def _check_clashes(self, topology, positions, threshold: float = 1.5) -> None:
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
        close_atoms = [(i, j) for i, j in zip(*close_atom_pair_indices) if not ((i,j) in bonded or (i,j) in bonded_13 or (i,j) in bonded_14)]

        logger.info(f"  close atom pairs (distance < {threshold} A): {len(close_atoms)}")
        for i, j in close_atoms:
            a1 = list(topology.atoms())[i]
            a2 = list(topology.atoms())[j]
            a1_id = f'{a1.residue.name:<4} {a1.residue.id:<4} {a1.name:<4}'
            a2_id = f'{a2.residue.name:<4} {a2.residue.id:<4} {a2.name:<4}'
            d = distances_matrix[i, j] # Get actual distance for reporting
            logger.info(f"  {a1_id} & {a2_id}: {d:.3f} A")


    def _check_ligand_topology(self, topology) -> None:
        indices = {a.index for a in topology.atoms()}
        bonds = [b for b in topology.bonds() if b.atom1.index in indices and b.atom2.index in indices]
        logger.info(f"checking ligand topology: atoms={len(indices)} bonds={len(bonds)}")  
        # bonds should be roughly atoms-1 to atoms+ring_count

    
    # def summary(self) -> None:
    #     lines = []
    #     for chain in self.modeller.topology.chains():
    #         residues = [f"{res.name:>3} {res.seqid.num:>4}" for res in chain.residues()]
    #         n = len(residues)
    #         if n == 1:
    #             lines.append(f"chain {chain.name} ({n:>3} residues): {residues[0]:<8}")
    #         else:
    #             lines.append(f"chain {chain.name} ({n:>3} residues): {residues[0]:<8} ... {residues[-1]}")
    #         # non standard residues
    #         for res in chain.residues():
    #             if res.name in ValidComplex.std_protein_residues:
    #                 continue
    #             elif res.name in ValidComplex.std_solvent_residues:
    #                 continue
    #             elif res.name in ValidComplex.std_divalent_ion_residues:
    #                 continue
    #             else:
    #                 lines.append(f"    non-standard residue  {res.name:>3} {res.seqid.num:>4}")

    #     print("\n".join(lines))