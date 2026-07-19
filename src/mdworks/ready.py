from pathlib import Path
from pdbfixer import PDBFixer  
from openmm.app import PDBFile, Topology, Modeller
from openmm.unit import angstroms, nanometers
from pdb2pqr.main import main_driver, build_main_parser

import re
import string
import shutil
import subprocess
import numpy
import logging

import gemmi

from .editor import PDBEditor
from .utils import setup_logger


logger = logging.getLogger(__name__)


def isolate(
        filename: str | Path, 
        ligand_resname: str, 
        output_ligand_file: str | Path) -> None:
    """
    Extract ligand from PDB file and save to a separate PDB file.
    """
    filename = Path(filename)
    output_ligand_file = Path(output_ligand_file)

    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist")

    fixer = PDBFixer(filename= filename.as_posix())
    ligand_atoms = [atom for atom in fixer.topology.atoms() if atom.residue.name == ligand_resname]

    if not ligand_atoms:
        raise ValueError(f"No atoms found for ligand with resname '{ligand_resname}' in {filename}")

    # Create a new topology and positions for the ligand
    ligand_topology = Topology()
    new_chain = ligand_topology.addChain()
    new_residue = ligand_topology.addResidue(ligand_resname, new_chain)

    ligand_positions = []
    atom_mapping = {}

    for atom in ligand_atoms:
        new_atom = ligand_topology.addAtom(atom.name, atom.element, new_residue)
        ligand_positions.append(fixer.positions[atom.index].value_in_unit(angstroms))
        atom_mapping[atom] = new_atom
    
    ligand_positions = numpy.array(ligand_positions)

    for bond in fixer.topology.bonds():
        if bond.atom1 in atom_mapping and bond.atom2 in atom_mapping:
            ligand_topology.addBond(atom_mapping[bond.atom1], atom_mapping[bond.atom2])

    # Write the ligand to a new PDB file
    with open(output_ligand_file, "w") as f:
        PDBFile.writeFile(ligand_topology, ligand_positions, f, keepIds=True)



def merge(protein_pdb: str | Path, 
          ligand_pdb: str | Path, 
          complex_path: str | Path) -> None:
    """
    Programmatically merges the fixed protein and original ligand structuresusing OpenMM's Modeller framework 
    instead of raw string manipulation.
    """
    # 1. Load the fixed and protonated protein 
    receptor = PDBFile(protein_pdb)
    
    # 2. Load the isolated ligand file
    ligand = PDBFile(ligand_pdb)
    
    # 3. Initialize the Modeller object with the protein structure
    modeller = Modeller(receptor.topology, receptor.positions)
    
    # 4. Programmatically combine the ligand's topology and spatial coordinates
    modeller.add(ligand.topology, ligand.positions)
    
    # 5. Write the combined system out to a standard, compliant PDB file
    with open(complex_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


def complex(
        filename: str | None = None, 
        pdb_id: str | None = None,
        ligand_resname: str | None = None,
        obabel: str = shutil.which("obabel"),
        target_pH: float = 7.4,
        quiet: bool = False) -> None:  
    """  
    Fix complex/receptor structural issues and set protonation states.
    PDBFixer/PDB2PQR workflow excludes non-standard residus including ligands, cofactors, and water molecules.
    So, if the receptor structure contains a ligand, it should be extracted and processed separately.
    """  
    if filename:
        p = Path(filename)
        output_prefix = p.name.removesuffix("".join(p.suffixes))
        workdir = p.parent
    elif pdb_id:
        output_prefix = pdb_id
        workdir = Path.cwd()
    else:
        raise ValueError("Either filename or pdb_id must be provided")

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    fixed_pdb_path = f"{output_prefix}_fixed.pdb"  
    final_pqr_path = f"{output_prefix}_H.pqr"  
    final_pdb_path = f"{output_prefix}_H.pdb"

    complex_path = f"{output_prefix}_complex.pdb"

    if filename:
        logger.info(f"PDBFixer reading a file: {filename}")
        fixer = PDBFixer(filename= filename.as_posix())
    elif pdb_id:
        logger.info(f"PDBFixer downloading the PDB {pdb_id} structure from RCSB Protein Data Bank")
        fixer = PDBFixer(pdbid= pdb_id)

    if ligand_resname:
        ligand_pdb = f"{output_prefix}_{ligand_resname}.pdb"
        ligand_smi = f"{output_prefix}_{ligand_resname}.smi"
        logger.info(f"[Step 0] Isolating Ligand {ligand_resname} ...")
        isolate(filename or f"{pdb_id}.pdb", ligand_resname, ligand_pdb)
        logger.info(f"Isolated ligand {ligand_resname} saved to {ligand_pdb}")
        try:
            result = subprocess.run([obabel, "-ipdb", ligand_pdb, "-osmi"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True
                                    )
            output = result.stdout.strip()
            if output:
                # ex. <SMILES> <Name>
                smiles, name = output.split(maxsplit=1)
                with open(ligand_smi, "w") as f:
                    f.write(f"{smiles}\n")
                    logger.info(f"Isolated ligand {ligand_resname} saved to {ligand_smi}")
                return smiles
            else:
                raise ValueError(f"Could not guess ligand SMILES.")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error occurred while running obabel: {e}")
    
    # logger.info(f"[Step 0] Disconnect bonds with Zinc atom(s) ...")
    # # Track down the Zinc atom index
    # zinc_atom_indices = [
    #     atom.index for atom in fixer.topology.atoms() if "ZN" in atom.name.upper()
    # ]
    # # Rebuild the bond network, explicitly excluding any bonds involving the Zinc atom
    # clean_bonds = []
    # for bond in fixer.topology.bonds():
    #     if bond[0].index not in zinc_atom_indices and bond[1].index not in zinc_atom_indices:
    #         clean_bonds.append(bond)
    # Overwrite the topology's bond dictionary with the filtered list
    # fixer.topology._bonds = clean_bonds

    logger.info(f"[Step 1] Fetching and Fixing via PDBFixer ...")

    # PDBFixer uses geometry template to fill in missing residues and atoms, 
    # and to replace nonstandard residues with standard ones.
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    # Add missing heavy atoms (but do not add hydrogens yet; PDB2PQR will do that)  

    # The missingResidues dictionary stores missing residues as: 
    # (chainIndex, residueIndex): [list of residue names]
    # example:
    # {
    #   (0, 0): ['MET', 'GLU', ..., 'PRO', 'SER']), 
    #   (0, 108): ['PRO', 'VAL', ... , 'VAL'], 
    #   (0, 232): ['VAL', ..., 'ARG', 'LEU']
    # }

    chains = list(fixer.topology.chains())
    residues = list(fixer.topology.residues())
    skipped_missing_residues = {}

    for key, resnames in sorted(fixer.missingResidues.items()):
        (chain_idx, res_idx) = key
        # residues will be inserted at res_idx in chain_idx
        chain = chains[chain_idx]
        residue = residues[res_idx]
        n = len(resnames)
        at = int(residue.id)
        n_present = len(list(chain.residues()))
        is_n_terminal = res_idx == 0
        is_c_terminal = res_idx == n_present
        if is_n_terminal or is_c_terminal:
            skipped_missing_residues[key] = fixer.missingResidues.pop(key)
            if is_n_terminal:
                logger.info(f"{n} residues missing at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (N-ter; skipped)")
            if is_c_terminal:
                logger.info(f"{n} residues missing at {chain.id}:{at:<4d} {','.join(resnames[:3])}.. (C-ter; skipped)")
        else:
            logger.info(f"{n} residues missing at {chain.id}:{at:<4d} {','.join(resnames[:3])}..")
    
    fixer.addMissingAtoms()
    fixer.removeChains(chainIndices=[-1])
    
    # Write intermediate fixed heavy-atom structure  
    with open(fixed_pdb_path, "w") as f:  
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)  
        logger.info(f"PDBFixer fixed heavy atoms and saved to {fixed_pdb_path}")

    logger.info(f"[Step 2] Predicting pKa and Protonating via PDB2PQR at pH {target_pH} ...")  
    # Build PDB2PQR command-line arguments to run programmatically  
    # Using AMBER forcefield naming convention for downstream MD compatibility  
    pdb2pqr_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}", 
        fixed_pdb_path, 
        final_pqr_path]
    # Correct programmatic call sequence: parse options list via internal parser object
    parser = build_main_parser()
    parsed_args = parser.parse_args(pdb2pqr_args)
    # Execute driver engine
    main_driver(parsed_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR protonated PQR file and saved to {final_pqr_path}")
    logger.info(f"[Step 3] Generating clean, final PDB structure with hydrogens via PDB2PQR ...")
    
    # Generate companion structural pdb containing the newly calculated hydrogens
    pdb_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}",
        f"--pdb-output={final_pdb_path}",
        fixed_pdb_path,
        final_pqr_path]
    parsed_pdb_args = parser.parse_args(pdb_args)
    main_driver(parsed_pdb_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR saved final receptor to {final_pdb_path}")

    if ligand_resname:
        logger.info(f"[Step 4] Merging final receptor and original ligand {ligand_resname} ...")
        merge(final_pdb_path, ligand_output_file, complex_path)
        logger.info(f"Merged complex saved to {complex_path}")



def guess_smiles_from_pdb(filename: str, obabel: str = shutil.which("obabel")) -> str:
    try:
        p = Path(filename)
        prefix = p.name.removesuffix("".join(p.suffixes))
        assert p.exists(), "file not found"
        result = subprocess.run([obabel, "-ipdb", filename, "-osmi"], 
                                capture_output=True, 
                                text=True, 
                                check=True
                                )
        output = result.stdout.strip()
        if output:
            # ex. <SMILES> <Name>
            smiles, name = output.split(maxsplit=1)
            with open(f"{prefix}.smi", "w") as f:
                f.write(f"{smiles}\n")
            return smiles
        else:
            raise ValueError(f"Could not guess SMILES from {filename}.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error occurred while running obabel on {filename}: {e}")



def parse_selection(spec_string: str) -> list[tuple]:
    """Parse residue range expressions

    Args:
        spec_string (str): example - "A:10-30,A:100-120,B:1-50,C:1,D"

    Returns:
        list : [(chain_id, int(start), int(end)), ...]
    """
    # pattern = r"^(?:#(?P<model>\d+(?:\.\d+)*))?(?:\/(?P<chain>[A-Za-z0-9]+))?(?::(?P<residue>\d+(?:-\d+)?))?(?:@(?P<atom>[A-Za-z0-9*?]+))?$"
    pattern = r"^(?P<chain>[A-Za-z0-9]+)?(?::(?P<residue>\d+(?:-\d+)?))?$"

    parsed_specs = []
    for sub_spec_string in spec_string.split(","):
        match = re.match(pattern, sub_spec_string)
        if match:
            d = match.groupdict()
            chain_id = d.get('chain', None)
            i = None
            j = None
            if d.get('residue'):
                ij = d.get('residue').split('-')
                if len(ij) == 2:
                    i = int(ij[0])
                    j = int(ij[1])
                elif len(ij) == 1:
                    i = int(ij[0])
                    j = None
            parsed_specs.append((chain_id, i, j))

    return parsed_specs



def parse_mapping(spec_string: str) -> dict:
    """Parse chain id mapping expressions

    Args:
        spec_string (str): example - "A:B,B:A,X:C"

    Returns:
        dict : {old_chain_id : new_chain_id, ...}
    """
    pattern = r"^(?P<old_chain>[A-Za-z0-9]+):(?P<new_chain>[A-Za-z0-9]+)$"
    parsed_specs = {}
    for sub_spec_string in spec_string.split(","):
        match = re.match(pattern, sub_spec_string)
        if match:
            d = match.groupdict()
            old = d.get('old_chain')
            new = d.get('new_chain')
            parsed_specs[old] = new
    return parsed_specs


def cut(filename: str, selection: str, quiet: bool = False) -> None:
    """Cut residues

    Args:
        filename (str): Input .pdb or .cif file
        selection (str): Selection of chain:residues. 
            Example: `A:1-10,A:130-150,B:1-30`
            - chain and residue range is separated by `:`
            - residue range is defined by `-`
            - multiple selections are separated by `,`
        output_prefix (str): Output prefix
        quiet (bool): If True, no output is shown.
    """
    p = Path(filename)
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    workdir = p.parent
    targets = parse_selection(selection)
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_cut.pdb"  

    with open(filename, "r") as f:
        logger.info(f"PDBFixer reading a PDB file: {filename}")
        pdb = PDBFixer(pdbfile=f)
        modeller = Modeller(pdb.topology, pdb.positions)
        
    residues_to_delete = []
    for chain in modeller.topology.chains():
        for residue in chain.residues():
            resseq = int(residue.id)
            for (chain_id, i, j) in targets:
                if all([
                    chain_id is None or chain_id == chain.id,
                    i is None or (i <= resseq),
                    j is None or (j >= resseq),
                    ]):
                    residues_to_delete.append(residue)
    modeller.delete(residues_to_delete)

    # Write intermediate fixed heavy-atom structure  
    with open(output_filename, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Modified coordinates are saved to {output_filename}")


def summary(filename: str, quiet: bool = False):
    """Show summary of structure"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    print(PDBEditor.load(filename).summary())


def rename(filename: str, chain_map: str, quiet: bool = False):
    """Rename chain id"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    output_filename = f"{output_prefix}_rename.pdb"
    st = PDBEditor.load(filename)
    parsed_map = parse_mapping(chain_map)
    
    # Resolve potential conflicts
    old_ids = list(parsed_map.keys())
    new_ids = list(parsed_map.values())
    chain_ids = st.chain_names()
    assert set(old_ids).issubset(set(chain_ids)), "invalid chain id(s)"

    std_chain_ids = set(string.ascii_uppercase + string.ascii_lowercase + string.digits) # 62
    unused_chain_ids = sorted(list(std_chain_ids - set(chain_ids) -set(new_ids))) 
    
    # resolve conflict with intermediate chain id(s)
    resolved = {}
    for k, v in parsed_map.items():
        if v in chain_ids:
            w = unused_chain_ids.pop(0) # take out the first candidate
            st = st.rename_chain(v, w)
            resolved[v] = w

    for k, v in parsed_map.items():
        if k in resolved:
            st = st.rename_chain(resolved[k], v)
        else:
            st = st.rename_chain(k, v)

    st.write(output_filename)
    logger.info(f"Renamed coordinates saved to {output_filename}")
       

def reorder(filename: str, quiet: bool = False):
    """Reorder by chain id"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    output_filename = f"{output_prefix}_ordered.pdb"
    PDBEditor.load(filename).reorder_chains().write(output_filename)
    logger.info(f"Reordered coordinates saved to {output_filename}")


def split(filename: str, quiet: bool = False):
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    st = PDBEditor.load(filename)
    for model_idx, model in enumerate(st.structure, start=1):
        single_model_st = gemmi.Structure()
        
        # Preserve original metadata if desired (e.g., cell, spacegroup)
        try:
            single_model_st.cell = st.cell
        except:
            pass
        try:
            single_model_st.spacegroup_name = st.spacegroup_name
        except:
            pass
        
        # 3. Add a copy of the current model to the new structure
        # (Using .clone() prevents altering or corrupting the source object)
        # single_model_st.models.append(model.clone())
        single_model_st.add_model(model.clone())
        
        # 4. Generate a file name and write out the single PDB
        output_filename = f"{output_prefix}_{model_idx}.pdb"
        single_model_st.write_pdb(output_filename)
        logger.info(f"Model {model_idx} saved to {output_filename}")