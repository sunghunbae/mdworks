from pathlib import Path
from collections import defaultdict
from io import StringIO

from pdbfixer import PDBFixer  
from openmm.app import PDBFile, PDBxFile, Topology, Modeller
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


def extract_ligand(filename: str, ligand_resname: str, output_ligand_pdb: str) -> None:
    """
    Extract ligand from PDB file and save to a separate PDB file.
    """
    fixer = PDBFixer(filename= filename)
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
    with open(output_ligand_pdb, "w") as f:
        PDBFile.writeFile(ligand_topology, ligand_positions, f, keepIds=True)



def _chain_id_order(char):
    if char.isupper():
        return (0, char)  # Highest priority (0)
    elif char.islower():
        return (1, char)  # Medium priority (1)
    elif char.isdigit():
        return (2, char)  # Lowest priority (2)
    else:
        return (3, char)  # Fallback for symbols/punctuation

def merge_receptor_and_ligand(receptor_pdb: str, ligand_pdb: str, complex_cif: str) -> None:
    """Merges the fixed protein and original ligand structures using OpenMM's Modeller"""
    # 1. Load the fixed and protonated protein 
    receptor = PDBFile(receptor_pdb)
    # Extract all unique chain IDs in a single line
    chain_ids = sorted(list(set(c.id for c in receptor.topology.chains())))
    std_chain_ids = set(string.ascii_uppercase + string.ascii_lowercase + string.digits) # 62
    unused_chain_ids = sorted(list(std_chain_ids - set(chain_ids)), key=_chain_id_order)

    # 2. Load the isolated ligand file
    ligand = PDBFile(ligand_pdb)
    w = unused_chain_ids.pop(0) # take out the first candidate
    for chain in ligand.topology.chains():
        chain.id = w

    # 3. Initialize the Modeller object with the protein structure
    modeller = Modeller(receptor.topology, receptor.positions)
    
    # 4. Programmatically combine the ligand's topology and spatial coordinates
    modeller.add(ligand.topology, ligand.positions)
    
    # 5. Write the combined system out to a standard, compliant PDB file
    with open(complex_cif, "w") as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)


def remove_zn_bonds(fixer: PDBFixer) -> None:
    # # Track down the Zinc atom index
    zinc_atom_indices = [atom.index for atom in fixer.topology.atoms() if "ZN" in atom.name.upper()]
    # Rebuild the bond network, explicitly excluding any bonds involving the Zinc atom
    clean_bonds = []
    for bond in fixer.topology.bonds():
        i, j = bond[0], bond[1]
        if i.index in zinc_atom_indices or j.index in zinc_atom_indices:
            logger.info(f"  Break Zn bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
            continue
        clean_bonds.append(bond)
    fixer.topology._bonds = clean_bonds


def strip_spurious_disulfides(fixer: PDBFixer, zn_cys: set) -> None:
    """Strip any spurious disulfide bond (SG-SG) created between two zinc-coordinating cysteines

    Note:
        Real disulfid bonds between two non-zinc coordinating cysteines are left untouched.
        Keeping Zn-CYS bond is recommended because it stabilizes the structure during MD simulation.

    Args:
        fixer (PDBFixer): PDBFixer class instance
        zinc_cys (set): output from `find_zn_coord_cys()`, {(chain_id, resseq), ...}
    """
    kept_bonds = []
    for bond in fixer.topology.bonds():
        i, j = bond[0], bond[1]
        is_i_zn = i.name == 'ZN'
        is_i_zn_cys = i.name == 'SG' and i.residue.name == 'CYS' and (i.residue.chain.id, int(i.residue.id)) in zn_cys
        is_j_zn = j.name == 'ZN'
        is_j_zn_cys = j.name == 'SG' and j.residue.name == 'CYS' and (j.residue.chain.id, int(j.residue.id)) in zn_cys
        
        # handle CYS-CYS
        if is_i_zn_cys and is_j_zn_cys:
            logger.info(f"  Break S-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")
            continue

        # inform Zn-CYS
        if (is_i_zn and is_j_zn_cys) or (is_i_zn_cys and is_j_zn):
            logger.info(f"  Zn-S bond btw {i.residue.chain.id} {i.residue.id} and {j.residue.chain.id} {j.residue.id}")

        kept_bonds.append(bond)
    fixer.topology._bonds = kept_bonds
    # rename zinc-coordinating cysteines: CYS to CYM
    # for res in zn_cys_residues:
    #     res.name = 'CYM'


def do_not_add_missing_atoms_at_terminals(fixer: PDBFixer) -> None:
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



def test(filename: str):
    if filename:
        p = Path(filename)
        output_prefix = p.name.removesuffix("".join(p.suffixes))
        workdir = p.parent
    
    test_output = f"{output_prefix}_test.pdb"
    st = PDBEditor.load(filename).new_chains_for_non_std_residues()
    
    # 1. Assuming 'st' is your existing gemmi.Structure object
    # st = gemmi.read_structure("input.cif") 

    # 2. Convert gemmi.Structure to PDB block string (in-memory)
    pdb_string = st.structure.make_mmcif_document().as_string() 
    # Alternatively use st.write_minimal_pdb("") if PDB block is preferred

    # 3. Create a file-like stream to pass to PDBFixer
    pdb_stream = StringIO(pdb_string)

    # 4. Initialize PDBFixer using the in-memory stream
    fixer = PDBFixer(pdbxfile=pdb_stream)

    with open("dump.pdb", "w") as f:  
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    # 5. You can now use PDBFixer natively
    # fixer.findNonstandardResidues()
    # fixer.replaceNonstandardResidues()
    # fixer.findMissingResidues()
    # ... etc.




def complex(
        filename: str | None = None, 
        pdb_id: str | None = None,
        ligand_resname: str | None = None,
        waters: bool = False,
        separate_hetgens: bool = True,
        zinc: bool = True,
        terminals: bool = False,
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

    fixed_pdb_file = f"{output_prefix}_fixed.pdb"  
    protonated_receptor_pqr = f"{output_prefix}_H.pqr"  
    protonated_receptor_pdb = f"{output_prefix}_H.pdb"
    cmplx_cif_file = f"{output_prefix}_complex.cif"

    st = PDBEditor.load(filename)

    if not waters:
        st = st.remove_waters()

    if separate_hetgens:
        st = st.new_chains_for_non_std_residues()

    if zinc:
        zn_cys = st.find_zn_coord_cys(model_idx=0)
    else:
        zn_cys = set()

    # 2. Convert gemmi.Structure to PDB block string (in-memory)
    cif_string = st.structure.make_mmcif_document().as_string() 
    # Alternatively use st.write_minimal_pdb("") if PDB block is preferred
    
    # 3. Create a file-like stream to pass to PDBFixer
    pdb_stream = StringIO(cif_string)
    
    # 4. Initialize PDBFixer using the in-memory stream
    fixer = PDBFixer(pdbxfile= pdb_stream)

    # if filename:
    #     logger.info(f"PDBFixer reading a file: {filename}")
    #     fixer = PDBFixer(filename= filename)
    # elif pdb_id:
    #     logger.info(f"PDBFixer downloading the PDB {pdb_id} structure from RCSB Protein Data Bank")
    #     fixer = PDBFixer(pdbid= pdb_id)

    if ligand_resname:
        ligand_pdb = f"{output_prefix}_{ligand_resname}.pdb"
        ligand_smi = f"{output_prefix}_{ligand_resname}.smi"
        logger.info(f"[Step 0] Extracting Ligand {ligand_resname} ...")
        extract_ligand(filename or f"{pdb_id}.pdb", ligand_resname, ligand_pdb)
        logger.info(f"Extracted ligand {ligand_resname} saved to {ligand_pdb}")
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
                    logger.info(f"Extracted ligand {ligand_resname} saved to {ligand_smi}")
            else:
                raise ValueError(f"Could not guess ligand SMILES.")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error occurred while running obabel: {e}")
    

    logger.info(f"[Step 1] Fetching and Fixing via PDBFixer ...")

    # PDBFixer uses geometry template to fill in missing residues and atoms, 
    # and to replace nonstandard residues with standard ones.
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
      
    # By default, when you run fixer.findMissingAtoms(), PDBFixer automatically 
    # calculates proximity between all cysteine sulfur atoms. 
    # If a cysteine has multiple partners within a standard distance cutoff, 
    # it logs the warning and picks one, which can lead to incorrect topology generation.
    
    # The missingResidues dictionary stores missing residues as: 
    # (chainIndex, residueIndex): [list of residue names]
    # example:
    # {
    #   (0, 0): ['MET', 'GLU', ..., 'PRO', 'SER']), 
    #   (0, 108): ['PRO', 'VAL', ... , 'VAL'], 
    #   (0, 232): ['VAL', ..., 'ARG', 'LEU']
    # }

    if not terminals:
        do_not_add_missing_atoms_at_terminals(fixer)

    if zn_cys:
        logger.info(f"Stripping spurious S-S bonds around a zinc")
        remove_zn_bonds(fixer)
        strip_spurious_disulfides(fixer, zn_cys)
        
    # Add missing heavy atoms (but do not add hydrogens yet; PDB2PQR will do that)
    fixer.addMissingAtoms()
    fixer.removeChains(chainIndices=[-1])

    # Write intermediate fixed heavy-atom structure  
    with open(fixed_pdb_file, "w") as f:  
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)  
        logger.info(f"PDBFixer fixed heavy atoms and saved to {fixed_pdb_file}")


    logger.info(f"[Step 2] PDB2PQR predicting pKa at pH {target_pH} and protonating ...")
    # PDB2PQR automatically removes ligand
    pdb2pqr_args = [
        "--ff=AMBER", 
        f"--with-ph={target_pH}",
        f"--pdb-output={protonated_receptor_pdb}",
        fixed_pdb_file,
        protonated_receptor_pqr]
    parser = build_main_parser()
    parsed_pdb2pqr_args = parser.parse_args(pdb2pqr_args)
    main_driver(parsed_pdb2pqr_args)

    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    logger.info(f"PDB2PQR saved protonated receptor to {protonated_receptor_pdb}")

    if ligand_resname:
        logger.info(f"[Step 3] Merging protonated receptor and original ligand {ligand_resname} ...")
        merge_receptor_and_ligand(protonated_receptor_pdb, ligand_pdb, cmplx_cif_file)
        logger.info(f"Merged complex saved to {cmplx_cif_file}")


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


def parse_chain_residue_selection(spec_string: str) -> list[tuple]:
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


def parse_chain_id_mapping(spec_string: str) -> dict:
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


def delete(filename: str, selection: str, tag: str, quiet: bool = False) -> None:
    """Delete chain(s) and residue(s)

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
    targets = parse_chain_residue_selection(selection)
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_{tag}.pdb"  

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



def peek(filename: str, quiet: bool = False):
    """Peek and show summary of model(s) and chain(s)"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    print(PDBEditor.load(filename).summary())

def chain_id_order(char):
        if char.isupper():
            return (0, char)  # Highest priority (0)
        elif char.islower():
            return (1, char)  # Medium priority (1)
        elif char.isdigit():
            return (2, char)  # Lowest priority (2)
        else:
            return (3, char)  # Fallback for symbols/punctuation
        
def rename(filename: str, chain_map: str, tag: str, quiet: bool = False):
    """Rename chain id(s)"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    output_filename = f"{output_prefix}_{tag}.pdb"

    st = PDBEditor.load(filename)
    parsed_map = parse_chain_id_mapping(chain_map)
    
    # Resolve potential conflicts
    old_ids = list(parsed_map.keys())
    new_ids = list(parsed_map.values())
    chain_ids = st.chain_names()
    assert set(old_ids).issubset(set(chain_ids)), "invalid chain id(s)"

    std_chain_ids = set(string.ascii_uppercase + string.ascii_lowercase + string.digits) # 62
    unused_chain_ids = sorted(list(std_chain_ids - set(chain_ids) -set(new_ids)), key=chain_id_order) 
    
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

    # reorder
    st.reorder_chains().write(output_filename)
    logger.info(f"Renamed coordinates saved to {output_filename}")
       

def reorder(filename: str, tag: str, quiet: bool = False):
    """Reorder chains by chain id"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)
    output_filename = f"{output_prefix}_{tag}.pdb"

    PDBEditor.load(filename).reorder_chains().write(output_filename)

    logger.info(f"Reordered coordinates saved to {output_filename}")


def split(filename: str, quiet: bool = False):
    """Split and write individual models"""
    p = Path(filename)
    workdir = p.parent
    output_prefix = p.name.removesuffix("".join(p.suffixes))
    logging.getLogger().handlers.clear()
    setup_logger(logger, workdir, output_prefix, quiet=quiet)

    st = PDBEditor.load(filename)

    for model_idx, model in enumerate(st.structure, start=1):
        output_filename = f"{output_prefix}_{model_idx}.pdb"
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
        
        # Add a copy of the current model to the new structure
        # (Using .clone() prevents altering or corrupting the source object)
        single_model_st.add_model(model.clone())
        
        single_model_st.write_pdb(output_filename)
        logger.info(f"Model {model_idx} saved to {output_filename}")
