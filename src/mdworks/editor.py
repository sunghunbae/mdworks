from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from collections import defaultdict

import re
import gemmi
import string
import logging


logger = logging.getLogger(__name__)



class PDBEditor:
    """Load once, chain edits, write once.
    A thin, chainable wrapper around gemmi.Structure for the structure-editing
    tasks that come up repeatedly when preparing PDB/mmCIF files for MD setup:
    reordering chains, deleting chains, deleting residues, selecting subsets,
    and renumbering. Gemmi does the heavy lifting (parsing, format handling,
    PDB/mmCIF round-tripping); this class just gives you a task-oriented API
    on top of it so pipeline code reads like a sequence of intentions rather
    than gemmi object-model plumbing.

    Example
    -------
    >>> (
    ...     PDBEditor.load("input.cif")
    ...     .remove_chains(["C"])
    ...     .remove_residues("A", [45, 46, 47])
    ...     .reorder_chains(["B", "A"])
    ...     .write("output.pdb")
    ... )
    """

    std_chain_ids = set(string.ascii_uppercase + string.ascii_lowercase + string.digits) # 62

    def __init__(self, 
                 structure: gemmi.Structure | None = None,
                 model_index: int = 0,
                 prefix: str = ''):
        self.structure = structure
        self.model_index = model_index
        self.prefix = prefix
        self.sel = gemmi.Selection()

    # ------------------------------------------------------------------ #
    # Construction / IO
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path, model_index: int = 0) -> "PDBEditor":
        """Read a PDB, mmCIF, or mmJSON file (format inferred from contents)."""
        p = Path(path)
        structure = gemmi.read_structure(str(p))
        structure.setup_entities()
        prefix =  p.name.removesuffix("".join(p.suffixes))
        return cls(structure, model_index= model_index, prefix=prefix)


    def write(self, 
              path: str | Path | None = None, 
              minimal: bool = False, 
              tag: str = 'out', 
              split: bool = False) -> "PDBEditor":
        """Write to disk. Format is inferred from the file extension
        (.pdb / .ent -> PDB, .cif / .mmcif -> mmCIF)."""
        if path is None:
            p = Path(f'{self.prefix}_{tag}.cif')
        else:
            p = Path(path)

        if not split:
            if p.suffix.lower() in (".cif", ".mmcif"):
                doc = self.structure.make_mmcif_document() if not minimal \
                    else self.structure.make_mmcif_headers()
                doc.write_file(str(p))
            else:
                self.structure.write_pdb(str(p))
        else:
            for model_idx, model in enumerate(self.structure, start=1):
                p = Path(f"{self.prefix}_{model_idx}.cif") # tag is ignored
                single_model_st = gemmi.Structure()   
                # Preserve original metadata if desired (e.g., cell, spacegroup)
                try:
                    single_model_st.cell = self.cell
                except:
                    pass
                try:
                    single_model_st.spacegroup_name = self.spacegroup_name
                except:
                    pass
                
                # Add a copy of the current model to the new structure
                # (Using .clone() prevents altering or corrupting the source object)
                single_model_st.add_model(model.clone())
                if p.suffix.lower() in (".cif", ".mmcif"):
                    doc = single_model_st.structure.make_mmcif_document() if not minimal \
                        else single_model_st.structure.make_mmcif_headers()
                    doc.write_file(str(p))
                else:
                    single_model_st.structure.write_pdb(str(p))

        return self

    
        

    def clone(self) -> "PDBEditor":
        """Return an independent deep copy, useful before a destructive edit
        when you still need the original in memory."""
        return PDBEditor(self.structure.clone(), self.model_index)


    def to_mmcif_str(self) -> str:
        cif_doc = self.structure.make_mmcif_document()
        cif_string = cif_doc.as_string()
        return cif_string


    def to_pdb_str(self) -> str:
        pdb_string = self.structure.make_pdb_string()
        return pdb_string


    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> gemmi.Model:
        return self.structure[self.model_index]


    def chain_names(self) -> list[str]:
        return [c.name for c in self.model]


    def _get_chain(self, chain_id: str) -> gemmi.Chain:
        chain = self.model.find_chain(chain_id)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_id}' not found. Available: {self.chain_names()}"
            )
        return chain


    @staticmethod
    def _chain_id_order(char):
        if char.isupper():
            return (0, char)  # Highest priority (0)
        elif char.islower():
            return (1, char)  # Medium priority (1)
        elif char.isdigit():
            return (2, char)  # Lowest priority (2)
        else:
            return (3, char)  # Fallback for symbols/punctuation


    def _get_next_chain_id(self) -> str:
        chain_ids = set(self.chain_names())
        unused_chain_ids = sorted(list(self.std_chain_ids - chain_ids), key=PDBEditor._chain_id_order)
        return unused_chain_ids.pop(0)


    def select(self, expr: str) -> "PDBEditor":
        """Parse Ambertools style chain/residue selection expressions and 
        generate gemmi.Selection filter.

        Coordinate ID (or CID):
            https://gemmi.readthedocs.io/en/stable/analysis.html#selections-cid

        Args:
            expr (str): example - "A:10-30,A:100-120,B:1-50,C:1,D,UNL"
        """
        pattern = r"^(?:(?P<chain>[A-Za-z0-9]+):)?(?P<resname_or_resseq>[A-Za-z0-9]+)(?:-(?P<resseq_end>[0-9]+))?$"

        self.sel = gemmi.Selection()

        for sub_expr in expr.split(","):
            match = re.match(pattern, sub_expr)
            if match:
                spec = match.groupdict()
                chain_id = spec.get('chain', '*')
                res_start = spec.get('resname_or_resseq')
                res_end = spec.get('resseq_end')
                if res_start.isdigit():
                    if res_end.isdigit():
                        cid = f'//{chain_id}/{res_start}-{res_end}'
                    else:
                        cid = f'//{chain_id}/{res_start}'
                else:
                    cid = f'//{chain_id}/({res_start})'
                self.sel = self.sel | gemmi.Selection(cid) # OR (Union)

        return self


    def delete(self, invert: bool = False) -> "PDBEditor":
        if invert:
            self.sel.remove_not_selected(self.model)
        else:
            self.sel.remove_selected(self.model)
        return self

    
    def merge(self, other: PDBEditor) -> "PDBEditor":
        for chain in other.model:
            self.model.add_chain(chain.clone())
        return self


    def extract(self) -> "PDBEditor":
        extracted = self.sel.copy_structure_selection(self.structure)
        return PDBEditor(extracted)
    
    # def extract(self, resname: str) -> "PDBEditor":
    #     target_chain_id = None
    #     target_residue = None
    #     for chain in self.model:
    #         for residue in chain:
    #             if residue.name == resname:
    #                 target_chain_id = chain.name
    #                 target_residue = residue.clone()

    #     assert target_residue, f"residue {resname} not found"
        
    #     chain = gemmi.Chain(target_chain_id)
    #     chain.add_residue(target_residue)

    #     model = gemmi.Model(0)
    #     model.add_chain(chain)

    #     st = gemmi.Structure()
    #     st.add_model(model)

    #     return PDBEditor(st) 

    def new_chains_for_non_std_residues(self) -> "PDBEditor":
        """
        Moves specified non-standard residues out of their original chains 
        and puts them all into a brand new chain with a unique ID.
        """
        for model_idx, model in enumerate(self.structure, start=1):
            new_chains = defaultdict(list)
            for chain in model:
                residues = [f"{res.name:<3} {res.seqid.num}" for res in chain]
                n = len(residues)
                residues_to_delete = []
                for res_idx, residue in enumerate(chain):
                    if residue.is_water():
                        continue
                    # look up the chemical component details in Gemmi's table
                    chem_comp = gemmi.find_tabulated_residue(residue.name)
                    # check if the residue name is missing or explicitly non-standard
                    if n == 1 or chem_comp is None or not chem_comp.is_standard():
                        # Identify if it is a modified polymer or a ligand block
                        w = self._get_next_chain_id()
                        new_chains[w].append(residue.clone())
                        residues_to_delete.append(res_idx)
                        logger.info(f"assign a new chain id {w} to {chain.name} {residue.name:<3} {residue.seqid.num}")
                # remove residue(s)
                for res_idx in sorted(residues_to_delete, reverse=True):
                    del chain[res_idx] # delete residues from the original chain backward
            # add new chain(s)
            for new_chain_id, residues in new_chains.items():
                if residues:
                    new_chain = gemmi.Chain(new_chain_id)
                    for res in residues:
                        new_chain.add_residue(res)
                    model.add_chain(new_chain)

        return self


    
    # ------------------------------------------------------------------ #
    # Chain-level operations
    # ------------------------------------------------------------------ #

    def remove_chains(self, chain_ids: Iterable[str]) -> "PDBEditor":
        """Delete one or more chains by ID.

        Note: gemmi's underlying remove_chain() silently no-ops on an
        unknown chain ID, so we validate up front and raise instead.
        """
        model = self.model
        available = set(self.chain_names())
        chain_ids = list(chain_ids)
        missing = [cid for cid in chain_ids if cid not in available]
        if missing:
            raise KeyError(
                f"Chain(s) {missing} not found. Available: {sorted(available)}"
            )
        for cid in chain_ids:
            model.remove_chain(cid)
        return self

    def keep_chains(self, chain_ids: Iterable[str]) -> "PDBEditor":
        """Keep only the listed chains, dropping everything else."""
        keep = set(chain_ids)
        drop = [c for c in self.chain_names() if c not in keep]
        return self.remove_chains(drop)


    def reorder_chains(self, order: Sequence[str] | None = None) -> "PDBEditor":
        """Reorder chains in the model. `order` must be a permutation of
        the existing chain IDs (use keep_chains/remove_chains first if you
        also want to drop some)."""
        model = self.model
        current = self.chain_names()
        if not order:
            order = sorted(list(current))
        if set(order) != set(current):
            raise ValueError(
                f"reorder_chains requires a full permutation of {sorted(current)}, "
                f"got {sorted(order)}"
            )
        # Clone chains out before clearing, since chain objects reference
        # into the model's storage.
        chains_by_name = {c.name: c.clone() for c in model}
        del model[:]
        for name in order:
            model.add_chain(chains_by_name[name])
        return self


    @staticmethod
    def _parse_chain_id_mapping(spec_string: str) -> dict:
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


    def rename_chain(self, old_id: str, new_id: str) -> "PDBEditor":
            chain = self._get_chain(old_id)
            chain.name = new_id
            return self


    def rename_chains(self, chain_map: str) -> "PDBEditor":
        """Rename chain id(s)"""
        parsed_map = PDBEditor._parse_chain_id_mapping(chain_map)
        old_ids = list(parsed_map.keys())
        chain_ids = self.chain_names()
        assert set(old_ids).issubset(set(chain_ids)), "invalid chain id(s)"
        
        # resolve conflict with intermediate chain id(s)
        resolved = {}
        for k, v in parsed_map.items():
            if v in chain_ids:
                w = self._get_next_chain_id()
                self = self.rename_chain(v, w)
                resolved[v] = w

        for k, v in parsed_map.items():
            if k in resolved:
                self = self.rename_chain(resolved[k], v)
            else:
                self = self.rename_chain(k, v)

        self.reorder_chains()
        return self


    def remove_waters(self) -> "PDBEditor":
        self.model.remove_waters()
        return self


    def remove_ligands_and_waters(self) -> "PDBEditor":
        self.model.remove_ligands_and_waters()
        return self


    def remove_hydrogens(self) -> "PDBEditor":
        self.model.remove_hydrogens()
        return self


    def remove_alternative_conformations(self) -> "PDBEditor":
        """Collapse altlocs down to a single conformer (highest occupancy)."""
        self.model.remove_alternative_conformations()
        return self


    def renumber_residues(self, chain_id: str, start: int = 1) -> "PDBEditor":
        """Renumber a chain's residues sequentially starting at `start`,
        clearing insertion codes."""
        chain = self._get_chain(chain_id)
        for offset, res in enumerate(chain):
            res.seqid = gemmi.SeqId(start + offset, " ") 
            # residue number, icode (empty/null character if no insertion code)
        return self


    def find_zn_coord_cys(self, model_idx: int = 0, cutoff: float = 3.5) -> set:
        """Find Cysteine residues possibly coordinating a zinc atom.

        Args:
            filename (str): _description_
            model_idx (int, optional): _description_. Defaults to 0.
            cutoff (float, optional): _description_. Defaults to 3.5.
                Although ideal Zn-SG(CYS) distance is 2.34 A, here we use 3.5 A as initial
                cutoff to identify Cys residues around a zinc atom.

        Returns:
            set: {(chain_id, resseq), ...}
        """
        model = self.structure[model_idx]
        ns = gemmi.NeighborSearch(model, self.structure.cell, 5.0).populate()
        zn_cys = set()
        for chain in model:
            for res in chain:
                if res.name != 'ZN':
                    continue
                zn_atom = res[0] # there must be only one atom in the ZN residue
                logger.info(f"Found a zinc atom {chain.name} {res.seqid.num}")
                for mark in ns.find_atoms(zn_atom.pos, '\0', radius=cutoff):
                    cra = mark.to_cra(model)
                    if cra.residue.name == 'CYS' and cra.atom.name == 'SG':
                        zn_cys.add((cra.chain.name, cra.residue.seqid.num))
                        logger.info(f"Found a zinc coordinating CYS at {cra.chain.name} {cra.residue.seqid.num}")
        return zn_cys

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def residue_ids(self, chain_id: str) -> list[tuple[str, int]]:
        """[(resname, seqnum), ...] for a chain, handy for sanity-checking
        edits before writing to disk."""
        return [(r.name, r.seqid.num) for r in self._get_chain(chain_id)]


    def summary(self) -> None:
        lines = []
        for model_idx, model in enumerate(self.structure, start=1):
            lines.append(f"model {model_idx}")
            for chain in model:
                residues = [f"{res.name:<3} {res.seqid.num}" for res in chain]
                n = len(residues)
                if n == 1:
                    lines.append(f"  chain {chain.name} ({n:>3} residues): {residues[0]}")
                else:
                    lines.append(f"  chain {chain.name} ({n:>3} residues): {residues[0]:<7} ... {residues[-1]}")
                # non standard residues
                for residue in chain:
                    if residue.is_water():
                        continue
                    # look up the chemical component details in Gemmi's table
                    chem_comp = gemmi.find_tabulated_residue(residue.name)
                    # check if the residue name is missing or explicitly non-standard
                    if chem_comp is None or not chem_comp.is_standard():
                        # Identify if it is a modified polymer or a ligand block
                        res_type = "Ligand/Unknown" if chem_comp is None else chem_comp.kind.name
                        lines.append(f"    non-standard residue {residue.name:<3} {residue.seqid.num} {res_type}")

        print("\n".join(lines))