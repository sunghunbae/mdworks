from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import gemmi


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

    def __init__(self, 
                 structure: gemmi.Structure | None = None, 
                 model_index: int = 0):
        self.structure = structure
        self.model_index = model_index

    # ------------------------------------------------------------------ #
    # Construction / IO
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path, model_index: int = 0) -> "PDBEditor":
        """Read a PDB, mmCIF, or mmJSON file (format inferred from contents)."""
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()
        return cls(structure, model_index=model_index)

    def write(self, path: str | Path, minimal: bool = False) -> "PDBEditor":
        """Write to disk. Format is inferred from the file extension
        (.pdb / .ent -> PDB, .cif / .mmcif -> mmCIF)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            doc = self.structure.make_mmcif_document() if not minimal \
                else self.structure.make_mmcif_headers()
            doc.write_file(str(path))
        else:
            self.structure.write_pdb(str(path))
        return self

    def clone(self) -> "PDBEditor":
        """Return an independent deep copy, useful before a destructive edit
        when you still need the original in memory."""
        return PDBEditor(self.structure.clone(), self.model_index)

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

    def rename_chain(self, old_id: str, new_id: str) -> "PDBEditor":
        chain = self._get_chain(old_id)
        chain.name = new_id
        return self

    # ------------------------------------------------------------------ #
    # Residue-level operations
    # ------------------------------------------------------------------ #

    def remove_residues(
        self, chain_id: str, seq_nums: Iterable[int]
    ) -> "PDBEditor":
        """Delete residues from a chain by author sequence number
        (the number you see in column 23-26 of an ATOM record)."""
        chain = self._get_chain(chain_id)
        targets = set(seq_nums)
        idx_to_remove = [
            i for i, res in enumerate(chain) if res.seqid.num in targets
        ]
        for i in reversed(idx_to_remove):  # reverse so indices stay valid
            del chain[i]
        return self

    def keep_residue_range(
        self, chain_id: str, start: int, end: int
    ) -> "PDBEditor":
        """Keep only residues with seqid.num in [start, end] on one chain."""
        chain = self._get_chain(chain_id)
        idx_to_remove = [
            i for i, res in enumerate(chain)
            if not (start <= res.seqid.num <= end)
        ]
        for i in reversed(idx_to_remove):
            del chain[i]
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
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def residue_ids(self, chain_id: str) -> list[tuple[str, int]]:
        """[(resname, seqnum), ...] for a chain, handy for sanity-checking
        edits before writing to disk."""
        return [(r.name, r.seqid.num) for r in self._get_chain(chain_id)]

    def summary(self) -> str:
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

        return "\n".join(lines)

