from typing import Dict, Set

from more_itertools.recipes import pairwise

flatten = lambda l: [item for sublist in l for item in sublist]


class SkeletonType:
    def __init__(self, lines, names=None, one_sided=None, composed=False):
        self.lines = lines
        if composed:
            self.subskel_mapper = {
                key: set(flatten([i for i in val.values()]))
                for key, val in self.lines.items()
            }
        self.names = names
        self.build_graphs()
        self.max_kp = 0
        for line in self.lines_flat:
            for kp in line:
                if kp > self.max_kp:
                    self.max_kp = kp
        self.max_kp += 1
        self.one_sided = one_sided

    @property
    def lines_flat(self):
        from .utils import flatten

        yield from flatten(self.lines)

    def build_graphs(self):
        self.graph: Dict[int, Set[int]] = {}
        self.digraph: Dict[int, Set[int]] = {}
        self.kp_idxs: Set[int] = set()
        for line in self.lines_flat:
            for n1, n2 in pairwise(line):
                if n1 > n2:
                    n1, n2 = n2, n1
                self.graph.setdefault(n1, set()).add(n2)
                self.digraph.setdefault(n1, set()).add(n2)
                self.digraph.setdefault(n2, set()).add(n1)
                self.kp_idxs.add(n1)
                self.kp_idxs.add(n2)

    def _get_subskel(self, idx):
        if hasattr(self, "subskel_mapper"):
            return [key for key, val in self.subskel_mapper.items() if idx in val][0]

    def adj(self, idx):
        return self.graph[idx]

    def adj_ordered(self, idx):
        return self.digraph[idx]

    def iter_limbs(self, kps, kp_idxs=None):
        if kp_idxs is None:
            kp_idxs = range(self.max_kp)
        for idx in kp_idxs:
            for other_idx in self.digraph.get(idx, set()):
                if other_idx not in kp_idxs:
                    continue
                yield kps[idx], kps[other_idx], self._get_subskel(idx)

    def iter_limb_pairs(self, kps, kp_idxs=None):
        if kp_idxs is None:
            kp_idxs = range(self.max_kp)
        for idx in kp_idxs:
            for outwards1 in self.graph.get(idx, set()):
                if outwards1 not in kp_idxs:
                    continue
                for outwards2 in self.graph.get(idx, set()):
                    if outwards2 < outwards1 or outwards2 not in kp_idxs:
                        continue
                    yield kps[idx], kps[outwards1], kps[outwards2]

    def export(self):
        res = {
            "lines": self.lines,
            "names": self.names,
            "graph": self.graph,
            "digraph": self.digraph,
            "max_kp": self.max_kp,
        }
        if self.one_sided is not None:
            res["one_sided"] = self.one_sided
        return res
