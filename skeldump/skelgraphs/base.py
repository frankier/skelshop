from more_itertools.recipes import pairwise


class SkeletonType:
    def __init__(self, lines, names=None):
        self.lines = lines
        self.names = names
        self.build_graphs()
        self.num_kps = 0
        for line in self.lines_flat:
            for kp in line:
                if kp > self.num_kps:
                    self.num_kps = kp

    @property
    def lines_flat(self):
        for part in self.lines.values():
            if isinstance(part, dict):
                yield from part.values()
            else:
                yield part

    def build_graphs(self):
        self.graph = {}
        self.digraph = {}
        for line in self.lines_flat:
            for n1, n2 in pairwise(line):
                if n1 > n2:
                    n1, n2 = n2, n1
                self.graph.setdefault(n1, set()).add(n2)
                self.digraph.setdefault(n1, set()).add(n2)
                self.digraph.setdefault(n2, set()).add(n1)

    def adj(self, idx):
        return self.graph[idx]

    def adj_ordered(self, idx):
        return self.digraph[idx]

    def iter_limbs(self, kps, kp_idxs=None):
        if kp_idxs is None:
            kp_idxs = range(self.num_kps)
        for idx in kp_idxs:
            for other_idx in self.digraph.get(idx, set()):
                if other_idx not in kp_idxs:
                    continue
                yield kps[idx], kps[other_idx]

    def iter_limb_pairs(self, kps, kp_idxs=None):
        if kp_idxs is None:
            kp_idxs = range(self.num_kps)
        for idx in kp_idxs:
            for outwards1 in self.graph.get(idx, set()):
                if outwards1 not in kp_idxs:
                    continue
                for outwards2 in self.graph.get(idx, set()):
                    if outwards2 < outwards1 or outwards2 not in kp_idxs:
                        continue
                    yield kps[idx], kps[outwards1], kps[outwards2]
