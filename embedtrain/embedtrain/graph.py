from mmskeleton.ops.st_gcn.graph import Graph, get_hop_distance


class GraphAdapter(Graph):
    def __init__(self, skelshop_graph, strategy="uniform", max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.setup_edge(skelshop_graph)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def setup_edge(self, skelshop_graph):
        # Can use max_kp as num_kps here because we know we have been passed a
        # skel graph Perhaps this info should be indicated in an asserted
        # value/type
        self.num_node = skelshop_graph.max_kp
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(k, v) for k, vs in skelshop_graph.graph.items() for v in vs]
        self.edge = self_link + neighbor_link
        self.center = 0
