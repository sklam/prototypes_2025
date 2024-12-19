"""
Custom cost based extraction
"""

from collections import defaultdict
from dataclasses import dataclass
import networkx as nx


@dataclass
class Node:
    children: list[str]
    cost: float
    eclass: str
    op: str
    subsumed: bool


def _convert_cyclic_to_dag(graph, root):
    # Create a copy of the original graph
    dag: nx.DiGraph = graph.copy()

    backedges = set()

    worklist = [(root, ())]
    # DFS
    while worklist:
        current, parents = worklist.pop()
        new_parents = (*parents, current)
        for neighbor in dag.neighbors(current):
            if neighbor in parents:
                backedges.add((current, neighbor))
            else:
                worklist.append((neighbor, new_parents))

    dag.remove_edges_from(backedges)
    return dag


class Extraction:
    nodes: dict[str, Node]
    nxg: nx.DiGraph
    root_eclass: str

    def __init__(self, graph_json, root_eclass, cost_model):
        self.root_eclass = root_eclass
        self.class_data = defaultdict(set)
        self.nodes = {k: Node(**v) for k, v in graph_json["nodes"].items()}
        for k, node in self.nodes.items():
            self.class_data[node.eclass].add(k)
        self.cost_model = cost_model
        self.nxg = self._make_nx()
        self.dag = _convert_cyclic_to_dag(self.nxg, self.root_eclass)

    @staticmethod
    def draw_graph(G, filename):
        A = nx.nx_pydot.to_pydot(G)
        svg = A.create_svg()
        with open(f"{filename}", "wb") as fout:
            fout.write(svg)

    def choose(self):
        self._compute_cost()

        nodes = self.nodes
        cost, chosen_root = min(
            (nodes[k].cost, k)
            for k in nodes
            if nodes[k].eclass == self.root_eclass
        )
        # make selected graph
        G = nx.MultiDiGraph()
        todolist = [chosen_root]
        visited = set()
        while todolist:
            cur = todolist.pop()
            if cur in visited:
                continue
            visited.add(cur)

            for i, u in enumerate(nodes[cur].children):
                child = nodes[u]
                candidates = self.class_data[child.eclass]

                if len(candidates) > 1:
                    v = min(candidates, key=lambda v: nodes[v].cost)
                    G.add_edge(cur, v, label=str(i))
                    todolist.append(v)
                else:
                    G.add_edge(cur, u, label=str(i))
                    todolist.append(u)

        return cost, G

    def _compute_cost(self):
        dag = self.dag
        rev_topo = list(nx.dfs_postorder_nodes(dag))
        nodes = self.nodes
        costmap = {}
        for k in rev_topo:
            if k in self.class_data:
                costmap[k] = min(costmap[x] for x in self.class_data[k])
            else:
                if k.startswith("primitive-"):
                    cost = 0
                elif k.startswith("function-"):
                    # costmap lookup here can see cycles;
                    # in that case use the actual node cost.
                    # if that is not computed, use 1.
                    child_costs = [
                        costmap.get(nodes[x].eclass, costmap.get(x, 1))
                        for x in nodes[k].children
                    ]
                    cost = self.cost_model.get_cost_function(
                        k, nodes[k].op, nodes, child_costs
                    )
                else:
                    raise ValueError(f"unknown node {k}")

                costmap[k] = cost
                nodes[k].cost = costmap[k]

    def _make_nx(self) -> nx.DiGraph:
        """
        Make NX graph that start the node children always points to the eclass,
        which then contains the node (enode)
        """
        nodes = self.nodes

        G = nx.DiGraph()
        for k, node in nodes.items():
            G.add_node(node.eclass)
            G.add_node(k)
            G.add_edge(node.eclass, k)

        for k, node in nodes.items():
            for v in node.children:
                child = nodes[v]
                G.add_edge(k, child.eclass)
        return G
