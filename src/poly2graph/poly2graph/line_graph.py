import numpy as np
import networkx as nx
from functools import partial

from typing import Optional, Union, Callable, Any

def angle_between_vecs(v1, v2, origin=None):
    if origin is not None: v1 = v1 - origin; v2 = v2 - origin
    l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
    if l1 == 0 or l2 == 0: return 0
    else: return np.arccos(np.clip(np.dot(v1, v2)/(l1*l2), -1.0, 1.0))

def LG_undirected(
    G: Union[nx.Graph, nx.MultiGraph],
    selfloops: Optional[bool] = False,
    create_using: Optional[Any] = None,
    triplet_feature: Optional[bool] = False
):
    """Returns the line graph L of the (multi)-graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    triplet_feature : bool
        If `True`, calculate the angles between edges in the line graph.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges.
    get_edges = partial(G.edges, keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    
    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    # Introduce numbering of nodes
    node_index = {n: i for i, n in enumerate(G)}

    # Lift canonical representation of nodes to edges in line graph
    edge_key_function = lambda edge: (node_index[edge[0]], node_index[edge[1]])

    edges = set()
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        # Decide on representation of {u, v} as (u, v) or (v, u) depending on node_index.
        # -> This ensures a canonical representation and avoids comparing values of different types.
        nodes = [tuple(sorted(x[:2], key=node_index.get)) + (x[2],) for x in get_edges(u)]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            edge = nodes[0]
            canonical_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2])
            L.add_node(canonical_edge, **G.get_edge_data(*edge[:3]))

        for i, a in enumerate(nodes):
            canonical_a = (min(a[0], a[1]), max(a[0], a[1]), a[2])
            L.add_node(canonical_a, **G.get_edge_data(*a[:3]))  # Transfer edge attributes to node
            for b in nodes[i + shift:]:
                canonical_b = (min(b[0], b[1]), max(b[0], b[1]), b[2])
                edge = tuple(sorted((canonical_a, canonical_b), key=edge_key_function))
                if edge not in edges:
                    # find the common node u. TODO: modify for self-loops
                    u = set(a[:2]).intersection(set(b[:2])).pop()
                    attr = {}
                    if triplet_feature:
                        attr['joint_node_attr'] = G.nodes[u]
                        v = a[0] if a[0] != u else a[1]
                        w = b[0] if b[0] != u else b[1]
                        pos_u = attr['joint_node_attr']['o']

                        # Calculate the center of the triplet
                        attr['triplet_center'] = np.mean([G.nodes[v]['o'], G.nodes[w]['o'], pos_u], axis=0)
                        
                        # Calculate the angle between edges
                        angle = [angle_between_vecs(G.nodes[v]['o'], G.nodes[w]['o'], origin=pos_u)]
                        if 'pts2' in G.get_edge_data(*a[:3]) and 'pts2' in G.get_edge_data(*b[:3]):
                            for pos in G.get_edge_data(*a[:3])['pts2']:
                                angle.append(angle_between_vecs(G.nodes[w]['o'], pos, origin=pos_u))
                            for pos in G.get_edge_data(*b[:3])['pts2']:
                                angle.append(angle_between_vecs(G.nodes[v]['o'], pos, origin=pos_u))
                        attr['angle'] = np.array(angle, dtype=np.float32)
                    L.add_edge(canonical_a, canonical_b, **attr)
                    edges.add(edge)
                    # print(f"Added edge: {canonical_a} -> {canonical_b} with attributes {attr}") # Debugging
    return L