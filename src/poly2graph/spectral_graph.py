import numpy as np
import networkx as nx
from skimage.filters import laplace, gaussian
from sklearn.metrics import pairwise_distances

from numpy.typing import ArrayLike
from typing import Union, Optional, Callable, Iterable, TypeVar, List
NetworkXGraph = TypeVar('nxGraph', nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)



def minmax_normalize(
    image: ArrayLike,
    copy: bool = True
) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Parameters
    ----------
    image : ArrayLike
        Input image to be normalized.

    Returns
    -------
    np.ndarray
        Normalized image with values scaled to the range [0, 1].
    """
    # Ensure a floating point copy to avoid integer division artifacts.
    image = np.array(image, dtype=float, copy=copy)
    image -= np.min(image)
    img_max = np.max(image)
    if img_max > 0:
        image /= img_max
    return image


def PosGoL(
    image: ArrayLike,
    sigmas: Optional[Iterable[int]] = [0, 1],
    ksizes: Optional[Iterable[int]] = [5],
    black_ridges: Optional[bool] = False,
    power_scaling: Optional[float] = None,
    min_max_normalize: Optional[bool] = True,
    copy: Optional[bool] = True
) -> np.ndarray:
    """
    Apply the Positive Laplacian of Gaussian (PosGoL) filter to enhance spectral graph features.

    The filter applies a Gaussian blur followed by the Laplace operator to emphasize 
    positive curvature regions, making it useful for detecting features in a spectral 
    potential landscape.

    Parameters
    ----------
    image : ArrayLike
        Input image to be filtered.
    sigmas : Iterable[int], optional
        Scales (sigma values) for the Gaussian filter. Default is [0, 1].
    ksizes : Iterable[int], optional
        Kernel sizes for the discrete Laplacian operator. Default is [5].
    black_ridges : bool, optional
        If True, invert the image so that black ridges are detected instead of white ridges.
        Default is False.
    power_scaling : float, optional
        Exponent to which the normalized image is raised before filtering. This can sharpen 
        the spectral graph features if set to 1/n (with n = 3, 5, etc.).
    min_max_normalize : bool, optional
        If True, normalize the image to [0, 1] before processing. Default is True.
    copy : bool, optional
        If True, make a copy of the input image to avoid modifying it in-place. Default is True.

    Returns
    -------
    np.ndarray
        Filtered image with the pixel-wise maximum response across all scales and kernel sizes.
    """
    if copy:
        image = np.array(image, copy=True)

    if black_ridges:
        image = -image

    if min_max_normalize:
        image = minmax_normalize(image)

    if power_scaling is not None:
        image = image**power_scaling

    filtered_max = np.zeros_like(image)
    for ksize in ksizes:
        for sigma in sigmas:
            # Apply Laplacian filter with the specified kernel size.
            lap = laplace(image, ksize=ksize)
            # Apply Gaussian smoothing if sigma > 0; otherwise, use the Laplacian response.
            gauss = gaussian(lap, sigma=sigma) if sigma > 0 else lap
            # Retain only the positive curvature response.
            pos = np.maximum(gauss, 0)
            # Normalize the positive response to have a maximum of 1, if nonzero.
            max_val = pos.max()
            if max_val > 0:
                pos /= max_val
            # Compute the pixel-wise maximum response.
            filtered_max = np.maximum(filtered_max, pos)
            
    return filtered_max

def remove_isolates(
    G: NetworkXGraph, 
    copy: bool = False
) -> NetworkXGraph:
    """
    Remove isolated nodes (nodes with zero degree) from a NetworkX graph.

    Parameters
    ----------
    G : nxGraph
        Input graph from which isolated nodes will be removed.
    copy : bool, optional
        If True, the function works on a copy of the graph. Default is False (in-place).

    Returns
    -------
    nxGraph
        Graph with isolated nodes removed.
    """
    graph = G.copy() if copy else G
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    return graph


def add_edges_within_threshold(
    G: NetworkXGraph, 
    threshold: float = 5.0, 
    copy: bool = False
) -> NetworkXGraph:
    """
    Add edges between nodes if their positions are within a specified distance threshold.

    This function computes the pairwise distances of node positions (stored in the 'pos'
    attribute) and adds an edge (with a weight equal to the distance) between any two nodes
    that are closer than the threshold value.

    Parameters
    ----------
    G : nxGraph
        Input graph whose nodes are expected to have a 'pos' attribute.
    threshold : float, optional
        Distance threshold under which an edge is added. Default is 5.0.
    copy : bool, optional
        If True, the function works on a copy of the graph. Default is False (in-place).

    Returns
    -------
    nxGraph
        Graph with edges added connecting nodes that are within the distance threshold.
    """
    graph = G.copy() if copy else G
    pos = nx.get_node_attributes(graph, 'pos')
    nodes = list(pos.keys())
    
    # Convert node positions to a numpy array for efficient computation.
    pos_array = np.array([pos[n] for n in nodes])
    
    # Compute pairwise distances among nodes.
    dist_matrix = pairwise_distances(pos_array)
    
    # Extract only upper triangular indices (excluding diagonal) to avoid duplicate pairs.
    i_upper, j_upper = np.triu_indices(len(nodes), k=1)
    distances = dist_matrix[i_upper, j_upper]
    mask = distances < threshold

    for i, j, d in zip(i_upper[mask], j_upper[mask], distances[mask]):
        if not graph.has_edge(nodes[i], nodes[j]):
            graph.add_edge(nodes[i], nodes[j], weight=d)
    return graph


def _average_attributes(node_attrs: dict):
    """
    Recursively average node attributes from the original node and any contracted nodes.

    Attributes considered include:
      - 'pos': Position of the node (sequence of numbers).
      - 'dos': Density of states value (float), optional.
      - 'potential': Potential value (float), optional.
      - 'contraction': A dictionary of attributes from contracted nodes, optional.

    Returns
    -------
    tuple
        A tuple containing:
          - avg_pos (np.ndarray): Averaged position.
          - avg_dos (float): Averaged density of states.
          - avg_potential (float): Averaged potential.
          - count (int): Number of nodes included in the average.
    """
    sum_pos = np.zeros(2)
    sum_dos, sum_potential, count = 0.0, 0.0, 0

    if 'pos' in node_attrs:
        sum_pos = np.array(node_attrs['pos'], dtype=float)
        sum_dos = node_attrs.get('dos', 0.0)
        sum_potential = node_attrs.get('potential', 0.0)
        count = 1

    for contracted_node in node_attrs.get('contraction', {}).values():
        avg_pos, avg_dos, avg_potential, n = _average_attributes(contracted_node)
        sum_pos += avg_pos
        sum_dos += avg_dos
        sum_potential += avg_potential
        count += n

    if count > 0:
        avg_pos = sum_pos / count
        avg_dos = sum_dos / count
        avg_potential = sum_potential / count
    else:
        avg_pos, avg_dos, avg_potential = sum_pos, sum_dos, sum_potential

    return avg_pos, avg_dos, avg_potential, count


def process_contracted_graph(
    G: NetworkXGraph, 
    copy: bool = False
) -> NetworkXGraph:
    """
    Process a contracted graph by averaging node attributes from contracted nodes.

    The attributes 'pos', 'dos', and 'potential' are updated by averaging over the original
    node and any nodes contracted into it. The temporary 'contraction' field is removed after processing.

    Parameters
    ----------
    G : nxGraph
        Input graph with nodes containing contraction information.
    copy : bool, optional
        If True, operate on a copy of the graph. Default is False (in-place).

    Returns
    -------
    nxGraph
        Processed graph with updated attributes and without the 'contraction' field.
    """
    graph = G.copy() if copy else G
    for node, attr in list(graph.nodes(data=True)):
        avg_pos, avg_dos, avg_potential, _ = _average_attributes(attr)
        graph.nodes[node]['pos'] = avg_pos
        if avg_dos != 0.0:
            graph.nodes[node]['dos'] = avg_dos
        if avg_potential != 0.0:
            graph.nodes[node]['potential'] = avg_potential
        if 'contraction' in attr:
            del graph.nodes[node]['contraction']
    return graph


def contract_close_nodes(
    G: NetworkXGraph,
    threshold: Union[int, float],
) -> NetworkXGraph:
    """
    Contract nodes in a spectral graph that are connected by edges shorter than a threshold.

    This function first removes isolated nodes and then repeatedly contracts
    nodes connected by an edge with a weight below the specified threshold. Contracted
    node attributes are updated using an averaging procedure, and the process repeats
    until all edges have a weight exceeding the threshold.

    Parameters
    ----------
    G : nxGraph
        Input spectral graph.
    threshold : int or float
        Distance threshold below which nodes are contracted.

    Returns
    -------
    nxGraph
        Graph with nodes contracted and isolated nodes removed.
    """
    
    current_graph = remove_isolates(G, copy=True) # initial cleaning

    contracted_graph = current_graph.copy()

    while True:
        # Get all edges with their weights.
        edges = list(contracted_graph.edges(data='weight'))
        # Filter out the edges with weights below the threshold.
        lower_edges = [edge for edge in edges if edge[2] < threshold]
        if not lower_edges:
            break

        # Contract the edge with the minimum weight among those below the threshold.
        u, v, _ = min(lower_edges, key=lambda x: x[2])
        try:
            # Contract the nodes (using a copy for safety), then process the contracted graph.
            temp_graph = process_contracted_graph(
                nx.contracted_nodes(contracted_graph, u, v, self_loops=False, copy=True),
                # copy=True
            )
            # If contraction results in a graph with no edges, return the original cleaned graph.
            if temp_graph.number_of_edges() == 0:
                return current_graph
            contracted_graph = temp_graph
        except Exception:
            # If contraction fails, exit the loop.
            break

    return remove_isolates(contracted_graph, copy=False) # final cleaning


def spectral_potential_batch(
    z_arr: ArrayLike, 
    coeff_arr: ArrayLike, 
    q: int, 
    method: str = 'ronkin'
) -> np.ndarray:
    """
    Compute the spectral potential landscape from the roots of a characteristic polynomial.

    The spectral potential is calculated using one of three methods:
    'ronkin', 'diff_log', or 'log_diff'. In the Ronkin method, the potential is computed
    from the largest q magnitudes of the roots. The other methods use differences of logarithms
    of the roots or their derivatives.

    Parameters
    ----------
    z_arr : ArrayLike
        Array of complex numbers representing the roots of a polynomial.
    coeff_arr : ArrayLike
        Coefficients of the polynomial with the highest degree coefficient first.
    q : int
        Number of positive monomials (or the degree) used in the computation.
    method : str, optional
        Method for computing the spectral potential. Options are:
        'ronkin' (default), 'diff_log', or 'log_diff'.

    Returns
    -------
    np.ndarray
        The computed spectral potential landscape.

    Raises
    ------
    ValueError
        If an invalid method is specified.
    """
    z_arr = np.asarray(z_arr)
    coeff_arr = np.asarray(coeff_arr)

    if method.lower() == 'ronkin':
        # Ronkin function: 
        # using the largest q magnitudes of the roots + leading coefficient
        large_qs = np.sort(np.abs(z_arr), axis=-1)[..., -q:]
        phi = - np.log(np.abs(coeff_arr[..., 0])) - np.sum(np.log(large_qs), axis=-1)
    elif method.lower() == 'diff_log':
        # Difference of logarithms: 
        # the difference of the largest two inverse skin depths (kappa)
        kappas = -np.log(np.sort(np.abs(z_arr), axis=-1))
        kappa_diffs = kappas[..., :-1] - kappas[..., 1:]
        phi = - kappa_diffs[..., -q]
    elif method.lower() == 'log_diff':
        # Log difference of the two smallest |z| values
        zs = np.sort(np.abs(z_arr), axis=-1)
        min_z_diffs = zs[..., -q] - zs[..., -q-1]
        phi = - np.log(min_z_diffs / np.max(min_z_diffs))
        # Replace NaN values (which may occur due to division by zero) with the maximum finite value.
        phi[np.isnan(phi)] = np.nanmax(phi)
    else:
        raise ValueError("Invalid method specified. Choose 'ronkin', 'diff_log', or 'log_diff'.")

    return phi