import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import njit
from .skeleton2graph import skeleton2graph
from .parallel_roots import poly_roots_tf_batch
from skimage.morphology import skeletonize, dilation, disk
from skimage.filters import laplace, gaussian
from skimage.util import view_as_blocks

from numpy.typing import ArrayLike
from typing import Union, Sequence, Optional, Callable, Any, Iterable, TypeVar
# Type for all networkx graph types
nxGraph = TypeVar('nxGraph', nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)


@njit #TODO support multi-band polynomials
def auto_Emaxes(
    c: Iterable,
    N: Optional[int] = 40,
    pbc: Optional[bool] = False,
    pad_factor: Optional[float] = 0.1
):
    """
    Automatically determine the E_max range for spectral_potential / spectral_graph.

    Only applies to one-band polynomials.
    
    Calculate the energy spectrum of a real-space Hamiltonian corresponding
    to the given characteristic polynomial coefficients. Return a square E
    region that covers the entire spectrum.

    Parameters
    ----------
    c : Iterable
        Coefficients of the 1-band polynomial. Must be homogeneous, otherwise
        numba will raise an error.
    N : int, optional
        Number of unit cells to construct the real-space Hamiltonian. 
        Default is 40.
    pbc : bool, optional
        If True, implement periodic boundary conditions. Default is False.
    pad_factor : float, optional
        Factor to pad the E_max range. The TDL spectrum is usually slightly
        larger than the spectrum of a finite system. Default is 0.1.

    Returns
    -------
    E_re_min, E_re_max, E_im_min, E_im_max : float

    """
    # Ensure one-band polynomial
    if len(c.shape) != 1:
        raise ValueError("Only one-band polynomials are supported.")
    # Ensure the coefficients list is symmetric
    if len(c) % 2 == 0:
        raise ValueError("The length of coefficients 'c' must be odd."
                        " The middle coefficient is the z^0 term's.")
    
    mid_idx = len(c) // 2  # Middle index for the z^0 term
    
    # Create the Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.complex128)
    
    # Add the hopping terms based on the coefficients
    for i, coeff in enumerate(c):
        if coeff != 0:
            offset = i - mid_idx  # Determine the diagonal offset
            H += np.eye(N, k=offset) * coeff
            # Implement periodic boundary conditions if pbc is True
            if pbc:
                if offset > 0:
                    H += np.eye(N, k=offset-N) * coeff
                elif offset < 0:
                    H += np.eye(N, k=N+offset) * coeff

    # Compute eigenvalues of the Hamiltonian
    E = np.linalg.eigvals(H)

    re_min, re_max = np.min(E.real), np.max(E.real)
    im_min, im_max = np.min(E.imag), np.max(E.imag)
    len_re, len_im = re_max - re_min, im_max - im_min

    # # Use this for rectangular E_max range
    # pad_re, pad_im = pad_factor * len_re, pad_factor * len_im
    # return re_min - pad_re, re_max + pad_re, im_min - pad_im, im_max + pad_im

    re_center, im_center = (re_max + re_min)/2, (im_max + im_min)/2
    radius = max(len_re, len_im)*(1+2*pad_factor)/2
    return re_center - radius, re_center + radius, im_center - radius, im_center + radius

def minmax_normalize(image: ArrayLike) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Parameters
    ----------
    image : ArrayLike
        Input image to be normalized.

    Returns
    -------
    image : ndarray
        Normalized image with values scaled to the range [0, 1].
    """
    image -= np.min(image)
    img_max = np.max(image)
    if img_max > 0: image /= img_max
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
    Positive Laplacian of Gaussian (PosGoL) filter

    This filter is designed for detecting spectral graph from the spectral
    potential landscape. It applies a Gaussian blur followed by the Laplace
    operator.

    Parameters
    ----------
    image : ArrayLike
        Input image to be filtered.
    sigmas : iterable of floats, optional
        Sigmas used as scales for the Gaussian filter. Default is [0,1].
    ksizes : iterable of ints, optional
        Sizes of the discrete Laplacian operator. Default is [5].
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when False, 
        it detects white ridges.
    power_scaling : float, optional
        If provided, raises the normalized image to the given power before 
        applying the filter. For spectral potential landscape, power scaling
        by 1/n (n = 3, 5, ...) would sharpen the spectral graph.
    minmax_normalize : boolean, optional
        If True (the default), the image is normalized to the range [0, 1].

    Returns
    -------
    filtered_max : ndarray
        Filtered image with pixel-wise maximum response across all scales and 
        kernel sizes.

    Notes
    -----
    The PosGoL filter applies a Gaussian blur followed by the Laplace operator.
    Positive curvature regions are enhanced by taking the maximum of the Laplace 
    response and zero, normalizing it, and then taking the pixel-wise maximum 
    response across all specified scales and kernel sizes.
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
            lap = laplace(image, ksize=ksize)
            gauss = gaussian(lap, sigma=sigma) if sigma > 0 else lap
            # remove negative curvature
            pos = np.maximum(gauss, 0)
            # normalize to max = 1 unless all zeros
            max_val = pos.max()
            if max_val > 0: pos /= max_val
            filtered_max = np.maximum(filtered_max, pos)
            
    return filtered_max

def _trim_c(
    c: np.ndarray,
    is_one_band: bool
) -> tuple:
    
    if is_one_band:
        c_nonzeros = c != 0
    else:
        c_nonzeros = np.any(c, axis=0)
    
    l0 = np.argmax(c_nonzeros)  # first non-zero index
    m0 = len(c_nonzeros) // 2  # middle index
    r0 = np.argmax(c_nonzeros.cumsum())  # last non-zero index

    z0 = m0 - l0  # position of z^0 after trimming zeros
    q = r0 - m0  # largest power of z
    if z0 <= 0 or q <= 0:
        raise ValueError("Unphysical polynomial coefficients. Ensure both negative and positive powers are present.")
    
    c_trimmed = c[..., l0:r0 + 1] # trim the zeros
    # length = r0 - l0 + 1 # length of the pure polynomial

    return c_trimmed, z0, q

@njit
def _coeff_one_band(
    c: np.ndarray,
    E_array: np.ndarray,
    z0: int
) -> np.ndarray:
    
    coeff = np.zeros((E_array.size, len(c)), dtype=np.complex64)
    for i in range(len(c)):
        coeff[:, i] = c[i]
    coeff[:, z0] -= E_array.ravel()

    return coeff

def _coeff_multi_band(
    c_grid: np.ndarray,
    E_array: np.ndarray,
) -> np.ndarray:
    
    E_pow_len = c_grid.shape[0] # number of E powers
    powers = np.arange(E_pow_len) - E_pow_len//2 # exponents of E
    E_powers = E_array[..., np.newaxis] ** powers # shape: (E_len, E_len, E_pow_len)
    coeff_grid = np.einsum('ijk,kl->ijl', E_powers, c_grid) # shape: (E_len, E_len, i_len)

    return coeff_grid.reshape(-1, coeff_grid.shape[-1])

def spectral_potential(
    c: ArrayLike,
    E_max: Optional[Union[int, float, Sequence[float]]] = 3,
    E_len: Optional[int] = 400,
    E_array: Optional[np.ndarray] = None,
    method: Optional[Union[int, str]] = None
) -> np.ndarray:
    '''
    Generate the spectral potential landscape Phi(E) for a given multi-band polynomial.

    Parameters
    ----------
    c : ArrayLike
        Coefficient matrix of the polynomial. 
        - For one-band polynomial, c should be a 1D array of (symmetric) coefficients, 
        ignoring the -E term. The middle term corresponds to z^0.
        - For multi-band polynomial, c should be a 2D array of coefficients, with the
        c[i_max//2, j_max//2] element representing the coefficient of (z^0 E^0). The
        shape of c in the second dimension (z powers) should be odd, with the middle
        term corresponding to z^0 E^*.
    E_max : int, float, or Sequence[float], optional
        Maximum energy range for the landscape. If a float is provided, the range is 
        [-E_max, E_max] for both real and imaginary parts. If a list(like), it should be 
        [E_real_min, E_real_max, E_imag_min, E_imag_max].
    E_len : int, optional
        Number of points in the energy range. Default is 400.
    E_array : ndarray, optional
        Complex energy array to compute the TDL spectra. If provided, it will override
        the energy range calculated from E_max and E_len. Default is None.
    method : int or str, optional
        Method to calculate the spectral landscape where the *ridges* are the spectral
        graph skeleton:
            - None or 1 or 'spectral' : spectral potential landscape
            - 2 or 'diff_log' : difference of inverse skin depths corresponding to 
                                the least 2 |z|'s
            - 3 or 'log_diff' : log difference of the least 2 |z|'s
        Default is None.

    Returns
    -------
    phi : ndarray
        The 2D image of the spectral potential landscape Phi(E).

    Examples
    --------
    One-band polynomial:
    >>> c = np.array([1, .4, 1, .1, 0, 0, .2, -.4, 1])
    >>> phi = spectral_potential(c, E_max=3.5)

    Multi-band polynomial (Hatano-Nelson model):
    >>> c = np.array([[0,  0, 0],
                      [.5, 0, 1],
                      [0, -1, 0]])
    >>> phi = spectral_potential(c, E_max=2, E_len=400)
    '''

    c = np.asarray(c)
    is_one_band = len(c.shape) == 1

    if E_array is None:
        if isinstance(E_max, (int, float)):
            E_re_min, E_re_max, E_im_min, E_im_max = -E_max, E_max, -E_max, E_max
        elif isinstance(E_max, (tuple, list, np.ndarray)) & (len(E_max) == 4):
            E_re_min, E_re_max, E_im_min, E_im_max = E_max
        else:
            raise ValueError("Invalid E_max. Provide a float or a list of 4 floats.")
        E_array = np.linspace(E_re_min, E_re_max, E_len) + \
                    1j*np.linspace(E_im_min, E_im_max, E_len)[:, None]

    c_trimmed, z0, q = _trim_c(c, is_one_band)
    if is_one_band:
        coeff = _coeff_one_band(c_trimmed, E_array, z0)
    else:
        coeff = _coeff_multi_band(c_trimmed, E_array)

    # Compute roots in z for each E
    coeff_tf = tf.constant(coeff, dtype=tf.complex64)
    z = poly_roots_tf_batch(coeff_tf).numpy()  # Shape: (E_array.size, degree)

    # Proceed to compute phi based on the selected method
    if method is None or method == 1 or method == 'spectral':
        # Method 1: spectral potential landscape
        zs_q = np.sort(np.abs(z), axis=-1)[:, -q:]  # q largest |z|'s
        phi = np.log(np.abs(coeff[:, -1])) + np.sum(np.log(zs_q), axis=-1)
    elif method == 2 or method == 'diff_log':
        # Method 2: the difference of kappa's corresponding to the least 2 |z|'s
        kappas = -np.log(np.sort(np.abs(z), axis=-1))
        kappa_diffs = kappas[:, :-1] - kappas[:, 1:]
        phi = -kappa_diffs[:, 0]
        # phi = -np.min(kappa_diffs, axis=-1)
    elif method == 3 or method == 'log_diff':
        # Method 3: log difference of the least 2 |z|'s
        zs = np.sort(np.abs(z), axis=-1)
        min_z_diffs = zs[:, 1] - zs[:, 0]
        # min_z_diffs = np.min(zs[:, 1:] - zs[:, :-1], axis=-1)
        phi = -np.log(min_z_diffs/np.max(min_z_diffs))
        # replace nan with the max of the rest
        phi[np.isnan(phi)] = np.nanmax(phi)
    else:
        raise ValueError("Invalid method specified. Choose 1, 2, or 3.")
    
    return -phi.reshape(E_array.shape)

def spectral_images_adaptive_resolution(
    c: ArrayLike,
    E_max: Union[int, float, Sequence[float]],
    E_len: Optional[int] = 512,
    E_splits: Optional[int] = 1,
    thresholder: Optional[Callable] = np.mean,
    PosGoL_kwargs: Optional[dict] = {},
    potential_method: Optional[Union[int, str]] = None
) -> np.ndarray:
    """
    Generate the spectral potential landscape, density of states, and graph
    'strips' of a given one-band / multi-band characteristic polynomial.
    Adaptive resolution and parallellized root-finder computation are used.

    Parameters
    ----------
    c : ArrayLike
        Coefficient matrix of the polynomial. 
        - For one-band polynomial, c should be a 1D array of (symmetric) coefficients, 
        ignoring the -E term. The middle term corresponds to z^0.
        - For multi-band polynomial, c should be a 2D array of coefficients, with the
        c[i_max//2, j_max//2] element representing the coefficient of (z^0 E^0). The
        shape of c in the second dimension (z powers) should be odd, with the middle
        term corresponding to z^0 E^*.
    E_max : int, float, or Sequence[float], optional
        Maximum energy range for the landscape. If a real number is provided, the range
        is [-E_max, E_max] for both real and imaginary parts. If a list, it should 
        contain [E_real_min, E_real_max, E_imag_min, E_imag_max].
    E_len : int, optional
        Number of point per dimension of the energy grid. Used to filter a rough region
        covering the spectral graph. Default is 512.
    E_splits : float, optional
        Energy grid resolution enhancement factor. The resolution applied on the filtered
        region is E_splits times higher than the original grid. Default is 1.
    thresholder : callable, optional
        Function to threshold the ridge image. Default is np.mean.
    PosGoL_kwargs : dict, optional
        Additional keyword arguments for the PosGoL filter.
    potential_method : int or str, optional
        Method to calculate the spectral potential landscape where the *ridges* are the 
        spectral graph skeleton:
            - None or 1 or 'spectral' : spectral potential landscape
            - 2 or 'diff_log' : difference of inverse skin depths corresponding to 
                                the least 2 |z|'s
            - 3 or 'log_diff' : log difference of the least 2 |z|'s
        Default is None.

    Returns
    -------
    (binary, phi, ridge) : tuple
    
    binary : ndarray
        Binarized density of states image, i.e. the graph 'strips' before skeletonization.
    phi : ndarray
        The 2D image of the spectral potential landscape Phi(E).
    dos : ndarray
        The 2D image of the density of states.
    """
    
    c = np.asarray(c)
    
    if isinstance(E_max, (int, float)):
        E_re_min, E_re_max, E_im_min, E_im_max = -E_max, E_max, -E_max, E_max
    elif isinstance(E_max, (tuple, list, np.ndarray)) & (len(E_max) == 4):
        E_re_min, E_re_max, E_im_min, E_im_max = E_max
    else:
        raise ValueError("Invalid E_max. Provide a float or a list of 4 floats.")
    E_arr = np.linspace(E_re_min, E_re_max, E_len) + \
                1j*np.linspace(E_im_min, E_im_max, E_len)[:, None]

    phi = spectral_potential(c, E_array=E_arr, method=potential_method)
    if potential_method in [None, 1, 'spectral']:
        ridge = PosGoL(phi, **PosGoL_kwargs)
    else:
        ridge = phi
    binary = ridge > thresholder(ridge)
    if E_splits <= 1 or E_splits is None:
        return binary, phi, ridge
    
    mask1 = np.where(binary)
    mask0 = np.where(~binary)
    dilated = dilation(binary, disk(2))
    mask1_ = np.where(dilated)
    # mask0_ = np.where(~dilated)

    E_split = np.linspace(E_re_min, E_re_max, E_splits*E_len) + \
                        1j*np.linspace(E_im_min, E_im_max, E_splits*E_len)[:, None]
    E_block = view_as_blocks(E_split, (E_splits, E_splits))
    masked_E_block = E_block[mask1_]

    phi_ = np.repeat(np.repeat(phi, E_splits, axis=0), E_splits, axis=1)
    phi_block = view_as_blocks(phi_, (E_splits, E_splits))
    phi_dense = spectral_potential(c, E_array=masked_E_block)
    phi_block[mask1_] = phi_dense
    ridge_ = PosGoL(phi_, **PosGoL_kwargs)
    ridge_block = view_as_blocks(ridge_, (E_splits, E_splits))
    ridge_block[mask0] = 0
    
    weights = np.array([ridge_block[mask1].size, ridge[mask0].size * E_splits**2])
    means = np.array([np.mean(ridge_block[mask1]), np.mean(ridge[mask0])])
    threshold = np.dot(weights, means) / np.sum(weights)
    binary_ = ridge_ > threshold
    binary_block = view_as_blocks(binary_, (E_splits, E_splits))
    binary_block[mask0] = 0

    return binary_, phi_, ridge_

### Spectral Graph utensils and generator ###

def delete_iso_nodes(
    G: nxGraph,
    copy: Optional[bool] = True
) -> nxGraph:
    '''
    Remove isolated nodes from a networkx graph.
    '''
    del_G = G.copy() if copy else G
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    del_G.remove_nodes_from(isolated_nodes)
    return del_G

# def delete_iso_nodes(G, copy=True):
#     del_G = G.copy() if copy else G
#     return del_G.subgraph([n for n in G.nodes() if G.degree(n) > 0])

# TODO: modify the function to handle any set of attributes
def _average_attributes(node):
    # Check if attributes exist, otherwise initialize them
    # if all(key in node for key in ['o', 'dos', 'potential']):
    if 'o' in node:
        sum_o = np.array(node['o'], dtype=float)
        sum_dos = node['dos'] if 'dos' in node else 0.0
        sum_potential = node['potential'] if 'potential' in node else 0.0
        count = 1
    else:
        sum_o = np.zeros(2, dtype=float)  # Assuming 'o' is a 2D array based on the initial example
        sum_dos = 0.0; sum_potential = 0.0
        count = 0
    
    # If there is no 'contraction' field, return the current sums and count
    if 'contraction' not in node:
        return sum_o, sum_dos, sum_potential, count
    
    # Recursively process the contracted nodes
    for _, contracted_node in node['contraction'].items():
        o, dos, potential, n = _average_attributes(contracted_node)
        sum_o += np.array(o, dtype=float)
        sum_dos += dos; sum_potential += potential
        count += n
    
    # Avoid division by zero
    if count > 0:
        avg_o = sum_o / count
        avg_dos = sum_dos / count
        avg_potential = sum_potential / count
    else:
        avg_o = sum_o; avg_dos = sum_dos; avg_potential = sum_potential
    
    return avg_o, avg_dos, avg_potential, count

def process_contracted_graph(G: nxGraph) -> nxGraph:
    processed_graph = G.copy()
    for node, attr in processed_graph.nodes(data=True):
        avg_o, avg_dos, avg_potential, _ = _average_attributes(attr)
        processed_graph.nodes[node]['o'] = avg_o
        if avg_dos != 0.0:
            processed_graph.nodes[node]['dos'] = avg_dos
        if avg_potential != 0.0:
            processed_graph.nodes[node]['potential'] = avg_potential
        # Remove the contraction field as it's no longer needed
        if 'contraction' in attr:
            del processed_graph.nodes[node]['contraction']
    return processed_graph

def contract_close_nodes(
    G: nxGraph,
    threshold: Union[int, float]
) -> nxGraph:
    '''
    Delete isolated nodes and contract short edges in a spectral graph.
    '''
    G = delete_iso_nodes(G) # remove isolated nodes
    contracted_graph = G.copy()
    while True:
        sorted_edges = sorted(contracted_graph.edges(data='weight'), key=lambda x: x[2])
        # Check if all edges are above the threshold
        if all(l >= threshold for _, _, l in sorted_edges):
            break
        for u, v, l in sorted_edges:
            if l < threshold:
                try:
                    temp_graph = process_contracted_graph(nx.contracted_nodes(contracted_graph, u, v, self_loops=False, copy=True))
                    # print(temp_graph.nodes(data=True)) # Debugging
                    if temp_graph.number_of_edges() == 0:
                        return G  # Return the original graph if contraction leads to no edges
                    contracted_graph = temp_graph
                except:
                    pass
    return delete_iso_nodes(contracted_graph) # remove isolated clusters

def spectral_graph(
    c: ArrayLike,
    E_max: Union[int, float, Sequence[float]],
    E_len: Optional[int] = 512,
    E_splits: Optional[int] = 1,
    Potential_feature: Optional[bool] = True,
    DOS_feature: Optional[bool] = True,
    scale_features: Optional[Union[int, float]] = 1,
    thresholder: Optional[Callable] = np.mean,
    contract_threshold: Optional[Union[int, float]] = 15,
    s2g_kwargs: Optional[dict] = {},
    PosGoL_kwargs: Optional[dict] = {},
    potential_method: Optional[Union[int, str]] = None
) -> nxGraph:
    '''
    Generate the spectral graph of a given one-band / multi-band characteristic polynomial.
    Adaptive resolution and parallellized root-finder computation are used.

    Parameters
    ----------
    c : ArrayLike
        Coefficient matrix of the polynomial. 
        - For one-band polynomial, c should be a 1D array of (symmetric) coefficients, 
        ignoring the -E term. The middle term corresponds to z^0.
        - For multi-band polynomial, c should be a 2D array of coefficients, with the
        c[i_max//2, j_max//2] element representing the coefficient of (z^0 E^0). The
        shape of c in the second dimension (z powers) should be odd, with the middle
        term corresponding to z^0 E^*.
    E_max : int, float, or Sequence[float], optional
        Maximum energy range for the landscape. If a real number is provided, the range
        is [-E_max, E_max] for both real and imaginary parts. If a list, it should 
        contain [E_real_min, E_real_max, E_imag_min, E_imag_max].
    E_len : int, optional
        Number of point per dimension of the energy grid. Used to filter a rough region
        covering the spectral graph. Default is 512.
    E_splits : float, optional
        Energy grid resolution enhancement factor. The resolution applied on the filtered
        region is E_splits times higher than the original grid. Default is 1.
    Potential_feature : bool, optional
        If True (the default), the spectral potential landscape is included as a node
        feature.
    DOS_feature : bool, optional
        If True (the default), the density of states is included as a node feature.
    scale_features : int or float, optional
        If None, the positions of nodes and edges are pixel indices, i.e., in 
        [0, E_len*E_splits-1]. If provided, the node features are scaled by this factor. 
        Default is 1.
    thresholder : callable, optional
        Function to threshold the ridge image. Default is np.mean.
    contract_threshold : int or float, optional
        Threshold for contracting close nodes. Default is 15.
    s2g_kwargs : dict, optional
        Additional keyword arguments for skeleton2graph.
    PosGoL_kwargs : dict, optional
        Additional keyword arguments for the PosGoL filter.
    potential_method : int or str, optional
        Method to calculate the spectral potential landscape where the *ridges* are the 
        spectral graph skeleton:
            - None or 1 or 'spectral' : spectral potential landscape
            - 2 or 'diff_log' : difference of inverse skin depths corresponding to 
                                the least 2 |z|'s
            - 3 or 'log_diff' : log difference of the least 2 |z|'s
        Default is None.

    Returns
    -------
    graph : (networkx.Graph or networkx.MultiGraph)
        The spectral graph of the given characteristic polynomial.

    Examples
    --------
    One-band polynomial:
    >>> c = np.array([1, .4, 1, .1, 0, 0, .2, -.4, 1])
    >>> graph = spectral_graph(c, E_max=3.5)

    Multi-band polynomial (Hatano-Nelson model):
    >>> c = np.array([[0,  0, 0],
                      [.5, 0, 1],
                      [0, -1, 0]])
    >>> graph = spectral_graph(c, E_max=2)
    '''

    binary, phi, ridge = spectral_images_adaptive_resolution(c, E_max, 
                E_len, E_splits, thresholder, PosGoL_kwargs, potential_method)

    ske = skeletonize(binary, method='lee')

    Potential_image = phi.astype(np.float32) if Potential_feature else None
    DOS_image = ridge.astype(np.float32) if DOS_feature else None
    graph = skeleton2graph(ske, Potential_image=Potential_image, DOS_image=DOS_image, **s2g_kwargs)
    if contract_threshold is not None:
        graph = contract_close_nodes(graph, contract_threshold)

    if scale_features is not None:
        Elen = E_len * E_splits
        if isinstance(E_max, (int, float)):
            center = np.array([0, 0])
            scale = E_max / Elen
        elif isinstance(E_max, (tuple, list, np.ndarray)) & (len(E_max) == 4):
            center = np.array([E_max[2] + E_max[3], E_max[0] + E_max[1]]) / 2
            scale = (E_max[1] - E_max[0]) / Elen
        _center = np.array([Elen-1, Elen-1])/2 # offset 1 for 0-based indexing

        for node in graph.nodes(data=True):
            if 'o' in node[1]:
                # to float32 for compatibility with torch_geometric
                node[1]['o'] = (((node[1]['o']-_center)*scale + center)*scale_features).astype(np.float32)
        
        for edge in graph.edges(data=True):
            if 'weight' in edge[2]:
                edge[2]['weight'] = (edge[2]['weight']*scale*scale_features).astype(np.float32)
            if 'pts' in edge[2]:
                edge[2]['pts'] = (((edge[2]['pts']-_center)*scale + center)*scale_features).astype(np.float32)
        E_len = Elen

    attrs = {'polynomial_coeff': c.astype(np.float32), 'E_max': E_max, 'E_len': E_len}
    graph.graph.update(attrs)

    return graph

def draw_image(
    image: ArrayLike,
    ax: Optional[Any] = None,
    overlay_graph: Optional[bool] = False,
    contract_threshold: Optional[Union[int, float]] = None,
    ax_set_kwargs: Optional[dict] = {},
    s2g_kwargs: Optional[dict] = {}
) -> None:
    def to_graph(img, **kwargs):
        ske = skeletonize(img, method='lee')
        graph = skeleton2graph(ske, **kwargs)
        return graph

    if ax is None: ax = plt.gca()
    ax.imshow(image, cmap='bone')
    ax.set(xlabel='Re(E)', ylabel='Im(E)', **ax_set_kwargs)
    ax.axis('off')

    if overlay_graph:
        overlay_graph = to_graph(image, add_pts=True, **s2g_kwargs)
        if contract_threshold is not None:
            overlay_graph = contract_close_nodes(overlay_graph, contract_threshold)
        for (s, e, key, ps) in overlay_graph.edges(keys=True, data='pts'):
            # ps = overlay_graph[s][e][key]['pts']
            ax.plot(ps[:,1], ps[:,0], 'b-', lw=1, alpha=0.8)
        nodes = overlay_graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:,1], ps[:,0], 'r.')