import numpy as np
import networkx as nx
import tensorflow as tf

import sympy as sp
from sympy.polys.polytools import Poly

from skimage.morphology import skeletonize, dilation, binary_closing, disk
from skimage.util import view_as_blocks

from .skeleton2graph import skeleton2graph
from .spectral_graph import (
    PosGoL,
    spectral_potential,
    add_edges_within_threshold,
    contract_close_nodes
)
from .hamiltonian import (
    hk2hz_1d, hz2hk_1d,
    expand_hz_as_hop_dict_1d,
    H_1D_batch_from_hop_dict
)
from .util import companion_batch

from numpy.typing import ArrayLike
from typing import Union, Optional, Callable, Iterable, TypeVar

nxGraph = TypeVar('nxGraph', nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)


class SpectralGraph:
    """
    A class that encapsulates the construction and analysis of a Bloch Hamiltonian 
    and its associated (complex-energy) spectral geometry.

    Attributes:
        k (sp.Symbol): Symbolic variable representing momentum (real).
        z (sp.Symbol): Symbolic variable for the complex exponential `z = e^{i k}`.
        E (sp.Symbol): Symbolic variable for energy.
        ChP (Poly): The characteristic polynomial (in E and z).
        h_z (sp.Matrix): Bloch Hamiltonian expressed in terms of `z`.
        h_k (sp.Matrix): Bloch Hamiltonian expressed in terms of `k`.
        num_bands (int): Number of bands (i.e., dimension of the Hamiltonian matrix).
        Poly_z_coeff (Sequence[sp.Expr]): Coefficients (in z) after multiplying by a power 
            of z to make it a polynomial in z alone.
        poly_p (int): The largest power of 1/z in the characteristic polynomial factor.
        poly_q (int): The largest power of z in the characteristic polynomial factor.
        Poly_E_coeff (Sequence[sp.Expr]): Coefficients (in E) of the characteristic polynomial.
        companion_E (sp.Matrix): Companion matrix in z for a fixed E. 
        spectral_square (np.ndarray): A bounding box around the spectrum in the complex plane.
    """

    def __init__(
        self,
        characteristic: Union[Poly, str, sp.Matrix],
        k: sp.Symbol,
        z: sp.Symbol,
        E: sp.Symbol
    ) -> None:
        """
        Initialize the SpectralGraph object.

        This constructor accepts a characteristic polynomial (as a sympy Poly, 
        a string that sympify can parse, or a sympy Matrix that either depends 
        on `k` or on `z`). Depending on the input, it sets up the Bloch Hamiltonian 
        and the characteristic polynomial in a consistent manner.

        Args:
            characteristic (Union[Poly, str, sp.Matrix]): The characteristic polynomial 
                or Hamiltonian matrix (depending on usage).
            k (sp.Symbol): Symbolic variable representing momentum (real).
            z (sp.Symbol): Symbolic variable for the complex exponential `z = e^{i k}`.
            E (sp.Symbol): Symbolic variable for energy.

        Raises:
            ValueError: If the characteristic polynomial does not include the required 
                symbols, or if it includes both `k` and `z` (ambiguous), or if the form 
                is unrecognized.
        """
        self.k, self.z, self.E = k, z, E

        if isinstance(characteristic, Poly):
            self.ChP = characteristic
            self._init_ChP()
        elif isinstance(characteristic, str):
            expr = sp.sympify(characteristic, locals={'z': z, 'E': E})
            assert {E, z}.issubset(expr.free_symbols), (
                f"ChP must include {E} AND {z} as free symbols"
            )
            self.ChP = Poly(expr, z, 1/z, E)
            self._init_ChP()
        elif isinstance(characteristic, sp.Matrix):
            free_sym = characteristic.free_symbols
            if self.k in free_sym and self.z not in free_sym:
                self.h_k = characteristic
                self.h_z = hk2hz_1d(self.h_k, k, z)
            elif self.z in free_sym and self.k not in free_sym:
                self.h_z = characteristic
                self.h_k = hz2hk_1d(self.h_z, k, z)
            else:
                raise ValueError(
                    f"Characteristic polynomial must include {k} XOR {z} as a free symbol"
                )
            self._init_bloch()
        else:
            raise ValueError("Characteristic polynomial must be a Poly, string, or Matrix.")

        self._companion_E()
        self._spectral_boundaries()

    def _init_ChP(self) -> None:
        """
        Internal method to initialize the Bloch Hamiltonian and polynomial data
        from a given characteristic polynomial (ChP).

        The method constructs:
        - self.h_z: Bloch Hamiltonian in terms of z
        - self.h_k: Bloch Hamiltonian in terms of k
        - self.num_bands: Dimension of the Hamiltonian
        - self.Poly_E_coeff: Coefficients in E

        Raises:
            AssertionError: If the polynomial does not include E and z as free symbols.
        """
        k, z, E = self.k, self.z, self.E
        assert {E, z}.issubset(self.ChP.free_symbols), (
            "ChP must include E and z as free symbols"
        )
        assert set(self.ChP.gens) == {z, 1/z, E}, (
            "ChP's generators must be {z, 1/z, E}"
        )

        # Treat z as constant and E as variable
        Poly_E = Poly(self.ChP.as_expr(), E)
        self.Poly_E_coeff = Poly_E.all_coeffs()
        self.num_bands = Poly_E.degree()

        # Bloch Hamiltonian
        if self.num_bands == 1:
            coeff = Poly_E.monic().all_coeffs()[-1]
            self.h_z = sp.Matrix([-coeff])
        else:
            self.h_z = sp.Matrix.companion(Poly_E.monic()).applyfunc(sp.expand)
        self.h_k = hz2hk_1d(self.h_z, k, z)

    def _init_bloch(self) -> None:
        """
        Internal method to initialize characteristic polynomial data 
        from a given Bloch Hamiltonian (h_k or h_z).
        
        The method sets up:
        - self.h_z, self.h_k (ensuring both exist)
        - self.ChP
        - self.Poly_E_coeff
        - self.num_bands
        """
        z, E = self.z, self.E
        # Characteristic polynomial
        Poly_E = self.h_z.charpoly(E)
        self.Poly_E_coeff = Poly_E.all_coeffs()
        self.num_bands = Poly_E.degree()
        self.ChP = Poly(Poly_E.as_expr(), z, 1/z, E)

    def _companion_E(self) -> None:
        """
        Construct and store the companion matrix in z for a fixed E. 
        
        This is done by treating E as a constant and z as a variable, and 
        multiplying the polynomial by the appropriate power of z to get 
        a standard polynomial form in z.
        """
        # Treat E as constant and z as variable
        Poly_z_bigen = Poly(self.ChP.as_expr(), self.z, 1/self.z)
        self.poly_p = Poly_z_bigen.degree(1/self.z)
        self.poly_q = Poly_z_bigen.degree(self.z)
        Poly_z = Poly(
            sp.expand(self.ChP.as_expr() * self.z**self.poly_p),
            self.z
        )
        self.Poly_z_coeff = Poly_z.all_coeffs()
        # Companion matrix of P(E)(z) for efficient root finding
        self.companion_E = sp.Matrix.companion(Poly_z.monic()).applyfunc(sp.expand)

    def real_space_H(
        self,
        N: int = 40,
        max_dim: int = 150,
        pbc: bool = False,
        param_dict: dict = {}
    ) -> np.ndarray:
        """
        Construct a finite real-space Hamiltonian of size (num_bands*N) x (num_bands*N).

        Args:
            N (int, optional): Number of unit cells in 1D chain. Defaults to 40.
            max_dim (int, optional): Maximum dimension of the Hamiltonian matrix. 
                If `num_bands * N > max_dim`, N is reduced accordingly. Defaults to 150.
            pbc (bool, optional): If True, use periodic boundary conditions. 
                Otherwise, open boundary conditions. Defaults to False.
            param_dict (dict, optional): Dictionary of any symbolic parameters 
                (other than z and E) to be substituted into the Hamiltonian. Defaults to {}.

        Returns:
            np.ndarray: The finite real-space Hamiltonian.
        """
        if self.num_bands * N > max_dim:
            N = max_dim // self.num_bands
        hop_dict = expand_hz_as_hop_dict_1d(self.h_z, self.z)
        H = H_1D_batch_from_hop_dict(hop_dict, N, pbc, param_dict)
        return H

    def _spectral_boundaries(self) -> None:
        """
        Estimate a bounding circle and square around the spectrum in the complex plane
        by diagonalizing a moderate-size finite chain Hamiltonian.

        The results are stored in:
        - self.spectral_center
        - self.spectral_radius
        - self.spectral_square
        """
        finite_chain = self.real_space_H()
        E = np.linalg.eigvals(finite_chain)

        # dilate the boundary for safety
        pad_factor = 0.05
        re_min, re_max = np.min(E.real), np.max(E.real)
        re_radius, re_center = (re_max - re_min)/2, (re_max + re_min)/2
        im_min, im_max = np.min(E.imag), np.max(E.imag)
        im_radius, im_center = (im_max - im_min)/2, (im_max + im_min)/2
        radius = max(re_radius, im_radius) * (1 + pad_factor)

        self.spectral_center = np.array([re_center, im_center])
        self.spectral_radius = radius
        self.spectral_square = np.array([
            re_center - radius, re_center + radius,
            im_center - radius, im_center + radius
        ])

    def _Poly_z_coeff_arr(self, E_array: ArrayLike) -> np.ndarray:
        """
        Evaluate the coefficients (in z) of the characteristic polynomial 
        for a given array of E values.

        Args:
            E_array (ArrayLike): Array of complex energies for which to evaluate.

        Returns:
            np.ndarray: Coefficients of the polynomial in z, 
            shape = (*E_array.shape, len(self.Poly_z_coeff)).
        """
        E_array = np.asarray(E_array)
        coeff_arr = np.zeros(
            (*E_array.shape, len(self.Poly_z_coeff)), dtype=np.complex128
        )
        for i, coeff in enumerate(self.Poly_z_coeff):
            if coeff.free_symbols == set():
                coeff_arr[..., i] = coeff
            elif coeff.free_symbols == {self.E}:
                f = sp.lambdify(self.E, coeff, modules='numpy')
                coeff_arr[..., i] = f(E_array)
            else:
                raise ValueError("Poly_z_coeff must be a function of E only")
        return coeff_arr

    def Poly_z_roots(
        self,
        E_array: ArrayLike,
        device: str = '/cpu:0'
    ) -> np.ndarray:
        """
        Compute the roots in z for each energy in E_array by constructing and 
        diagonalizing the companion matrices (in z).

        Args:
            E_array (ArrayLike): Array of complex energies for which to find z-roots.
            device (str, optional): TensorFlow device string to control GPU/CPU usage. 
                Defaults to '/cpu:0'.

        Returns:
            np.ndarray: Roots in z for each energy, shape = (*E_array.shape, poly_degree).
        """
        coeff_arr = self._Poly_z_coeff_arr(E_array)
        companion_arr = companion_batch(coeff_arr)
        with tf.device(device):
            companion_tensor = tf.convert_to_tensor(companion_arr)
            roots = tf.linalg.eigvals(companion_tensor)
        return roots.numpy()

    def spectral_potential(
        self,
        E_array: ArrayLike,
        method: str = 'ronkin',
        device: str = '/cpu:0'
    ) -> np.ndarray:
        """
        Compute the spectral potential (e.g., the Ronkin function) for an array of energies.

        Args:
            E_array (ArrayLike): Array of complex energies.
            method (str, optional): Which method to use for the potential 
                ('ronkin' or other). Defaults to 'ronkin'.
            device (str, optional): TensorFlow device (e.g., '/cpu:0' or '/gpu:0'). 
                Defaults to '/cpu:0'.

        Returns:
            np.ndarray: The evaluated potential for each E in E_array, 
            shape = E_array.shape.
        """
        coeff_arr = self._Poly_z_coeff_arr(E_array)
        roots = self.Poly_z_roots(E_array, device=device)
        phi = spectral_potential(roots, coeff_arr, self.poly_q, method=method)
        return phi

    def spectral_images(
        self,
        resolution: int = 256,
        resolution_enhancement: int = 4,
        device: str = '/cpu:0',
        method: str = 'ronkin',
        DOS_filter_kwargs: Optional[dict] = None
    ) -> tuple:
        """
        Create images of the spectral potential, a filtered (DOS-like) image, 
        and a binary mask of the spectral region.

        Args:
            resolution (int, optional): Resolution in each axis. Defaults to 256.
            resolution_enhancement (int, optional): Factor by which to refine the 
                resolution in the region near the spectral boundary. Defaults to 4.
            device (str, optional): TensorFlow device. Defaults to '/cpu:0'.
            method (str, optional): Method for computing spectral potential. 
                Defaults to 'ronkin'.
            DOS_filter_kwargs (dict, optional): Keyword arguments for `PosGoL` 
                filter or similar. Defaults to None.

        Returns:
            tuple: (phi, ridge, binary), or the enhanced versions (phi_, ridge_, binary_) 
            if `resolution_enhancement > 1`.
        """
        if DOS_filter_kwargs is None:
            DOS_filter_kwargs = {}

        E_box = self.spectral_square
        E_arr = (
            np.linspace(*E_box[:2], resolution)
            + 1j*np.linspace(*E_box[2:], resolution)[:, None]
        )

        phi = self.spectral_potential(E_arr, method=method, device=device)
        if method == 'ronkin':
            ridge = PosGoL(phi, **DOS_filter_kwargs)
        else:
            ridge = phi

        binary = ridge > np.mean(ridge)
        if resolution_enhancement <= 1 or resolution_enhancement is None:
            return phi, ridge, binary
        
        # resolution enhancement on a filtered region
        mask1 = np.where(binary)
        mask0 = np.where(~binary)
        dilated = dilation(binary, disk(2))
        mask1_ = np.where(dilated)
        
        enhanced_resolution = resolution * resolution_enhancement
        E_split = (
            np.linspace(*E_box[:2], enhanced_resolution)
            + 1j*np.linspace(*E_box[2:], enhanced_resolution)[:, None]
        )
        E_block = view_as_blocks(E_split, (resolution_enhancement, resolution_enhancement))
        masked_E_block = E_block[mask1_]

        split_kernel = np.ones((resolution_enhancement, resolution_enhancement))
        phi_ = np.kron(phi, split_kernel)
        phi_block = view_as_blocks(phi_, (resolution_enhancement, resolution_enhancement))
        phi_dense = self.spectral_potential(masked_E_block, method=method, device=device)
        phi_block[mask1_] = phi_dense

        ridge_ = PosGoL(phi_, **DOS_filter_kwargs)
        ridge_block = view_as_blocks(ridge_, (resolution_enhancement, resolution_enhancement))
        ridge_block[mask0] = 0

        weights = np.array([
            ridge_block[mask1].size, 
            ridge[mask0].size * resolution_enhancement**2
        ])
        means = np.array([
            np.mean(ridge_block[mask1]),
            np.mean(ridge[mask0])
        ])
        threshold = np.dot(weights, means) / np.sum(weights)
        binary_ = ridge_ > threshold
        binary_block = view_as_blocks(binary_, (resolution_enhancement, resolution_enhancement))
        binary_block[mask0] = 0

        return phi_, ridge_, binary_

    def spectral_graph(
        self,
        resolution: int = 256,
        resolution_enhancement: int = 4,
        device: str = '/cpu:0',
        method: str = 'ronkin',
        short_edge_threshold: Optional[float] = 20,
        skeleton2graph_kwargs: Optional[dict] = None,
        DOS_filter_kwargs: Optional[dict] = None,
        magnify: float = 1.0
    ) -> nxGraph:
        """
        Build a graph representation from the complex-energy spectral geometry.

        The routine computes a potential (e.g. Ronkin function), extracts 
        a skeleton from the thresholded DOS/ridge, and then converts 
        that skeleton into a NetworkX graph.

        Args:
            resolution (int, optional): Resolution in each axis for the coarse grid. 
                Defaults to 256.
            resolution_enhancement (int, optional): Factor by which to refine resolution. 
                Defaults to 4.
            device (str, optional): TensorFlow device. Defaults to '/cpu:0'.
            method (str, optional): Method for computing the spectral potential. 
                Defaults to 'ronkin'.
            short_edge_threshold (Optional[float], optional): If provided, edges shorter 
                than this threshold (in pixel units of the final image) are 
                added and contracted to reduce graph noise. Defaults to 20.
            skeleton2graph_kwargs (Optional[dict], optional): Additional kwargs for 
                the skeleton-to-graph conversion. Defaults to None.
            DOS_filter_kwargs (Optional[dict], optional): Additional kwargs for the 
                DOS or ridge filtering method. Defaults to None.
            magnify (float, optional): A factor to scale the positions and edge weights 
                in the final graph for numeric stability or aesthetics. Defaults to 1.0.

        Returns:
            nxGraph: A (NetworkX) graph representation of the spectral skeleton.
        """
        if skeleton2graph_kwargs is None:
            skeleton2graph_kwargs = {}
        if DOS_filter_kwargs is None:
            DOS_filter_kwargs = {}

        phi, ridge, binary = self.spectral_images(
            resolution=resolution,
            resolution_enhancement=resolution_enhancement,
            device=device,
            method=method,
            DOS_filter_kwargs=DOS_filter_kwargs
        )
        # obtain graph skeleton
        ske = skeletonize(binary, method='lee')
        # construct skeleton graph
        graph = skeleton2graph(
            ske,
            Potential_image=phi.astype(np.float32),
            DOS_image=ridge.astype(np.float32),
            **skeleton2graph_kwargs
        )

        ### post-process the extracted graph
        # merge close nodes and short edges
        if short_edge_threshold is not None and short_edge_threshold > 0:
            graph = add_edges_within_threshold(graph, short_edge_threshold)
            graph = contract_close_nodes(graph, short_edge_threshold)

        # convert the graph attributes from pixel index to energy values
        final_res = resolution * resolution_enhancement
        cen = self.spectral_center
        scale = self.spectral_radius * 2 / final_res
        cen_ = np.array([final_res - 1, final_res - 1]) / 2  # offset for 0-based indexing

        if magnify is not None and magnify > 0:
            for node in graph.nodes(data=True):
                if 'pos' in node[1]:
                    pos = np.asarray(node[1]['pos'], dtype=np.float32)
                    # Recover the (x, y) coordinates from the 2D array indices
                    new_pos = (pos[::-1] - cen_) * scale + cen
                    node[1]['pos'] = new_pos * magnify
                if 'pts' in node[1]:
                    pts = np.asarray(node[1]['pts'], dtype=np.float32)
                    new_pts = (pts[..., ::-1] - cen_) * scale + cen
                    node[1]['pts'] = new_pts * magnify

            for edge in graph.edges(data=True):
                if 'weight' in edge[2]:
                    weight = np.asarray(edge[2]['weight'], dtype=np.float32)
                    new_weight = weight * scale
                    edge[2]['weight'] = new_weight * magnify
                if 'pts' in edge[2]:
                    pts = np.asarray(edge[2]['pts'], dtype=np.float32)
                    new_pts = (pts[..., ::-1] - cen_) * scale + cen
                    edge[2]['pts'] = new_pts * magnify

        return graph