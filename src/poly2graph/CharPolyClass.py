import numpy as np
import sympy as sp
import networkx as nx
import tensorflow as tf

from skimage.morphology import skeletonize, dilation, binary_closing, disk
from skimage.util import view_as_blocks

from joblib import Parallel, delayed
from functools import partial
import warnings

from poly2graph.skeleton2graph import skeleton2graph, skeleton2graph_batch
from poly2graph.spectral_graph import (
    PosGoL,
    spectral_potential,
    add_edges_within_threshold,
    contract_close_nodes
)
from poly2graph.hamiltonian import (
    hk2hz_1d, hz2hk_1d,
    expand_hz_as_hop_dict_1d,
    H_1D_batch_from_hop_dict
)
from poly2graph.util import companion_batch, kron_batch, eig_batch, eigvals_batch

from numpy.typing import ArrayLike
from typing import Union, Optional, Callable, Dict, List, Tuple, Iterable, Sequence, TypeVar
nxGraph = TypeVar('nxGraph', nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)


class CharPolyClass:
    def __init__(
        self,
        characteristic: Union[sp.Poly, str, sp.Matrix],
        k: sp.Symbol,
        z: sp.Symbol,
        E: sp.Symbol,
        param_dict: Optional[Dict[sp.Symbol, ArrayLike]] = {},
    ) -> None:
        
        self.k, self.z, self.E = k, z, E
        self.params = list(param_dict.keys())
        self.param_dict = {s: np.asarray(v) for s, v in param_dict.items()} # Ensure arrays

        # Check if all parameter values have the same shape
        batch_shapes = [v.shape for v in self.param_dict.values()]
        if not batch_shapes:
             self.batch_shape = ()
             self.num_samples = 1
        else:
            assert all(shape == batch_shapes[0] for shape in batch_shapes), \
                "Parameter values must have the same shape."
            self.batch_shape = batch_shapes[0]
            self.num_samples = int(np.prod(self.batch_shape))
        
        print(f"Hamiltonian Parameters: {self.params}; "
              f"batch shape {self.batch_shape}; "
              f"{self.num_samples} total instances.")

        # Initialize based on characteristic type
        if isinstance(characteristic, sp.Poly):
            self.ChP = characteristic
            self._init_from_ChP()
        elif isinstance(characteristic, str):
            param_maps = {k.name: k for k in self.params}
            expr = sp.sympify(characteristic, locals={'z': z, 'E': E, **param_maps})
            assert {E, z}.issubset(expr.free_symbols), (
                f"ChP string must include {E} AND {z} as free symbols"
            )
            self.ChP = sp.Poly(expr, z, 1/z, E)
            self._init_from_ChP()
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
                    f"Characteristic matrix must include {k} XOR {z} as a free symbol"
                )
            self._init_from_bloch()
        else:
            raise ValueError("Characteristic must be a sympy Poly, string, or Matrix.")

        # Prepare coefficients and companion matrix for z-polynomial
        self._prepare_poly_z()

    def _init_from_ChP(self) -> None:
        """Initialize attributes when starting from a Characteristic Polynomial."""
        k, z, E = self.k, self.z, self.E
        assert {E, z}.issubset(self.ChP.free_symbols), \
            "ChP must include E and z as free symbols"
        # Generators check, hoppings in both directions should exist
        assert set(self.ChP.gens) == {z, 1/z, E}, \
            f"ChP's generators must be {{z, 1/z, E}}, got {self.ChP.gens}"

        # Treat z as constant and E as variable to find num_bands and h_z
        Poly_E = sp.Poly(self.ChP.as_expr(), E)
        monic_Poly_E = Poly_E.monic()
        self.Poly_E_coeff = Poly_E.all_coeffs()
        self.num_bands = Poly_E.degree()

        # Derive Bloch Hamiltonian h_z (and h_k) from ChP
        if self.num_bands < 1:
            raise ValueError("Characteristic polynomial must be at least one-band.")
        if self.num_bands == 1:
            self.h_z = sp.Matrix([sp.expand(monic_Poly_E.TC())]) # Tailing coefficient
        else:
            self.h_z = sp.Matrix.companion(monic_Poly_E).applyfunc(sp.expand)

        self.h_k = hz2hk_1d(self.h_z, k, z)
        print(f"Derived Bloch Hamiltonian `h_z` with {self.num_bands} bands.")

    def _init_from_bloch(self) -> None:
        """Initialize attributes when starting from a Bloch Hamiltonian Matrix."""
        z, E = self.z, self.E
        # Calculate Characteristic polynomial from h_z
        # NOTE: charpoly() creates a new PurePoly object with dummy variables.
        Poly_E_pure = self.h_z.charpoly(E)
        self.Poly_E_coeff = Poly_E_pure.all_coeffs()
        self.num_bands = Poly_E_pure.degree()
        # Replace the dummy variable generated by charpoly with our predefined E
        Poly_E_expr = Poly_E_pure.as_expr().xreplace({Poly_E_pure.gens[0]: E})
        # Define the full ChP including z dependencies
        self.ChP = sp.Poly(Poly_E_expr, z, 1/z, E) # Define gens explicitly
        print(f"Derived Characteristic polynomial `ChP` with {self.num_bands} bands.")

    def _prepare_poly_z(self) -> None:
        """Prepare coefficients and companion matrix for the polynomial in z."""
        z = self.z
        # Treat E as constant and z as variable
        Poly_z_bigen = sp.Poly(self.ChP.as_expr(), z, 1/z)
        # Right hopping range
        self.poly_p = Poly_z_bigen.degree(1/z)
        # Left hopping range
        self.poly_q = Poly_z_bigen.degree(z)
        Poly_z = sp.Poly(sp.expand(self.ChP.as_expr() * z**self.poly_p), z)
        self.Poly_z_coeff = Poly_z.all_coeffs()
        # Companion matrix of P(E)(z) for efficient root finding
        monic_Poly_z = Poly_z.monic()
        self.companion_E = sp.Matrix.companion(monic_Poly_z).applyfunc(sp.expand)
        # Lambdify coefficients for numerical evaluation later
        self._lambdify_poly_z_coeffs()

    def _lambdify_poly_z_coeffs(self):
        """Create lambda functions for evaluating Poly_z coefficients."""
        self.Poly_z_coeff_funcs = []
        allowed_params_set = set(self.params)
        allowed_all_set = allowed_params_set | {self.E}

        # Prepare arguments for lambdify in the correct order
        args_order = self.params + [self.E] # E always comes last

        for i, expr in enumerate(self.Poly_z_coeff):
            syms = expr.free_symbols
            if not (syms <= allowed_all_set):
                raise ValueError(f"Coefficient {i} contains invalid symbols: {syms - allowed_all_set}")

            if not syms: # Constant
                func = lambda *args, val=complex(expr): np.full(args[-1].shape, val) # args[-1] is E_array
            elif syms == {self.E}: # Depends only on E
                func = sp.lambdify(self.E, expr, modules=["numpy"])
            elif syms <= allowed_params_set: # Depends only on params
                 # Need to broadcast param values to match E_array shape later
                 func = sp.lambdify(self.params, expr, modules=["numpy"])
            else: # Depends on E and params
                 func = sp.lambdify(args_order, expr, modules=["numpy"])

            self.Poly_z_coeff_funcs.append({'func': func, 'syms': syms})

    def real_space_H(
        self,
        N: int = 40,
        max_dim: int = 150,
        pbc: bool = False,
        param_dict: Optional[Dict[sp.Symbol, ArrayLike]] = None,
    ) -> np.ndarray:
        # Limit the size of the real space Hamiltonian to avoid numerical inaccuracies
        if self.num_bands * N > max_dim:
            N = max_dim // self.num_bands
        hop_dict = expand_hz_as_hop_dict_1d(self.h_z, self.z)
        if param_dict is None:
            param_dict = self.param_dict
        H = H_1D_batch_from_hop_dict(hop_dict, N, pbc, param_dict)
        return H

    def _spectral_boundaries(self, device='/CPU:0', pad_factor=0.05) -> None:
        finite_chain = self.real_space_H()
        E_arr = eigvals_batch(finite_chain, device, is_hermitian=False)
        
        re_min, re_max = np.amin(E_arr.real, axis=-1), np.amax(E_arr.real, axis=-1)
        im_min, im_max = np.amin(E_arr.imag, axis=-1), np.amax(E_arr.imag, axis=-1)
        
        re_center, re_radius = (re_max + re_min) / 2, (re_max - re_min) / 2
        im_center, im_radius = (im_max + im_min) / 2, (im_max - im_min) / 2
        
        radius = np.maximum(re_radius, im_radius) * (1 + pad_factor)
        
        self.spectral_center = np.stack([re_center, im_center], axis=-1)
        self.spectral_radius = radius
        self.spectral_square = np.stack([
            re_center - radius, re_center + radius, 
            im_center - radius, im_center + radius
        ], axis=-1)

    def _Poly_z_coeff_arr_from_E_arr(self, E_array: ArrayLike) -> np.ndarray:
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

    def Poly_z_roots_from_E_arr(
        self,
        E_array: ArrayLike,
        device: str = '/cpu:0'
    ) -> np.ndarray:
        coeff_arr = self._Poly_z_coeff_arr_from_E_arr(E_array)
        companion_arr = companion_batch(coeff_arr)
        with tf.device(device):
            companion_tensor = tf.convert_to_tensor(companion_arr)
            roots = tf.linalg.eigvals(companion_tensor)
        return roots.numpy()

    def spectral_potential_from_E_arr(
        self,
        E_array: ArrayLike,
        method: str = 'ronkin',
        device: str = '/cpu:0'
    ) -> np.ndarray:
        coeff_arr = self._Poly_z_coeff_arr_from_E_arr(E_array)
        roots = self.Poly_z_roots_from_E_arr(E_array, device=device)
        phi = spectral_potential(roots, coeff_arr, self.poly_q, method=method)
        return phi

    @staticmethod
    def _compute_masks(binary, dilation_radius=2):
        mask1 = np.where(binary)
        mask0 = np.where(~binary)
        dilated = dilation(binary, disk(dilation_radius))
        mask1_ = np.where(dilated)
        return mask1, mask0, mask1_
    
    @staticmethod
    def _compute_enhanced_threshold(ridge, ridge_block, mask1, mask0, resolution_enhancement):
        weights = np.array([
            ridge_block[mask1].size, 
            ridge[mask0].size * resolution_enhancement**2
        ])
        means = np.array([
            np.mean(ridge_block[mask1]),
            np.mean(ridge[mask0])
        ])
        threshold = np.dot(weights, means) / np.sum(weights)
        return threshold
    
    def _enhance_resolution(self, E_box, phi, ridge, binary, resolution, resolution_enhancement, method, device, DOS_filter_kwargs):
        enhanced_resolution = resolution * resolution_enhancement
        E_real_ = np.linspace(*E_box[:2], enhanced_resolution)
        E_imag_ = np.linspace(*E_box[2:], enhanced_resolution)
        E_split = E_real_ + 1j * E_imag_[:, None]
        
        mask1, mask0, mask1_ = self._compute_masks(binary)

        E_block = view_as_blocks(E_split, (resolution_enhancement, resolution_enhancement))
        masked_E_block = E_block[mask1_]

        split_kernel = np.ones((resolution_enhancement, resolution_enhancement))
        phi_ = np.kron(phi, split_kernel)
        phi_block = view_as_blocks(phi_, (resolution_enhancement, resolution_enhancement))
        phi_dense = self.spectral_potential_from_E_arr(masked_E_block, method=method, device=device)
        phi_block[mask1_] = phi_dense

        ridge_ = PosGoL(phi_, **DOS_filter_kwargs)
        ridge_block = view_as_blocks(ridge_, (resolution_enhancement, resolution_enhancement))
        ridge_block[mask0] = 0

        threshold = self._compute_enhanced_threshold(ridge, ridge_block, mask1, mask0, resolution_enhancement)
        binary_ = ridge_ > threshold
        binary_block = view_as_blocks(binary_, (resolution_enhancement, resolution_enhancement))
        binary_block[mask0] = 0

        return phi_, ridge_, binary_
    
    def spectral_images(
        self,
        resolution: int = 256,
        resolution_enhancement: int = 4,
        device: str = '/cpu:0',
        method: str = 'ronkin',
        DOS_filter_kwargs: Optional[dict] = {},
    ) -> tuple:
        E_box = self.spectral_square

        E_real = np.linspace(*E_box[:2], resolution)
        E_imag = np.linspace(*E_box[2:], resolution)
        E_arr = E_real + 1j * E_imag[:, None]

        phi = self.spectral_potential_from_E_arr(E_arr, method=method, device=device)
        if method == 'ronkin':
            ridge = PosGoL(phi, **DOS_filter_kwargs)
        else:
            ridge = phi

        binary = ridge > np.mean(ridge)
        
        result = None
        if resolution_enhancement <= 1 or resolution_enhancement is None:
            result = (phi, ridge, binary)
        else:
            # Apply resolution enhancement
            phi_, ridge_, binary_ = self._enhance_resolution(
                E_box, phi, ridge, binary, resolution, resolution_enhancement, 
                method, device, DOS_filter_kwargs
            )
            result = (phi_, ridge_, binary_)
        
        return result

    def _recover_energy_coordinates(
        self, 
        graph: nxGraph, 
        scale: float, 
        center_offset: np.ndarray, 
        magnify: float = 1.0
    ) -> nxGraph:
        if magnify <= 0:
            magnify = 1.0
            
        for node in graph.nodes(data=True):
            if 'pos' in node[1]:
                pos = np.asarray(node[1]['pos'], dtype=np.float32)
                # Recover the (x, y) coordinates from the 2D array indices
                new_pos = (pos[::-1] - center_offset) * scale + self.spectral_center
                node[1]['pos'] = new_pos * magnify
            if 'pts' in node[1]:
                pts = np.asarray(node[1]['pts'], dtype=np.float32)
                new_pts = (pts[..., ::-1] - center_offset) * scale + self.spectral_center
                node[1]['pts'] = new_pts * magnify

        for edge in graph.edges(data=True):
            if 'weight' in edge[2]:
                weight = np.asarray(edge[2]['weight'], dtype=np.float32)
                new_weight = weight * scale
                edge[2]['weight'] = new_weight * magnify
            if 'pts' in edge[2]:
                pts = np.asarray(edge[2]['pts'], dtype=np.float32)
                new_pts = (pts[..., ::-1] - center_offset) * scale + self.spectral_center
                edge[2]['pts'] = new_pts * magnify
                
        return graph

    def spectral_graph(
        self,
        resolution: int = 256,
        resolution_enhancement: int = 4,
        device: str = '/cpu:0',
        method: str = 'ronkin',
        short_edge_threshold: Optional[float] = 20,
        skeleton2graph_kwargs: Optional[dict] = {},
        DOS_filter_kwargs: Optional[dict] = {},
        magnify: float = 1.0,
    ) -> nxGraph:
        # Get spectral images (from cache if available)
        phi, ridge, binary = self.spectral_images(
            resolution=resolution,
            resolution_enhancement=resolution_enhancement,
            device=device,
            method=method,
            DOS_filter_kwargs=DOS_filter_kwargs,
        )
        
        # Obtain graph skeleton
        ske = skeletonize(binary, method='lee')
        
        # Construct skeleton graph
        graph = skeleton2graph(
            ske,
            Potential_image=phi.astype(np.float32),
            DOS_image=ridge.astype(np.float32),
            **skeleton2graph_kwargs
        )

        ### Post-process the extracted graph
        # Merge close nodes and short edges
        if short_edge_threshold is not None and short_edge_threshold > 0:
            graph = add_edges_within_threshold(graph, short_edge_threshold)
            graph = contract_close_nodes(graph, short_edge_threshold)

        # Calculate parameters for coordinate transformation
        final_res = resolution * resolution_enhancement
        scale = self.spectral_radius * 2 / final_res
        center_offset = np.array([final_res - 1, final_res - 1]) / 2  # offset for 0-based indexing

        # Process graph positions
        graph = self._recover_energy_coordinates(graph, scale, center_offset, magnify)

        return graph