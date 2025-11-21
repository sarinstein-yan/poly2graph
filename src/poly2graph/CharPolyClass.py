import numpy as np
import sympy as sp
import networkx as nx

from skimage.morphology import skeletonize, dilation, binary_closing, disk
from skimage.util import view_as_blocks

from joblib import Parallel, delayed
from functools import partial
import warnings

from poly2graph.skeleton2graph import skeleton2graph, skeleton2graph_batch
from poly2graph.spectral_graph import (
    PosGoL,
    spectral_potential_batch,
    add_edges_within_threshold,
    contract_close_nodes
)
from poly2graph.hamiltonian import (
    hk2hz_1d, hz2hk_1d,
    expand_hz_as_hop_dict_1d,
    H_1D_batch_from_hop_dict
)
from poly2graph.util import companion_batch, eigvals_batch

from numpy.typing import ArrayLike
from typing import Union, Optional, Callable, Iterable, TypeVar, Set, Dict, Tuple, List, TypeVar
nxGraph = TypeVar('nxGraph', nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)


class CharPolyClass:
    """
    A class for handling and analyzing characteristic polynomials of Hamiltonians.
    
    This class provides methods to initialize, manipulate, and analyze characteristic
    polynomials for quantum systems, particularly for spectral graph theory applications.
    It supports initialization from different representations (polynomial, string, or matrix),
    computation of spectral potentials, and generation of spectral images and graphs.
    
    Parameters
    ----------
    characteristic : Union[sp.Poly, str, sp.Matrix]
        The characteristic polynomial representation, which can be a SymPy polynomial,
        a string expression, or a matrix.
    k : sp.Symbol
        The symbol representing wavenumber (momentum space).
    z : sp.Symbol
        The symbol representing the complex variable for analytic continuation.
    E : sp.Symbol
        The symbol representing energy.
    params : Optional[Set[sp.Symbol]], default={}
        A set of symbolic parameters used in the Hamiltonian.
        
    Attributes
    ----------
    k : sp.Symbol
        Symbol for wavenumber (momentum space).
    z : sp.Symbol
        Symbol for the complex variable.
    E : sp.Symbol
        Symbol for energy.
    params : List[sp.Symbol]
        Sorted list of symbolic parameters.
    ChP : sp.Poly
        The characteristic polynomial.
    h_z : sp.Matrix
        The Hamiltonian in terms of z.
    h_k : sp.Matrix
        The Hamiltonian in terms of k.
    hop_dict : Dict
        Dictionary of hopping terms.
    num_bands : int
        Number of bands in the system.
    poly_p : int
        Right hopping range.
    poly_q : int
        Left hopping range.
    companion_E : sp.Matrix
        Companion matrix for P(E)(z).
    """

    # --- Initialization Methods --- #
    
    def __init__(
        self,
        characteristic: Union[sp.Poly, str, sp.Matrix],
        k: sp.Symbol,
        z: sp.Symbol,
        E: sp.Symbol,
        params: Optional[Set[sp.Symbol]] = {},
    ) -> None:
        """
        Initialize the CharPolyClass.
        
        Parameters
        ----------
        characteristic : Union[sp.Poly, str, sp.Matrix]
            The characteristic polynomial representation. Can be:
            - A SymPy polynomial
            - A string expression
            - A matrix (Bloch Hamiltonian)
        k : sp.Symbol
            Symbol for wavenumber (momentum space).
        z : sp.Symbol
            Symbol for complex variable.
        E : sp.Symbol
            Symbol for energy.
        params : Optional[Set[sp.Symbol]], default={}
            Set of symbolic parameters used in the Hamiltonian.
        
        Returns
        -------
        None
        
        Notes
        -----
        The initialization process depends on the type of characteristic input:
        - For SymPy polynomial: Directly uses it as ChP and derives h_z and h_k
        - For string: Converts to SymPy polynomial then follows polynomial path
        - For matrix: Uses it as h_k or h_z and derives the other along with ChP
        """
        
        self.k, self.z, self.E = k, z, E
        self.params = sorted(params, key=lambda s: s.name)

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

        self.hop_dict = expand_hz_as_hop_dict_1d(self.h_z, self.z)
        # Prepare coefficients and companion matrix for P(E)(z)
        self._prepare_Poly_z()


    def _init_from_ChP(self) -> None:
        """Initialize attributes from a Characteristic Polynomial."""
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
            self.h_z = sp.Matrix([sp.expand(-monic_Poly_E.TC())]) # Tailing coefficient
        else:
            self.h_z = sp.Matrix.companion(monic_Poly_E).applyfunc(sp.expand)

        self.h_k = hz2hk_1d(self.h_z, k, z)
        print(f"[{self.__class__.__name__}] Derived Bloch Hamiltonian `h_z` with {self.num_bands} bands.")


    def _init_from_bloch(self) -> None:
        """Initialize attributes from a Bloch Hamiltonian Matrix."""
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
        print(f"[{self.__class__.__name__}] Derived Characteristic polynomial `ChP` with {self.num_bands} bands.")


    def _prepare_Poly_z(self) -> None:
        """Prepare coefficients and companion matrix for the polynomial in z."""
        z = self.z
        # Treat E as constant and z as variable
        Poly_z_bigen = sp.Poly(self.ChP.as_expr(), z, 1/z)
        # Right hopping range
        self.poly_p = Poly_z_bigen.degree(1/z)
        # Left hopping range
        self.poly_q = Poly_z_bigen.degree(z)
        
        Poly_z = sp.Poly(sp.expand(self.ChP.as_expr() * z**self.poly_p), z)
        raw_coeffs = Poly_z.all_coeffs()
        self.Poly_z_coeff = []
        for coeff in raw_coeffs:
            replace_map = {}
            for sym in coeff.free_symbols:
                # If name matches self.E but object identity is different, map it to self.E
                if sym.name == self.E.name and sym is not self.E:
                    replace_map[sym] = self.E                
                # (Optional) Ensure parameters are also consistent
                else:
                    for p in self.params:
                        if sym.name == p.name and sym is not p:
                            replace_map[sym] = p
                            break
            # Apply replacement if needed; xreplace is faster/safer than subs for atomic swaps
            if replace_map:
                self.Poly_z_coeff.append(coeff.xreplace(replace_map))
            else:
                self.Poly_z_coeff.append(coeff)

        # Companion matrix of P(E)(z) for efficient root finding
        Poly_z_monic_coeff = sp.nsimplify(Poly_z.monic().as_expr(), rational=True)
        self.companion_E = sp.Matrix.companion(sp.Poly(Poly_z_monic_coeff, z)).applyfunc(sp.expand)
        # Lambdify coefficients for numerical evaluation later
        self._lambdify_Poly_z_coeffs()


    def _lambdify_Poly_z_coeffs(self):
        """Create lambda functions for evaluating Poly_z coefficients."""
        self.Poly_z_coeff_funcs = []
        allowed_params_set = set(self.params)
        allowed_all_set = allowed_params_set | {self.E}

        for i, expr in enumerate(self.Poly_z_coeff):
            syms = expr.free_symbols
            if not (syms <= allowed_all_set):
                raise ValueError(f"Coefficient {i} contains invalid symbols: {syms - allowed_all_set}")

            if not syms: # Constant
                def eval_const(E_array, param_dict, expr=expr):
                    return np.full_like(E_array, complex(expr), dtype=np.complex128)
                func = eval_const
            
            elif syms == {self.E}: # Depends only on E
                f = sp.lambdify(self.E, expr, modules=["numpy"])
                def eval_E(E_array, param_dict, f=f):
                    return f(E_array)
                func = eval_E
            
            elif syms <= allowed_params_set: # Depends only on params
                f = sp.lambdify(self.params, expr, modules=["numpy"])
                def eval_params(E_array, param_dict, f=f):
                    param_vals = [param_dict[s] for s in self.params]
                    val = f(*param_vals)[..., None, None] # shape: (batch_shape, 1, 1)
                    return np.broadcast_to(val, E_array.shape)
                func = eval_params

            else:  # Depends on E and params
                poly = sp.Poly(expr, self.E)
                # cache {degree: lambdified function} for each monomial coefficient
                monomial_funcs = {}
                for (deg,), coeff_expr in poly.as_dict().items():
                    if coeff_expr.free_symbols:
                        monomial_funcs[deg] = sp.lambdify(self.params, coeff_expr, modules="numpy")
                    else:
                        monomial_funcs[deg] = complex(coeff_expr)

                def eval_params_and_E(E_array, param_dict, monomial_funcs=monomial_funcs):
                    result = np.zeros_like(E_array, dtype=np.complex128)
                    param_vals = [param_dict[s] for s in self.params]
                    for deg, f in monomial_funcs.items():
                        if callable(f):
                            p_val = f(*param_vals)[..., None, None] # shape: (batch_shape, 1, 1)
                        else:
                            p_val = f # already a constant number
                        result += p_val * (E_array ** deg) # shape: E_array.shape
                    return result
                func = eval_params_and_E

            self.Poly_z_coeff_funcs.append(func)


    # --- Parameter Preprocessing --- #

    def _process_params_dict(self, param_dict):
        """Process parameter dictionary to ensure consistent shapes and return batch information."""
        # NOTE: Disable assertions when generating from safe batches
        assert set(param_dict.keys()) == set(self.params), \
            f"param_dict keys {param_dict.keys()} must match params {self.params}."
        param_dict = {s: np.asarray(v) for s, v in param_dict.items()} # Ensure arrays
        # Check if all parameter values have the same shape
        batch_shapes = [v.shape for v in param_dict.values()]
        if not batch_shapes:
            batch_shape = ()
            num_samples = 1
        else:
            assert all(shape == batch_shapes[0] for shape in batch_shapes), \
                "Parameter values must have the same shape."
            batch_shape = batch_shapes[0]
            num_samples = int(np.prod(batch_shape))
        return param_dict, batch_shape, num_samples


    def _process_E_array_and_param_dict(self, E_array, param_dict):
        """Process energy array and parameter dictionary to ensure consistent shapes."""
        E_array = np.asarray(E_array)
        resolution = E_array.shape[-1]
        param_dict, batch_shape, num_samples = self._process_params_dict(param_dict)
        # NOTE: Disable assertion when generating from safe batches
        assert E_array.shape[:-2] == batch_shape, \
            f"Batch shape of `param_dict` {batch_shape} must match that of `E_array` {E_array.shape[:-2]}."
        return E_array, param_dict, batch_shape, num_samples, resolution


    # --- Polynomial Processing --- #

    def get_Poly_z_coeff_arr(
        self, 
        E_array: ArrayLike, 
        param_dict: Dict[sp.Symbol, ArrayLike]
    ) -> np.ndarray:
        """
        Get polynomial coefficients as arrays for a batch of energy values and parameters.
        
        Parameters
        ----------
        E_array : ArrayLike
            Array of energy values.
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
            
        Returns
        -------
        np.ndarray
            Array of polynomial coefficients.
        """
        E_array, param_dict, batch_shape, num_samples, resolution = \
            self._process_E_array_and_param_dict(E_array, param_dict)
        coeff_arr = self._get_Poly_z_coeff_arr(E_array, param_dict)
        return coeff_arr


    def _get_Poly_z_coeff_arr(self, E_array, param_dict):
        """Get polynomial coefficients as array for batch computation."""
        n_coeff = len(self.Poly_z_coeff_funcs)
        target_shape = E_array.shape + (n_coeff,)
        coeff_arr = np.empty(target_shape, dtype=np.complex128)

        for i, func in enumerate(self.Poly_z_coeff_funcs):
            coeff_arr[..., i] = func(E_array, param_dict)

        return coeff_arr


    def _get_Poly_z_coeff(self, E_array, param_vals):
        """
        Get polynomial coefficients for a single set of parameter values.
        `param_vals` should be sorted according to `self.params`.
        """
        n_coeff = len(self.Poly_z_coeff)
        target_shape = np.shape(E_array) + (n_coeff,)
        coeff_arr = np.zeros(target_shape, dtype=np.complex128)
        sub_mapping = {s: v for s, v in zip(self.params, param_vals)}

        for i, coeff_expr in enumerate(self.Poly_z_coeff):
            if not coeff_expr.free_symbols:
                coeff_arr[..., i] = complex(coeff_expr)
            else:
                coeff_expr = coeff_expr.subs(sub_mapping)
                coeff_dict = coeff_expr.as_poly(self.E).as_dict()
                for d, c in coeff_dict.items():
                    coeff_arr[..., i] += complex(c) * (E_array ** int(d[0]))
        return coeff_arr


    def get_Poly_z_roots(
        self,
        E_array: ArrayLike,
        param_dict: Dict[sp.Symbol, ArrayLike],
        device: str = '/CPU:0',
    ):
        """
        Compute the roots of P(E)(z) for a batch of energy values and parameters.
        
        Parameters
        ----------
        E_array : ArrayLike
            Array of energy values.
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
            
        Returns
        -------
        np.ndarray
            Array of polynomial roots.
        """
        coeff_arr = self.get_Poly_z_coeff_arr(E_array, param_dict)
        roots = self._get_Poly_z_roots_from_coeff_arr(coeff_arr, device=device)
        return roots


    # NOTE: computation bottleneck, O[(p+q)**4]. tf is faster than torch, jax, numpy
    def _get_Poly_z_roots_from_coeff_arr(
        self,
        coeff_arr: ArrayLike,
        device: str = '/CPU:0',
    ) -> np.ndarray:
        """Compute roots of polynomial from coefficient array using TensorFlow."""
        # for moderate hopping range, complex64 is sufficient
        coeff_arr = np.asarray(coeff_arr, dtype=np.complex64)
        companion_arr = companion_batch(coeff_arr)
        roots = eigvals_batch(companion_arr, 
                              device=device, is_hermitian=False)
        return roots


    # --- Spectral Boundaries --- #

    def real_space_H(
        self,
        param_dict: Dict[sp.Symbol, ArrayLike],
        N: int = 40,
        max_dim: int = 150,
        pbc: bool = False,
    ) -> np.ndarray:
        """
        Construct the real space Hamiltonian.
        
        Parameters
        ----------
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        N : int, default=40
            Size of the real space lattice.
        max_dim : int, default=150
            Maximum dimension for the Hamiltonian matrix. Limit the size 
            of the real space Hamiltonian to avoid numerical inaccuracies
        pbc : bool, default=False
            Whether to use periodic boundary conditions.
            
        Returns
        -------
        np.ndarray
            Real space Hamiltonian matrix.
        """
        if self.num_bands * N > max_dim:
            N = max_dim // self.num_bands
        param_dict = {s: np.asarray(v) for s, v in param_dict.items()}
        H = H_1D_batch_from_hop_dict(self.hop_dict, N, pbc, param_dict)
        return H

    def get_spectral_boundaries(
        self,
        param_dict: Dict[sp.Symbol, ArrayLike],
        device='/CPU:0',
        pad_factor=0.05,
    ) -> None:
        """
        Determine the boundaries of the spectral region for plotting.
        
        Parameters
        ----------
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
        pad_factor : float, default=0.05
            Factor by which to pad the spectral region.
            
        Returns
        -------
        tuple
            Tuple containing (spectral_square, spectral_center, spectral_radius).
            spectral_square is a 4-tuple (re_min, re_max, im_min, im_max).
        
        Notes
        -----
        Uses eigenvalues of a finite chain to determine spectral boundaries.
        """
        finite_chain = self.real_space_H(param_dict=param_dict)
        E_arr = eigvals_batch(finite_chain, 
                              device=device, is_hermitian=False, chop=True)
        
        re_min, re_max = np.amin(E_arr.real, axis=-1), np.amax(E_arr.real, axis=-1)
        im_min, im_max = np.amin(E_arr.imag, axis=-1), np.amax(E_arr.imag, axis=-1)
        
        re_center, re_radius = (re_max + re_min) / 2, (re_max - re_min) / 2
        im_center, im_radius = (im_max + im_min) / 2, (im_max - im_min) / 2
        
        radius = np.maximum(re_radius, im_radius) * (1 + pad_factor)
        
        spectral_center = np.stack([re_center, im_center], axis=-1)
        spectral_radius = radius
        spectral_square = np.stack([
            re_center - radius, re_center + radius, 
            im_center - radius, im_center + radius
        ], axis=-1)

        if np.any(radius == 0):
             warnings.warn("Warning: Zero `spectral_radius` detected. "
                    "This may indicate a degenerate spectrum.")

        return spectral_square, spectral_center, spectral_radius


    # --- Spectral Potential --- #

    def get_spectral_potential_batch(
        self,
        E_array: ArrayLike,
        param_dict: Dict[sp.Symbol, ArrayLike],
        device: str = '/CPU:0',
        method: str = 'ronkin',
    ) -> np.ndarray:
        """
        Compute spectral potential for a batch of energy values and parameters.
        
        Parameters
        ----------
        E_array : ArrayLike
            Array of energy values.
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
        method : str, default='ronkin'
            Method to use for computing the spectral potential.
            
        Returns
        -------
        np.ndarray
            Array of spectral potential values.
        """
        coeff_arr = self._get_Poly_z_coeff_arr(E_array, param_dict)
        roots = self._get_Poly_z_roots_from_coeff_arr(coeff_arr, device=device)
        phi = spectral_potential_batch(roots, coeff_arr, self.poly_q, method=method)
        return phi


    def get_spectral_potential(
        self,
        E_array: ArrayLike,
        param_vals: Iterable, # sorted according to self.params
        device: str = '/CPU:0',
        method: str = 'ronkin',
    ) -> np.ndarray:
        """
        Compute spectral potential for a single set of parameter values.
        
        Parameters
        ----------
        E_array : ArrayLike
            Array of energy values.
        param_vals : Iterable
            Parameter values, sorted according to self.params.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
        method : str, default='ronkin'
            Method to use for computing the spectral potential.
            
        Returns
        -------
        np.ndarray
            Array of spectral potential values.
        """
        coeff_arr = self._get_Poly_z_coeff(E_array, param_vals)
        roots = self._get_Poly_z_roots_from_coeff_arr(coeff_arr, device=device)
        phi = spectral_potential_batch(roots, coeff_arr, self.poly_q, method=method)
        return phi


    # --- Spectral Images: Potential, Ridges, and Skeleton Masks --- #

    def spectral_images(
        self,
        param_dict: Dict[sp.Symbol, ArrayLike],
        n_jobs: Union[Callable, int] = -1,
        device: str = '/CPU:0',
        resolution: int = 256,
        resolution_enhancement: int = 4,
        method: str = 'ronkin',
        DOS_filter_kwargs: Optional[dict] = {},
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spectral images for a batch of parameter values.
        
        Parameters
        ----------
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        n_jobs : Union[Callable, int], default=-1
            Number of parallel jobs or a custom batcher function.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
        resolution : int, default=256
            Resolution of the spectral images.
        resolution_enhancement : int, default=4
            Factor by which to enhance the resolution in regions of interest.
        method : str, default='ronkin'
            Method to use for computing the spectral potential.
        DOS_filter_kwargs : Optional[dict], default={}
            Keyword arguments for DOS filtering.
            
        Returns
        -------
        tuple
            Tuple of (phis, ridges, binaries, spectral_square) arrays containing
            the spectral images and the spectral boundaries box.
        """

        param_dict, batch_shape, num_samples = self._process_params_dict(param_dict)
        
        spectral_square, spectral_center, spectral_radius = \
            self.get_spectral_boundaries(param_dict=param_dict, device=device)
        
        batcher = Parallel(n_jobs=n_jobs, prefer='threads')

        phis, ridges, binaries, final_res = \
            self._spectral_images_flat(
                param_dict, spectral_square, num_samples,
                device, batcher, resolution, resolution_enhancement,
                method, DOS_filter_kwargs,
            )

        if len(batch_shape) > 1:
            phis = np.reshape(phis, batch_shape + (final_res, final_res))
            ridges = np.reshape(ridges, batch_shape + (final_res, final_res))
            binaries = np.reshape(binaries, batch_shape + (final_res, final_res))

        return phis, ridges, binaries, spectral_square


    def _spectral_images_flat(
        self,
        param_dict: Dict[sp.Symbol, Iterable],
        spectral_square: np.ndarray,
        num_samples: int,
        device: str = '/CPU:0',
        batcher_or_n_jobs: Union[Callable, int] = -1,
        resolution: int = 256,
        resolution_enhancement: int = 4,
        method: str = 'ronkin',
        DOS_filter_kwargs: Optional[dict] = {},
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectral images for flattened parameter values."""
        if isinstance(batcher_or_n_jobs, int):
            batcher = Parallel(n_jobs=batcher_or_n_jobs, prefer='threads')
        else:
            batcher = batcher_or_n_jobs
        
        E_arr = CharPolyClass.get_E_array(spectral_square, resolution)

        phi = self.get_spectral_potential_batch(E_arr, param_dict, device=device, method=method)
        phi_flat = phi.reshape(num_samples, resolution, resolution)

        if method == 'ronkin':
            ridge_list = batcher(
                delayed(partial(PosGoL, **DOS_filter_kwargs))(phi)
                for phi in phi_flat
            )
            ridge_flat = np.array(ridge_list)
        else:
            ridge_flat = phi_flat

        binary_flat = ridge_flat > np.mean(ridge_flat, axis=(-2, -1), keepdims=True)
        
        result = None
        if resolution_enhancement <= 1 or resolution_enhancement is None:
            result = (phi_flat, ridge_flat, binary_flat, resolution)
        else: # Apply resolution enhancement
            final_res = resolution * resolution_enhancement
            E_split = CharPolyClass.get_E_array(
                spectral_square.reshape(num_samples, 4), final_res
            )
            param_vals_flat = np.array(
                [param_dict[s].ravel() for s in self.params]
            ).T
            
            result = batcher(
                delayed(self._enhance_resolution)(
                    E_split[i], param_vals_flat[i],
                    phi_flat[i], ridge_flat[i], binary_flat[i], 
                    device, resolution_enhancement, method, DOS_filter_kwargs
                ) for i in range(num_samples)
            )
            phi_flat, ridge_flat, binary_flat = zip(*result)
            result = (phi_flat, ridge_flat, binary_flat, 
                      resolution*resolution_enhancement)
        
        return result


    def _enhance_resolution(self, E_split, param_vals, phi, ridge, binary, device, 
                        resolution_enhancement, method, DOS_filter_kwargs):
        """Enhance the resolution of spectral images in regions of interest."""
        mask1, mask0, mask1_ = CharPolyClass._get_masks(binary)

        E_block = view_as_blocks(E_split, (resolution_enhancement, resolution_enhancement))
        masked_E_block = E_block[mask1_]

        phi_split = self.get_spectral_potential(
            E_array=masked_E_block, param_vals=param_vals,
            device=device, method=method,
        )

        split_kernel = np.ones((resolution_enhancement, resolution_enhancement))
        phi_ = np.kron(phi, split_kernel)
        phi_block = view_as_blocks(phi_, (resolution_enhancement, resolution_enhancement))
        phi_block[mask1_] = phi_split

        ridge_ = PosGoL(phi_, **DOS_filter_kwargs)
        ridge_block = view_as_blocks(ridge_, (resolution_enhancement, resolution_enhancement))
        ridge_block[mask0] = 0

        threshold = self._get_enhanced_threshold(
            ridge, ridge_block, mask1, mask0, resolution_enhancement
        )
        binary_ = ridge_ > threshold
        binary_block = view_as_blocks(binary_, (resolution_enhancement, resolution_enhancement))
        binary_block[mask0] = 0

        return phi_, ridge_, binary_


    @staticmethod
    def _get_masks(binary, dilation_radius=2):
        """Get masks for resolution enhancement."""
        mask1 = np.where(binary)
        mask0 = np.where(~binary)
        dilated = dilation(binary, disk(dilation_radius))
        mask1_ = np.where(dilated)
        return mask1, mask0, mask1_


    @staticmethod
    def _get_enhanced_threshold(ridge, ridge_block, mask1, mask0, resolution_enhancement):
        """Compute threshold for enhanced resolution binary image."""
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


    # --- Spectral Graph Generation --- #

    def spectral_graph(
        self,
        param_dict: Dict[sp.Symbol, ArrayLike],
        n_jobs: int = -1,
        device: str = '/CPU:0',
        resolution: int = 256,
        resolution_enhancement: int = 4,
        method: str = 'ronkin',
        short_edge_threshold: Optional[float] = 20,
        skeleton2graph_kwargs: Optional[dict] = {},
        DOS_filter_kwargs: Optional[dict] = {},
        magnify: float = 1.0,
    ) -> Tuple[List[nxGraph], np.ndarray]:
        """
        Generate spectral graphs for a batch of parameter values.
        
        Parameters
        ----------
        param_dict : Dict[sp.Symbol, ArrayLike]
            Dictionary mapping parameter symbols to their values.
        n_jobs : int, default=-1
            Number of parallel jobs.
        device : str, default='/CPU:0'
            Device to use for TensorFlow computations.
        resolution : int, default=256
            Resolution of the spectral images.
        resolution_enhancement : int, default=4
            Factor by which to enhance the resolution in regions of interest.
        method : str, default='ronkin'
            Method to use for computing the spectral potential.
        short_edge_threshold : Optional[float], default=20
            Threshold for merging short edges in the graph.
        skeleton2graph_kwargs : Optional[dict], default={}
            Keyword arguments for skeleton2graph function.
        DOS_filter_kwargs : Optional[dict], default={}
            Keyword arguments for DOS filtering.
        magnify : float, default=1.0
            Factor by which to magnify the graph coordinates.
            
        Returns
        -------
        tuple
            Tuple of (graph_flat, param_dict_flat) where graph_flat is a list of
            spectral graphs and param_dict_flat is a dictionary mapping parameter
            symbols to their flattened values.
        """
        
        param_dict, batch_shape, num_samples = self._process_params_dict(param_dict)
        
        spectral_square, spectral_center, spectral_radius = \
            self.get_spectral_boundaries(param_dict=param_dict, device=device)
        radius_flat = spectral_radius.ravel()
        center_flat = spectral_center.reshape(num_samples, 2)
        
        batcher = Parallel(n_jobs=n_jobs, prefer='threads')

        # Get spectral images
        phi_flat, ridge_flat, binary_flat, final_res = \
            self._spectral_images_flat(
                param_dict, spectral_square, num_samples,
                device, batcher, resolution, resolution_enhancement,
                method, DOS_filter_kwargs,
            )

        graph_flat = batcher(
            delayed(CharPolyClass._get_skeleton_graph)(
                binary_flat[i], phi_flat[i], ridge_flat[i],
                skeleton2graph_kwargs, short_edge_threshold, 
                radius_flat[i], center_flat[i], final_res, magnify
            ) for i in range(num_samples)
        )

        param_dict_flat = {}
        for s in self.params:
            param_dict_flat[s] = param_dict[s].ravel()

        return graph_flat, param_dict_flat
    
    @staticmethod
    def _get_skeleton_graph(binary, phi, ridge, 
                        skeleton2graph_kwargs, short_edge_threshold, 
                        spectral_radius, spectral_center, final_res, magnify):
        """Create a graph from a binary skeleton image."""
        # Obtain graph skeleton
        ske = skeletonize(binary, method='lee')
        # Construct skeleton graph
        graph = skeleton2graph(
            ske,
            Potential_image=phi.astype(np.float32),
            DOS_image=ridge.astype(np.float32),
            **skeleton2graph_kwargs
        )
        graph = CharPolyClass._process_skeleton_graph(
            graph, short_edge_threshold,
            spectral_radius, spectral_center,
            final_res, magnify
        )
        return graph

    @staticmethod
    def _process_skeleton_graph(graph, short_edge_threshold, 
                            spectral_radius, spectral_center,
                            final_res, magnify):
        """Process a skeleton graph by merging close nodes and transforming coordinates."""
        # Merge close nodes and short edges
        if short_edge_threshold is not None and short_edge_threshold > 0:
            graph = add_edges_within_threshold(graph, short_edge_threshold)
            graph = contract_close_nodes(graph, short_edge_threshold)

        # Calculate parameters for coordinate transformation
        scale = spectral_radius * 2 / final_res
        center_offset = np.array([final_res - 1, final_res - 1]) / 2

        # Process graph positions
        graph = CharPolyClass._recover_energy_coordinates(
            graph, spectral_center, scale, center_offset, final_res, magnify
        )
        return graph


    @staticmethod
    def _recover_energy_coordinates(
        graph: nxGraph, 
        spectral_center: np.ndarray,
        scale: float, 
        center_offset: np.ndarray, 
        final_res: int,
        magnify: float = 1.0
    ) -> nxGraph:
        """Transform graph coordinates from pixel space to energy space."""
        
        if magnify <= 0:
            magnify = 1.0
            
        for node in graph.nodes(data=True):
            if 'pos' in node[1]:
                pos = node[1]['pos']
                pos = np.asarray([pos[1], final_res-pos[0]], dtype=np.float32)
                # Recover the (x, y) coordinates from the 2D array indices
                new_pos = (pos - center_offset) * scale + spectral_center
                node[1]['pos'] = new_pos * magnify
            if 'pts' in node[1]:
                pts = node[1]['pts']
                pts = pts[:, ::-1]
                pts[:, 1] = final_res - pts[:, 1]
                pts = np.asarray(pts, dtype=np.float32)
                new_pts = (pts - center_offset) * scale + spectral_center
                node[1]['pts'] = new_pts * magnify

        for edge in graph.edges(data=True):
            if 'weight' in edge[2]:
                weight = np.asarray(edge[2]['weight'], dtype=np.float32)
                new_weight = weight * scale
                edge[2]['weight'] = new_weight * magnify
            if 'pts' in edge[2]:
                pts = edge[2]['pts']
                pts = pts[:, ::-1]
                pts[:, 1] = final_res - pts[:, 1]
                pts = np.asarray(pts, dtype=np.float32)
                new_pts = (pts - center_offset) * scale + spectral_center
                edge[2]['pts'] = new_pts * magnify

        return graph


    # --- Helpers --- #

    @staticmethod
    def get_E_array(spectral_square, resolution):
        """
        Generate an array of complex energy values within a spectral square.
        
        Parameters
        ----------
        spectral_square : np.ndarray
            Array of (re_min, re_max, im_min, im_max) defining the spectral region.
        resolution : int
            Resolution of the energy grid.
            
        Returns
        -------
        np.ndarray
            Array of complex energy values.
        """
        spectral_square = np.asarray(spectral_square)
        E_real = np.linspace(spectral_square[..., 0], spectral_square[..., 1], 
                             resolution, axis=-1)
        E_imag = np.linspace(spectral_square[..., 2], spectral_square[..., 3], 
                             resolution, axis=-1)
        return E_real[..., None, :] + 1j * E_imag[..., :, None]