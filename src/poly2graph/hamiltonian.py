import numpy as np
import sympy as sp
from poly2graph.util import kron_batch

def shift_matrix(N, shift=1, pbc=True):
    """
    Constructs the translation (shift) matrix for a 1D chain of length N.

    For a given shift value, the function creates a matrix that translates (or "shifts") 
    the basis elements. For periodic boundary conditions (pbc=True), the matrix is rolled; 
    otherwise, a diagonal matrix with shifted ones is returned.

    Parameters:
        N (int): The number of sites in the 1D chain.
        shift (int, optional): The number of sites to shift (default is 1).
        pbc (bool, optional): Whether to enforce periodic boundary conditions (default is True).

    Returns:
        A NumPy array representing the shift matrix. For N <= 1, returns a 1x1 identity matrix.
    """
    if N is None or N <= 1:
        return np.eye(1)
    if pbc:
        return np.roll(np.eye(N), shift, axis=-1)
    else:
        elem = np.ones(N - abs(shift))
        return np.diag(elem, k=shift)

# --- for 1D Bloch Hamiltonian --- #
def _apply_func(obj, func):
    """
    Applies a function 'func' to each element if 'obj' is a sympy Matrix,
    or directly to 'obj' if it is a sympy expression.
    
    Parameters:
        obj: sympy Matrix or expression.
        func: A function to apply.
        
    Returns:
        The transformed sympy Matrix or expression.
    """
    if isinstance(obj, sp.Matrix):
        return obj.applyfunc(func)
    return func(obj)

def hk2hz_1d(h_k, k, z):
    """
    Converts a 1D Bloch Hamiltonian from momentum-space (k) to a representation 
    in terms of a complex variable (z) via the logarithmic substitution k = -i log(z).
    
    Parameters:
        h_k: A sympy Matrix or expression representing the Bloch Hamiltonian.
        k: sympy symbol corresponding to the momentum.
        z: sympy symbol representing the complex variable.
    
    Returns:
        A simplified sympy expression or Matrix for the Hamiltonian expressed in 
        terms of z.
    """
    subs_dict = {k: sp.log(z) / sp.I}
    return _apply_func(h_k, 
        lambda expr: sp.simplify(expr.subs(subs_dict).rewrite(sp.exp))
    )

def hz2hk_1d(h_z, k, z):
    """
    Converts a Hamiltonian expressed in terms of a complex variable (z) back to 
    momentum-space (k) using the inverse of the logarithmic substitution, and 
    rewriting the expression using trigonometric functions.
    
    Parameters:
        h_z: A sympy Matrix or expression representing the Hamiltonian in terms of z.
        k: sympy symbol corresponding to the momentum.
        z: sympy symbol corresponding to the complex variable.
    
    Returns:
        A sympy expression or Matrix for the Hamiltonian expressed in terms of k.
    """
    # subs_dict = {z: sp.cos(k) + sp.I * sp.sin(k)}
    subs_dict = {z: sp.exp(sp.I*k)}
    return _apply_func(h_z,
        lambda expr: sp.expand(expr.subs(subs_dict))
    )

def expand_hz_as_hop_dict_1d(h_z, z):
    """
    Expands a 1D Hamiltonian expressed in terms of the complex variable (z) into its 
    polynomial form. Each matrix element is parsed to extract coefficients corresponding 
    to different powers of z, and the results are stored in a dictionary.
    
    Parameters:
        h_z: A sympy Matrix or expression representing the Hamiltonian in terms of z.
        z: sympy symbol corresponding to the complex variable (typically exp(i*k)).
    
    Returns:
        A dictionary where:
            - Keys are integers representing the exponent of z.
            - Values are sympy Matrices with the corresponding coefficients.
    """
    # Ensure h_z is a matrix.
    if not isinstance(h_z, sp.Matrix):
        h_z = sp.Matrix(h_z)

    d_rows, d_cols = h_z.shape
    poly_dict = {}

    for i in range(d_rows):
        for j in range(d_cols):
            expr = sp.expand(h_z[i, j])
            terms = expr.as_ordered_terms()
            for term in terms:
                coeff, factors = term.as_coeff_mul()
                prod = sp.Mul(*factors)
                pdict = prod.as_powers_dict()
                exp_z = pdict.get(z, 0)
                remainder = sp.simplify(prod / (z**exp_z))
                term_coeff = coeff * remainder
                key = int(exp_z)
                if key not in poly_dict:
                    poly_dict[key] = sp.zeros(d_rows, d_cols)
                poly_dict[key][i, j] += term_coeff

    return poly_dict

def _compute_hop_arr(val, param_dict, batch_shape):
    """
    Computes the batched hopping array for a given hopping term (a sympy Matrix).
    
    If the hopping matrix has no free symbols, it tiles the numerical array.
    Otherwise, it uses sympy.lambdify to evaluate the matrix element-wise.
    """
    if len(val.free_symbols) == 0:
        hop = np.array(val.tolist(), dtype=np.complex128)
        return np.tile(hop, (*batch_shape, 1, 1))
    else:
        hop_arr = np.zeros((*batch_shape, *val.shape), dtype=np.complex128)
        # Sort free symbols to ensure a consistent order with param_dict keys.
        f_vars = sorted(val.free_symbols, key=lambda s: s.name)
        for idx in np.ndindex(val.shape):
            f = sp.lambdify(f_vars, val[idx], modules='numpy')
            args = [param_dict[s] for s in f_vars]
            hop_arr[..., *idx] = f(*args)
        return hop_arr

def H_1D_batch_from_hop_dict(hop_dict, N, pbc=False, param_dict={}):
    """
    Constructs a (possibly batched) Hamiltonian matrix from a dictionary of hopping terms.
    
    Parameters:
        hop_dict (dict): Dictionary where keys are integers representing the exponent of z
                         and values are sympy Matrices with the corresponding hopping coefficients.
        N (int): Number of sites in the 1D chain.
        pbc (bool): Whether to enforce periodic boundary conditions.
        param_dict (dict): Dictionary mapping sympy symbols to numerical values (or arrays) 
                           for batch processing. All parameter arrays must share the same shape.
    
    Returns:
        A sympy Matrix (or a NumPy array in the batched case) representing the Hamiltonian 
        for the 1D chain.
    """
    # Ensure that all parameter arrays have the same shape.
    batch_shapes = [np.shape(v) for v in param_dict.values()]
    assert all(shape == batch_shapes[0] for shape in batch_shapes), "Parameter values must have the same shape."
    batch_shape = batch_shapes[0] if batch_shapes else ()
    
    H = None
    for shift, val in hop_dict.items():
        # Build the spatial shift matrix (using the proper boundary condition).
        T = shift_matrix(N, shift, pbc).astype(np.complex128)
        T_arr = np.tile(T, (*batch_shape, 1, 1))
        
        # Compute the hopping array (internal degree of freedom) for this term.
        hop_arr = _compute_hop_arr(val, param_dict, batch_shape)
        
        # Combine via the batched Kronecker product.
        H_temp = kron_batch(T_arr, hop_arr)
        H = H_temp if H is None else H + H_temp
    return H

def H_1D_batch_from_hz(h_z, z, N, param_dict):
    """
    Constructs the 1D Hamiltonian matrices for both open (OBC) and periodic (PBC) boundary conditions 
    from a Bloch Hamiltonian expressed in momentum space in terms of z = exp(i*k).

    Parameters
    ----------
    h_z : sympy expression or sympy Matrix
        The Bloch Hamiltonian expressed in momentum space in terms of z.
    z : sympy symbol
        The complex symbol representing the phase factor.
    N : int
        Number of lattice sites in the 1D system.
    param_dict : dict
        Dictionary mapping sympy symbols to numerical values (or arrays) for batch processing.
        All parameter arrays must share the same shape.
    
    Returns
    -------
    Hobc_arr : np.ndarray
        Hamiltonian matrix under open boundary conditions (OBC) with shape (*batch_shape, N*d, N*d),
        where d is the dimension of the internal degree of freedom.
    Hpbc_arr : np.ndarray
        Hamiltonian matrix under periodic boundary conditions (PBC) with the same shape as Hobc_arr.
    """
    # Expand the Bloch Hamiltonian into a hopping dictionary.
    hoppings = expand_hz_as_hop_dict_1d(h_z, z)
    
    # Reuse the base Hamiltonian constructor for both boundary conditions.
    Hobc_arr = H_1D_batch_from_hop_dict(hoppings, N, pbc=False, param_dict=param_dict)
    Hpbc_arr = H_1D_batch_from_hop_dict(hoppings, N, pbc=True, param_dict=param_dict)
    
    return Hobc_arr, Hpbc_arr

def H_1D_batch(h_k, k, N, param_dict):
    """
    Constructs the 1D Hamiltonian matrices for both open (OBC) and periodic (PBC) boundary conditions 
    from a Bloch Hamiltonian expressed in momentum space as a function of momentum k.

    Parameters
    ----------
    h_k : sympy expression or sympy Matrix
        The Bloch Hamiltonian in momentum space as a function of k.
    k : sympy symbol
        The momentum symbol used in the Bloch Hamiltonian.
    N : int
        The number of lattice sites in the 1D system.
    param_dict : dict
        Dictionary mapping sympy symbols to numerical values (or arrays) for batch processing.
        All parameter arrays must share the same shape.
    
    Returns
    -------
    Hobc_arr : np.ndarray
        Hamiltonian under open boundary conditions.
    Hpbc_arr : np.ndarray
        Hamiltonian under periodic boundary conditions.
    """
    # Define a dummy symbol for the phase factor z and convert h_k to h_z.
    z = sp.Symbol('z')
    h_z = hk2hz_1d(h_k, k, z)
    return H_1D_batch_from_hz(h_z, z, N, param_dict)


# --- for general 1D-3D Bloch Hamiltonian --- #
def hk2hz(h_k, kx, ky, kz, zx=None, zy=None, zz=None):
    """
    Converts a Bloch Hamiltonian from momentum-space (kx, ky, kz) to a representation 
    in terms of complex variables (zx, zy, zz) via the logarithmic substitution.
    
    For each momentum component, if the corresponding complex variable is provided and 
    the momentum symbol appears in the Hamiltonian, the substitution:
    
        k = log(z) / i
    
    is performed.
    
    Parameters:
        h_k : sympy Matrix or expression
            The Bloch Hamiltonian in momentum-space.
        kx, ky, kz : sympy symbols
            Symbols corresponding to the momentum components.
        zx, zy, zz : sympy symbols or None, optional
            Complex variables to be used in the substitution (e.g. zx = exp(i*kx)).
            If any is None or the corresponding momentum symbol is not present in h_k,
            no substitution is performed for that component.
            
    Returns:
        A simplified sympy expression or Matrix for the Hamiltonian expressed in 
        terms of zx, zy, zz.
    """
    # Collect free symbols from h_k.
    if isinstance(h_k, sp.Matrix):
        free_syms = set().union(*[expr.free_symbols for expr in h_k])
    else:
        free_syms = h_k.free_symbols

    # Build substitution dictionary.
    subs_dict = {}
    for momentum, z in [(kx, zx), (ky, zy), (kz, zz)]:
        if momentum in free_syms and z is not None:
            subs_dict[momentum] = sp.log(z) / sp.I

    # Apply the substitution and then rewrite in terms of exponentials.
    return _apply_func(h_k,
        lambda expr: sp.simplify(expr.subs(subs_dict).rewrite(sp.exp))
    )

def hz2hk(h_z, kx, ky, kz, zx=None, zy=None, zz=None):
    """
    Converts a Hamiltonian expressed in terms of complex variables (zx, zy, zz)
    back to momentum-space (kx, ky, kz) using the inverse of the logarithmic substitution,
    rewritten in terms of trigonometric functions.
    
    For every provided complex variable (if present in h_z), the substitution
        z = cos(k) + i*sin(k)
    is performed with the corresponding momentum variable.
    
    Parameters:
        h_z : sympy Matrix or expression
            The Hamiltonian expressed in terms of the complex variables.
        kx, ky, kz : sympy symbols
            Momentum space variables.
        zx, zy, zz : sympy symbols or None, optional
            Complex variable symbols corresponding to kx, ky, kz respectively.
            
    Returns:
        A sympy expression or Matrix for the Hamiltonian expressed in momentum-space.
    """
    # Collect free symbols from h_z.
    if isinstance(h_z, sp.Matrix):
        free_syms = set().union(*[expr.free_symbols for expr in h_z])
    else:
        free_syms = h_z.free_symbols

    # Build substitution dictionary for each provided z variable.
    subs_dict = {}
    for momentum, z in [(kx, zx), (ky, zy), (kz, zz)]:
        if z is not None and z in free_syms:
            # Substituting z with cos(k) + i*sin(k) which is equivalent to exp(i*k).
            subs_dict[z] = sp.exp(sp.I * momentum)
            
    return _apply_func(h_z,
        lambda expr: sp.expand(expr.subs(subs_dict))
    )

def expand_hz_as_hop_dict(h_z, zx=None, zy=None, zz=None):
    """
    Expands a Hamiltonian expressed in terms of complex variables (zx, zy, zz) into its 
    polynomial form.
    
    Each element of the Hamiltonian is expanded and parsed so that the coefficients for 
    all combinations of powers of the provided complex variables are extracted.
    The result is returned as a dictionary mapping the power tuple (ordered as (zx, zy, zz) 
    for those provided) to the corresponding coefficient matrix.
    
    Parameters:
        h_z : sympy Matrix or expression
            The Hamiltonian in terms of complex variables.
        zx, zy, zz : sympy symbols or None, optional
            Complex variables (typically exp(i*kx), etc.) to be expanded. Only variables 
            that are provided (non-None) are considered.
            
    Returns:
        A dictionary where:
          - Keys are tuples of integers representing the exponents of each provided z variable.
          - Values are sympy Matrices (of the same shape as h_z) whose (i, j) entry is the coefficient 
            of the corresponding monomial in the expansion of the (i, j) entry of h_z.
    """
    # Ensure the Hamiltonian is a sympy Matrix.
    if not isinstance(h_z, sp.Matrix):
        h_z = sp.Matrix(h_z)
    
    d_rows, d_cols = h_z.shape
    poly_dict = {}
    
    # Create an ordered list of the provided complex variable symbols.
    z_vars = [z for z in (zx, zy, zz) if z is not None]

    # Loop over every matrix element.
    for i in range(d_rows):
        for j in range(d_cols):
            expr = sp.expand(h_z[i, j])
            # Decompose the expanded expression term-by-term.
            for term in expr.as_ordered_terms():
                coeff, factors = term.as_coeff_mul()
                prod = sp.Mul(*factors)
                pdict = prod.as_powers_dict()
                exponents = []
                divisor = 1
                for z in z_vars:
                    exp_val = pdict.get(z, 0)
                    exponents.append(int(exp_val))
                    divisor *= z**exp_val
                remainder = sp.simplify(prod / divisor)
                term_coeff = coeff * remainder
                key = tuple(exponents)
                if key not in poly_dict:
                    poly_dict[key] = sp.zeros(d_rows, d_cols)
                poly_dict[key][i, j] += term_coeff
    return poly_dict