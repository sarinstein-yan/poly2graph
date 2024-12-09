"""
Real space Hamiltonian matrix construction and diagonalization for 1D chains.
"""

import numpy as np

from numpy.typing import ArrayLike
from typing import Optional, Sequence

def poly_to_H_1band(
    c: ArrayLike,
    N: int,
    pbc: Optional[bool] = False
) -> np.ndarray:
    '''
    Convert a symmetric characteristic polynomial to a real space Hamiltonian matrix for a 1D chain.
    
    Parameters:
    -----------
    c: array_like
        Symmetric coefficients of the polynomial. len(c) should be odd,
        with the middle one being the z^0 coefficient.
    N: int
        Number of unit cells.
    pbc: bool, optional
        If True, use periodic boundary conditions. Default is False,
        i.e., open boundary conditions.

    Returns:
    --------
    H: ndarray
        Hamiltonian matrix.
    '''
    
    c = np.asarray(c)
    # Ensure the coefficients list is symmetric
    if len(c) % 2 == 0:
        raise ValueError("The length of coefficients 'c' must be odd."
                        " The middle coefficient is the z^0 term's.")
    
    mid_idx = len(c) // 2  # Middle index for the z^0 term
    
    # Create the Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.float64)
    
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

    return H
    
def real_space_spectra_1band(
    c: ArrayLike,
    N: int,
    pbc: Optional[bool] = False
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calculate the energy spectrum of the real space Hamiltonian for a given symmetric characteristic polynomial.

    Parameters:
    -----------
    c: array_like
        Symmetric coefficients of the polynomial. len(c) should be odd,
        with the middle one being the z^0 coefficient.
    N: int
        Number of unit cells.
    pbc: bool, optional
        If True, use periodic boundary conditions. Default is False,
        i.e., open boundary conditions.

    Returns:
    --------
    (E_Re, E_Im): tuple
        Real and imaginary parts of the energy spectrum.
    '''

    H = poly_to_H_1band(c, N, pbc)
    E = np.linalg.eigvals(H)

    return E.real, E.imag