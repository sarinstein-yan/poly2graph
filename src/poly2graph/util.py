import numpy as np
import tensorflow as tf

def companion_batch(a):
    """
    Create a (/an array of) companion matrix.
    Same as `scipy.linalg.companion`, copied here for adaptation.

    Create the companion matrix [1]_ associated with the polynomial whose
    coefficients are given in `a`.

    Parameters
    ----------
    a : (N,) array_like
        1-D array of polynomial coefficients. The length of `a` must be
        at least two, and ``a[0]`` must not be zero.

    Returns
    -------
    c : (N-1, N-1) ndarray
        The first row of `c` is ``-a[1:]/a[0]``, and the first
        sub-diagonal is all ones.  The data-type of the array is the same
        as the data-type of ``1.0*a[0]``.

    Raises
    ------
    ValueError
        If any of the following are true: a) ``a.ndim != 1``;
        b) ``a.size < 2``; c) ``a[0] == 0``.

    Examples
    --------
    >>> companion([1, -10, 31, -30])
    array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]])

    """
    a = np.atleast_1d(a)
    n = a.shape[-1]

    if n < 2:
        raise ValueError("The length of `a` along the last axis must be at least 2.")

    if np.any(a[..., 0] == 0):
        raise ValueError("The first coefficient(s) of `a` (i.e. elements "
                         "of `a[..., 0]`) must not be zero.")

    first_row = -a[..., 1:] / (1.0 * a[..., 0:1])
    c = np.zeros(a.shape[:-1] + (n - 1, n - 1), dtype=first_row.dtype)
    c[..., 0, :] = first_row
    c[..., np.arange(1, n - 1), np.arange(0, n - 2)] = 1
    return c

def kron_batch(a, b):
    """
    Compute the Kronecker product for each pair of matrices in batches using NumPy.
    
    Args:
        a: A numpy array of shape (..., N, N). N could be the lattice length.
        b: A numpy array of shape (..., m, m). m could be the no. of bands.
    
    Returns:
        A numpy array of shape (..., N*m, N*m) corresponding to the Kronecker product
        of the last two dimensions of a and b.
    """
    a = np.asarray(a); b = np.asarray(b)
    # Compute the batched outer product using einsum
    # The resulting tensor has shape (..., N, m, N, m)
    kron = np.einsum('...ij,...kl->...ikjl', a, b)
    # Determine the output shape: reshape (..., N, m, N, m) to (..., N*m, N*m)
    new_shape = a.shape[:-2] + (a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1])
    return kron.reshape(new_shape)

def eig_batch(array_of_matrices, device='/CPU:0', is_hermitian=False):
    """
    Compute the eigenvalues and eigenvectors for a batch of matrices using TensorFlow.

    This function computes the eigen decomposition for a batch of matrices provided as an array.
    It supports both Hermitian matrices (using tf.linalg.eigh) and general matrices (using tf.linalg.eig).
    For general matrices, it improves numerical stability by setting near-zero entries (below a tolerance)
    to zero before computing the eigenvalues and eigenvectors.

    Parameters
    ----------
    array_of_matrices : array-like
        An array or tensor of shape (..., N, N) representing a batch of square matrices.
    device : str or tf.device
        The TensorFlow device (e.g., '/GPU:0' or '/CPU:0') on which the computation is performed.
    is_hermitian : bool, optional
        Flag indicating whether the input matrices are Hermitian. If True, uses tf.linalg.eigh.
        Otherwise, uses tf.linalg.eig with a numerical stability threshold. Default is False.

    Returns
    -------
    eigvals_np : np.ndarray
        A numpy array of eigenvalues with shape matching the batch dimensions and an extra dimension 
        for eigenvalues.
    eigvecs_np : np.ndarray
        A numpy array of eigenvectors with shape matching the batch dimensions and two extra dimensions
        for the eigenvector matrices.

    Raises
    ------
    ValueError
        If the tensor's dtype is not one of [tf.float32, tf.float64, tf.complex64, tf.complex128].

    Notes
    -----
    - For non-Hermitian matrices, the tolerance for setting near-zero values is chosen based on the data type:
      1e-14 for complex dtypes (tf.complex64, tf.complex128) and 1e-6 for float dtypes (tf.float32, tf.float64).
    - The resulting eigenvalues and eigenvectors are converted to numpy arrays, and the computation is performed
      on the specified device.
    """
    with tf.device(device):
        array_of_matrices = tf.convert_to_tensor(array_of_matrices)

        if is_hermitian:
            vals, vecs = tf.linalg.eigh(array_of_matrices)
        
        else:
            # Set near-zero entries to zero for numerical stability.
            if array_of_matrices.dtype in [tf.complex64, tf.complex128]:
                tol = 1e-14
            elif array_of_matrices.dtype in [tf.float32, tf.float64]:
                tol = 1e-6
            else: raise ValueError("Unsupported dtype. dtype must be one of [tf.float32, "
                                      "tf.float64, tf.complex64, tf.complex128].")
            array_of_matrices = tf.where(tf.abs(array_of_matrices) < tol, 0., array_of_matrices)
            vals, vecs = tf.linalg.eig(array_of_matrices)
        
        # Convert to numpy array; data now on CPU.
        eigvals_np = vals.numpy(); eigvecs_np = vecs.numpy()
    
    # # Clear the TensorFlow session/graph state to release GPU memory.
    # tf.keras.backend.clear_session()

    return eigvals_np, eigvecs_np