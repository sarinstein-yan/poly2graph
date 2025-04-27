import numpy as np

import logging
from typing import Literal, Optional
DeviceStr = Optional[
    Literal["cpu", "gpu", "gpu:0", "gpu:1",
            "cuda", "cuda:0", "cuda:1", None]
]

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


def _parse_device(backend: str, device: DeviceStr) -> Optional[str]:
    """
    Convert a user-supplied device string into the object that the active
    backend expects, or return a sensible default.

    Parameters
    ----------
    backend : {'tf', 'torch', 'numpy'}
    device  : str | None
        Examples the user might pass:
            • '/GPU:0', '/CPU:0'      (TensorFlow)
            ^ NOTE: TensorFlow's `eig` doesn't have GPU version
            • 'cuda:0', 'cuda', 'cpu' (PyTorch)
            • None                    (any backend)

    Returns
    -------
    backend-specific device handle, or None (NumPy).
    """
    if backend == "tf":
        return device or "/CPU:0"            # TensorFlow accepts strings
    if backend == "torch":
        import torch
        if device is None:
            return torch.device("cpu")
        dev = device.lower()
        if "cuda" in dev or "gpu" in dev:
            # accept '/GPU:0' or 'cuda:0' etc.
            idx = "".join(ch for ch in dev if ch.isdigit())
            return torch.device(f"cuda:{idx}" if idx else "cuda")
        return torch.device("cpu")
    return None                              # NumPy

def eig_batch(array_of_matrices,
              device: DeviceStr = None,
              is_hermitian: bool = False,
              chop: bool = False):
    """
    Batched eigen-decomposition with automatic backend selection.

    The preferred order is:
        1. TensorFlow  (tf.linalg.eigh / tf.linalg.eig)
        2. PyTorch     (torch.linalg.eigh / torch.linalg.eig)
        3. NumPy       (np.linalg.eigh / np.linalg.eig)

    Parameters
    ----------
    array_of_matrices : array-like, shape (..., N, N)
        Real or complex square matrices.
    device : str | None, optional
        Device specifier (TF or Torch style).  Ignored for NumPy.
    is_hermitian : bool, optional
        If True, use *-eigh; else use *-eig with small-value thresholding.
    chop : bool, optional
        If True, small values are set to zero before eigen-decomposition.

    Returns
    -------
    eigvals_np : np.ndarray, shape (..., N)
    eigvecs_np : np.ndarray, shape (..., N, N)
    """
    # ------------------------------------------------------------------ #
    # 1) Try TensorFlow
    try:
        import tensorflow as tf
        backend = "tf"
    except Exception:                               # pragma: no cover
        backend = None

    # 2) Try PyTorch
    if backend is None:
        try:
            import torch
            backend = "torch"
        except Exception:                           # pragma: no cover
            backend = None

    # 3) Fallback: NumPy
    if backend is None:
        backend = "numpy"

    tol_float64 = 1e-14
    tol_float32 = 1e-7

    if backend == "tf":
        import tensorflow as tf

        dev_tf = _parse_device("tf", device)
        with tf.device(dev_tf):
            tensor = tf.convert_to_tensor(array_of_matrices)

            if is_hermitian:
                vals, vecs = tf.linalg.eigh(tensor)
            else:
                if chop:
                    tol = tol_float64 if tensor.dtype in [tf.float64,
                                                        tf.complex128] else tol_float32
                    tensor = tf.where(tf.abs(tensor) < tol, 0., tensor)
                vals, vecs = tf.linalg.eig(tensor)

        return vals.numpy(), vecs.numpy()

    elif backend == "torch":
        import torch
        logging.warning("TensorFlow not available; using PyTorch instead.")

        dev_torch = _parse_device("torch", device)
        tensor = torch.as_tensor(array_of_matrices).to(dev_torch)

        if is_hermitian:
            vals, vecs = torch.linalg.eigh(tensor)
        else:
            if chop:
                tol = tol_float64 if tensor.dtype in [torch.float64,
                                                    torch.complex128] else tol_float32
                tensor = torch.where(tensor.abs() < tol,
                                    torch.zeros_like(tensor), tensor)
            vals, vecs = torch.linalg.eig(tensor)

        return vals.cpu().numpy(), vecs.cpu().numpy()

    else:  # ---------- NumPy ------------------------------------------------- #
        logging.warning("TensorFlow and PyTorch not available; fallback to NumPy.")
        array = np.asarray(array_of_matrices)

        if is_hermitian:
            vals, vecs = np.linalg.eigh(array)
        else:
            if chop:
                tol = tol_float64 if array.dtype in [np.float64,
                                                    np.complex128] else tol_float32
                array = np.where(np.abs(array) < tol, 0.0, array)
            vals, vecs = np.linalg.eig(array)

        return vals, vecs
    
def eigvals_batch(array_of_matrices,
              device: DeviceStr = None,
              is_hermitian: bool = False,
              chop: bool = False):
    """
    Batched eigenvalues with automatic backend selection.

    The preferred order is:
        1. TensorFlow  (tf.linalg.eigh / tf.linalg.eig)
        2. PyTorch     (torch.linalg.eigh / torch.linalg.eig)
        3. NumPy       (np.linalg.eigh / np.linalg.eig)

    Parameters
    ----------
    array_of_matrices : array-like, shape (..., N, N)
        Real or complex square matrices.
    device : str | None, optional
        Device specifier (TF or Torch style).  Ignored for NumPy.
    is_hermitian : bool, optional
        If True, use *-eigh; else use *-eig with small-value thresholding.
    chop : bool, optional
        If True, small values are set to zero before eigen-decomposition.

    Returns
    -------
    eigvals_np : np.ndarray, shape (..., N)
    """
    vals, _ = eig_batch(array_of_matrices,
                        device=device,
                        is_hermitian=is_hermitian,
                        chop=chop)
    return vals