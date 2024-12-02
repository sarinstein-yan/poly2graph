import tensorflow as tf

@tf.function
def poly_roots_tf_batch(c: tf.Tensor) -> tf.Tensor:
    """
    Calculate the roots of a monomial with coefficients `c`.
    The roots are the eigenvalues of the Frobenius companion matrix.

    Parameters
    ----------
    c : tf.Tensor, dtype=tf.complex64
        2-D tensor of polynomial coefficients ordered from low to high degree.
        The first axis is the batch axis.

    Returns
    -------
    roots : tf.Tensor, dtype=tf.complex64
        2-D tensor of roots of the polynomial.
    """
    n = c.shape[1]; batch_size = c.shape[0]
    if n < 2:
        return tf.constant([], dtype=c.dtype)
    if n == 2:
        return tf.constant(-c[:, 0] / c[:, 1], dtype=c.dtype)
    
    # Construct the Frobenius companion matrix
    lower_diagonal = tf.linalg.diag(tf.ones((batch_size, n - 2), dtype=c.dtype), k=-1)
    last_column = -c[:, :-1] / c[:, -1, None]
    last_column = tf.reshape(last_column, [batch_size, n - 1, 1])
    mat = tf.concat([lower_diagonal[..., :-1], last_column], axis=-1)
    mat = tf.reverse(mat, axis=[-2, -1]) # flip the matrix to reduce error
    
    # Calculate the eigenvalues of the companion matrix
    eigvals = tf.linalg.eigvals(mat)
    return eigvals

import torch

@torch.jit.script
def poly_roots_torch_batch(c: torch.Tensor) -> torch.Tensor:
    """
    Calculate the roots of a monomial with coefficients `c`.
    The roots are the eigenvalues of the Frobenius companion matrix.

    Parameters
    ----------
    c : torch.Tensor, dtype=torch.complex64
        2-D tensor of polynomial coefficients ordered from low to high degree.
        The first axis is the batch axis.

    Returns
    -------
    roots : torch.Tensor, dtype=torch.complex64
        2-D tensor of roots of the polynomial.
    """
    n = c.shape[1]
    batch_size = c.shape[0]
    if n < 2:
        return torch.empty(0, dtype=c.dtype, device=c.device)
    if n == 2:
        return -c[:, 0] / c[:, 1]
    
    # Construct the Frobenius companion matrix
    lower_diagonal = torch.diag_embed(torch.ones((batch_size, n - 2), dtype=torch.complex64, device=c.device), offset=-1)
    last_column = -c[:, :-1] / c[:, -1, None]
    last_column = last_column.view(batch_size, n - 1, 1)
    mat = torch.cat([lower_diagonal[..., :-1], last_column], dim=-1)
    mat = torch.flip(mat, dims=[-2, -1]) # flip the matrix to reduce error
    
    # Calculate the eigenvalues of the companion matrix
    eigvals = torch.linalg.eigvals(mat)
    return eigvals
