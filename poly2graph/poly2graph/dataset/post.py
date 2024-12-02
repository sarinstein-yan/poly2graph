import numpy as np

from numpy.typing import ArrayLike

def hash_labels(
    labels: ArrayLike,
    n: int,
    reindex: bool = False
) -> np.ndarray:
    '''
    Hash labels to integers in range(n**dim) using base n representation.
    I.e., treat each label list as a number in base n and convert to decimal.

    Parameters
    ----------
    labels : 2D Array of integers
        Labels to be hashed.
    n : int
        Base for hashing.
    reindex : bool, optional
        Whether to reindex the hash values to be precisely integers in
        range(n**dim). Default: False

    Returns
    -------
    np.ndarray
        Hashed labels.
    '''
    labels = np.asarray(labels)
    assert labels.ndim == 2, "labels must be 2D array"
    dim = labels.shape[1]
    base_vec = np.array([n**i for i in range(dim)])
    hash_value = base_vec @ labels.T
    if reindex:
        unique_hash = np.unique(hash_value)
        hash_map = {hash_val: i for i, hash_val in enumerate(unique_hash)}
        reassigned_hash_value = np.array([hash_map[val] for val in hash_value])
        return reassigned_hash_value
    else:
        return hash_value