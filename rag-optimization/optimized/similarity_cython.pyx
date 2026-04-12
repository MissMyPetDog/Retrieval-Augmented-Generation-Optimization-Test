# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated cosine similarity — Week 5 optimization.

Compile with:
    python setup_cython.py build_ext --inplace

Key techniques demonstrated:
  - Typed memoryviews for zero-copy NumPy access
  - Static type declarations (cdef)
  - Disabled bounds checking & wraparound
  - C-level math (libc.math)
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


def cosine_sim_cython(
    cnp.ndarray[cnp.float32_t, ndim=1] query_vec,
    cnp.ndarray[cnp.float32_t, ndim=2] corpus_matrix,
):
    """
    Cython cosine similarity.
    Same interface as cosine_sim_python / cosine_sim_numpy.
    """
    cdef:
        Py_ssize_t n = corpus_matrix.shape[0]
        Py_ssize_t d = corpus_matrix.shape[1]
        Py_ssize_t i, j
        double dot, q_norm, c_norm, val
        cnp.ndarray[cnp.float64_t, ndim=1] scores = np.empty(n, dtype=np.float64)

        # Typed memoryviews for fast element access
        float[:] q_view = query_vec
        float[:, :] c_view = corpus_matrix

    # Precompute query norm
    q_norm = 0.0
    for j in range(d):
        q_norm += q_view[j] * q_view[j]
    q_norm = sqrt(q_norm)

    if q_norm == 0.0:
        return np.zeros(n, dtype=np.float64)

    for i in range(n):
        dot = 0.0
        c_norm = 0.0
        for j in range(d):
            val = c_view[i, j]
            dot += q_view[j] * val
            c_norm += val * val
        c_norm = sqrt(c_norm)

        if c_norm == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = dot / (q_norm * c_norm)

    return scores
