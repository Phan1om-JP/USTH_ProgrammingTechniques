"""[DAY 4] Cholesky factorization - the only linear solver the course needs.

Never invert a matrix to solve a system. Factor once, then two triangular solves:
n^3/3 flops instead of n^3, and far better numerics.

The failure mode is the feature. `cholesky` succeeds if and only if the matrix is
positive definite, so a failure is information: geometrically the quadratic model has a
direction of non-positive curvature, statistically the Fisher information is singular
and the parameters are not identified from the data.
"""

import numpy as np

from ..errors import NotPositiveDefiniteError
from ..interfaces import ILinearSolver
from ..types import Mat, Vec


def cholesky(A: Mat) -> Mat:
    """The lower-triangular L with A = LL^T.

    Raises `NotPositiveDefiniteError` as soon as a pivot is non-positive. Vectorize the
    inner loop over i rather than writing three nested Python loops.
    """
    #raise NotImplementedError("[DAY 4] lab 1")
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    # pivot case
    for j in range (n):
        pivot = A[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= 0.0: # pivot
            raise NotPositiveDefiniteError(f"pivot {j} is {pivot:.3g}: matrix is not positive definite")
        L[j, j] = np.sqrt(pivot)
        # other cells
        L[j+1 :, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    
    return L


def solve_lower(L: Mat, b: Vec) -> Vec:
    """Forward substitution: solve L y = b for lower-triangular L."""
    raise NotImplementedError("[DAY 4] lab 1")


def solve_upper_from_lower(L: Mat, y: Vec) -> Vec:
    """Back substitution: solve L^T x = y, using L rather than forming its transpose."""
    raise NotImplementedError("[DAY 4] lab 1")


class CholeskySolver(ILinearSolver):
    """Solves A x = b for symmetric positive definite A, via A = LL^T.

    Written once on day 4, then injected unchanged into `NewtonDirection`,
    `GaussNewton` and `LevenbergMarquardt`. Three optimizers, one solver, no copies.
    """

    def solve(self, A: Mat, b: Vec) -> Vec:
        raise NotImplementedError("[DAY 4] lab 1")
