import numpy as np
from dataclasses import dataclass


@dataclass
class NURBSSurface:
    control_points: np.ndarray  # (n, m, dim)
    weights: np.ndarray         # (n, m)
    knots_u: np.ndarray         # (p,)
    knots_v: np.ndarray         # (q,)
    degree_u: int
    degree_v: int

    def evaluate(self, u: float, v: float) -> np.ndarray:
        span_u = self._find_span(u, self.knots_u, self.degree_u)
        span_v = self._find_span(v, self.knots_v, self.degree_v)

        basis_u = self._basis_functions(u, span_u, self.knots_u, self.degree_u)
        basis_v = self._basis_functions(v, span_v, self.knots_v, self.degree_v)

        return self._compute_surface_point(basis_u, basis_v, span_u, span_v)

    @staticmethod
    def _find_span(t: float, knots: np.ndarray, degree: int) -> int:
        # Same algorithm as curve version but parameterized
        n = len(knots) - degree - 2
        if t >= knots[n + 1]:
            return n
        low, high = degree, n + 1
        while low < high - 1:
            mid = (low + high) // 2
            if t < knots[mid]:
                high = mid
            else:
                low = mid
        return low

    @staticmethod
    def _basis_functions(t: float, span: int, knots: np.ndarray, degree: int) -> np.ndarray:
        # Same as curve version but parameterized
        left = np.zeros(degree + 1)
        right = np.zeros(degree + 1)
        N = np.zeros(degree + 1)
        N[0] = 1.0

        for j in range(1, degree + 1):
            left[j] = t - knots[span + 1 - j]
            right[j] = knots[span + j] - t
            saved = 0.0

            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved
        return N

    def _compute_surface_point(self, basis_u: np.ndarray, basis_v: np.ndarray,
                               span_u: int, span_v: int) -> np.ndarray:
        point = np.zeros(self.control_points.shape[2])
        total_weight = 0.0

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                idx_u = span_u - self.degree_u + i
                idx_v = span_v - self.degree_v + j

                temp = basis_u[i] * basis_v[j] * self.weights[idx_u, idx_v]
                point += temp * self.control_points[idx_u, idx_v]
                total_weight += temp

        return point / total_weight
