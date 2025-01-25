import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NURBS:
    control_points: np.ndarray  # (n, dim)
    weights: np.ndarray         # (n,)
    knots: np.ndarray           # (m,)
    degree: int

    def __post_init__(self):
        self._validate_inputs()
        self._cache: dict = {}

    def _validate_inputs(self) -> None:
        if len(self.control_points) != len(self.weights):
            raise ValueError("Mismatched control points/weights")
        if len(self.knots) != len(self.control_points) + self.degree + 1:
            raise ValueError("Invalid knot vector length")

    def evaluate(self, t: float) -> np.ndarray:
        span = self.find_span(t)
        basis = self.basis_functions(span, t)
        return self.calculate_point(basis, span)

    def find_span(self, t: float) -> int:
        knots = self.knots
        n = len(self.control_points) - 1
        if t >= knots[n + 1]:
            return n
        low, high = self.degree, n + 1

        while low < high - 1:
            mid = (low + high) // 2
            if t < knots[mid]:
                high = mid
            else:
                low = mid
        return low

    def basis_functions(self, span: int, t: float) -> np.ndarray:
        # Cox-de Boor formula: https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

        knots = self.knots
        degree = self.degree
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

    def calculate_point(self, basis: np.ndarray, span: int) -> np.ndarray:
        point = np.zeros(self.control_points.shape[1])
        weight = 0.0

        for i in range(self.degree + 1):
            idx = span - self.degree + i
            temp = basis[i] * self.weights[idx]
            point += temp * self.control_points[idx]
            weight += temp

        return point / weight
