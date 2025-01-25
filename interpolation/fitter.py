import numpy as np
from nurbs.curve import NURBSCurve


class NURBSFitter:
    @staticmethod
    def interpolate(points: np.ndarray, degree: int = 3) -> NURBSCurve:
        n = len(points)
        params = NURBSFitter.chord_length_parameterization(points)
        knots = NURBSFitter.generate_knots(params, degree)
        return NURBSFitter.solve_constraints(points, params, knots, degree)

    @staticmethod
    def chord_length_parameterization(points: np.ndarray) -> np.ndarray:
        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total = np.sum(distances)
        if total == 0:
            return np.linspace(0, 1, len(points))
        cumsum = np.insert(np.cumsum(distances)/total, 0, 0)
        return cumsum

    @staticmethod
    def generate_knots(params: np.ndarray, degree: int) -> np.ndarray:
        n = len(params)
        internal = np.zeros(n - degree - 1)
        for i in range(len(internal)):
            start = i + 1
            end = start + degree
            internal[i] = np.mean(params[start:end])
        return np.concatenate([
            np.zeros(degree + 1),
            internal,
            np.ones(degree + 1)
        ])

    @staticmethod
    def solve_constraints(points: np.ndarray, params: np.ndarray,
                          knots: np.ndarray, degree: int) -> NURBSCurve:
        n = len(points)
        A = np.zeros((n, n))
        temp_nurbs = NURBSCurve(
            control_points=np.zeros((n, points.shape[1])),
            weights=np.ones(n),
            knots=knots,
            degree=degree
        )

        for i, t in enumerate(params):
            span = temp_nurbs.find_span(t)
            basis = temp_nurbs.basis_functions(span, t)
            for j in range(degree + 1):
                col = span - degree + j
                A[i, col] = basis[j]

        return NURBSCurve(
            control_points=np.linalg.lstsq(A, points, rcond=None)[0],
            weights=np.ones(n),
            knots=knots,
            degree=degree
        )
