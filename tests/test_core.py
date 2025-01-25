import numpy as np
from nurbs.nurbs import NURBS


class TestNURBS:
    def test_circle_approximation(self):
        # Quarter circle approximation
        ctrl = np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float64)
        weights = np.array([1, np.sqrt(2)/2, 1])
        knots = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

        nurbs = NURBS(ctrl, weights, knots, 2)
        point = nurbs.evaluate(0.5)

        assert np.allclose(point, [np.sqrt(2)/2, np.sqrt(2)/2], atol=1e-3)

    def test_linear_curve(self):
        ctrl = np.array([[0, 0], [1, 1]], dtype=np.float64)
        knots = np.array([0, 0, 1, 1], dtype=np.float64)
        nurbs = NURBS(ctrl, np.ones(2), knots, 1)

        assert np.allclose(nurbs.evaluate(0.5), [0.5, 0.5])
