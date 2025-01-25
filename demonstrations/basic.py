import numpy as np
import matplotlib.pyplot as plt
from nurbs.nurbs import NURBS
from visualization.plotter import NURBSPresenter

def run():
    ctrl = np.array([[0, 0], [2, 5], [5, 3], [7, 4], [9, 2]], dtype=np.float64)
    weights = np.array([1, 0.5, 1, 1.5, 1])
    knots = np.array([0, 0, 0, 0.3, 0.6, 1, 1, 1], dtype=np.float64)

    curve = NURBS(
        control_points=ctrl,
        weights=weights,
        knots=knots,
        degree=2
    )

    fig, ax = NURBSPresenter.render(curve)
    plt.show()


if __name__ == "__main__":
    run()
