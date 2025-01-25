from matplotlib import pyplot as plt
import numpy as np
from visualization.plotter import NURBSPresenter
from interpolation.fitter import NURBSFitter


def run():
    targets = np.array([
        [0, 0, 0],
        [1, 3, 2],
        [4, 2, 5],
        [6, 5, 3],
        [8, 1, 4],
        [9, 4, 2]  # Added more points for better conditioning
    ], dtype=np.float64)

    reconstructed = NURBSFitter.interpolate(targets, degree=3)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    NURBSPresenter.render(reconstructed, ax=ax)
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='lime', s=100,
               edgecolors='black', label='Target Points')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run()
