import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

import numpy as np
from nurbs.core import NURBS

class NURBSPresenter:
    @staticmethod
    def render(nurbs: NURBS, samples: int = 100,
               ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        is_3d = nurbs.control_points.shape[1] > 2
        fig, ax = NURBSPresenter.prepare_canvas(ax, is_3d)
        curve = NURBSPresenter.sample_curve(nurbs, samples)

        if is_3d:
            NURBSPresenter.plot3d(curve, nurbs.control_points, ax)
        else:
            NURBSPresenter.plot2d(curve, nurbs.control_points, ax)

        return fig, ax

    @staticmethod
    def prepare_canvas(ax: Optional[plt.Axes], is_3d: bool) -> Tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d' if is_3d else None)
            return fig, ax
        return ax.figure, ax

    @staticmethod
    def sample_curve(nurbs: NURBS, samples: int) -> np.ndarray:
        domain = [nurbs.knots[nurbs.degree], nurbs.knots[-nurbs.degree-1]]
        return np.array([nurbs.evaluate(t) for t in np.linspace(*domain, samples)])

    @staticmethod
    def plot2d(curve: np.ndarray, ctrl: np.ndarray, ax: plt.Axes) -> None:
        ax.plot(curve[:, 0], curve[:, 1], 'steelblue',
                linewidth=2, label='NURBS Curve')
        ax.scatter(ctrl[:, 0], ctrl[:, 1], c='firebrick',
                   s=50, zorder=3, label='Control Points')
        ax.plot(ctrl[:, 0], ctrl[:, 1], '--',
                color='lightcoral', label='Control Polygon')
        ax.legend()

    @staticmethod
    def plot3d(curve: np.ndarray, ctrl: np.ndarray, ax: Axes3D) -> None:
        ax.plot(curve[:, 0], curve[:, 1],
                curve[:, 2], 'steelblue', linewidth=2)
        ax.scatter(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], c='firebrick', s=50)
        ax.plot(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], '--', color='lightcoral')
