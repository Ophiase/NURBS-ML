import numpy as np
import matplotlib.pyplot as plt
from interpolation.fitter import NURBSFitter
from nurbs.synthetic import SyntheticCurveGenerator
from nurbs.curve import NURBSCurve
from visualization.plotter import NURBSPresenter

ENABLE_FITTING = False


def generate_and_render(
        ax, curve_type: str,
        generator: SyntheticCurveGenerator,
        enable_fitting: bool = ENABLE_FITTING):
    points = generator.generate(curve_type)

    if ENABLE_FITTING:
        fitted_curve = NURBSFitter.interpolate(points, degree=3)
        NURBSPresenter.render(fitted_curve, ax=ax)
    else:
        fitted_curve = None

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', label='Point Cloud')
    ax.set_title(f"{curve_type.capitalize()} Curve")
    ax.legend()
    return fitted_curve


def run(
        config: dict,
        enable_fitting: bool = ENABLE_FITTING):
    generator = SyntheticCurveGenerator(config["synthetic"])
    fig = plt.figure(figsize=(15, 5))

    for idx, curve_type in enumerate(config["synthetic"]["curve_types"]):
        ax = fig.add_subplot(
            1, len(config["synthetic"]["curve_types"]), idx + 1, projection='3d')
        generate_and_render(ax, curve_type, generator,
                            enable_fitting=enable_fitting)

    plt.tight_layout()
    plt.show()
