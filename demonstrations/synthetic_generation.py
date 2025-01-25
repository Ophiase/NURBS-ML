import numpy as np
import matplotlib.pyplot as plt
from nurbs.synthetic import SyntheticCurveGenerator
from visualization.plotter import NURBSPresenter


def run(config: dict):
    generator = SyntheticCurveGenerator(config["synthetic"])

    fig = plt.figure(figsize=(15, 5))

    # Generate and plot different curve types
    for idx, curve_type in enumerate(config["synthetic"]["curve_types"]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        curve = generator.generate(curve_type)
        ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2],
                   c=curve[:, 2], cmap='viridis')
        ax.set_title(f"{curve_type.capitalize()} Curve")

    plt.tight_layout()
    plt.show()
