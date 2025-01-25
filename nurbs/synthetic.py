import numpy as np
import numpy.typing as npt
from typing import Literal


class SyntheticCurveGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate(
        self,
        curve_type: Literal["helix", "random", "constrained"]
    ) -> npt.NDArray[np.float64]:
        t = np.linspace(0, 2*np.pi, 100)

        match curve_type:
            case "helix":
                return self._helix(t)
            case "random":
                return self._random(t)
            case "constrained":
                return self._constrained(t)
            case _:
                raise ValueError(f"Unknown curve type: {curve_type}")

    def _helix(self, t: npt.NDArray) -> npt.NDArray:
        x = np.cos(t)
        y = np.sin(t)
        z = 0.5 * t
        return self._add_noise(np.stack([x, y, z], axis=1))

    def _random(self, t: npt.NDArray) -> npt.NDArray:
        rand_shape = np.random.randn(3, 3)  # Random 3x3 transformation
        base = np.stack([np.cos(t), np.sin(t), t], axis=1)
        return self._add_noise(base @ rand_shape)

    def _constrained(self, t: npt.NDArray) -> npt.NDArray:
        curve = self._random(t)
        curve[:, 2] = np.clip(curve[:, 2], *self.config["z_clip"])
        return curve

    def _add_noise(self, curve: npt.NDArray) -> npt.NDArray:
        noise = self.config["noise_level"] * np.random.randn(*curve.shape)
        return curve + noise
