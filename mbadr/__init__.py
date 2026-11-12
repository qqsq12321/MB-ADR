"""Neural end-to-end retargeting for dexterous robot hands.

Model-based in the strict sense: supervision comes from an *analytic*
differentiable forward-kinematics model of the robot plus the adaptive loss,
not from a learned FK surrogate and not from labels.

Supported hands are defined in :mod:`.robot_configs`; build a spec with
``HandSpec.from_name("shadow")``.
"""

from .chain import KinematicChain
from ._version import __version__
from .config import HandSpec, SUPPORTED_HANDS, default_config_path
from .fk_numpy import NumpyFK

__all__ = [
    "__version__",
    "KinematicChain",
    "HandSpec",
    "NumpyFK",
    "SUPPORTED_HANDS",
    "default_config_path",
]
