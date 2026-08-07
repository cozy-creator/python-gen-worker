"""micro-diffusion — the fleet's smallest REAL endpoint family.

Its reason to exist is cycle time: a full production-path AOT mint against
sdxl is 36 export entries and ~95 minutes on a pod, which makes every mint
change a multi-hour experiment. This family declares THREE entries over a
generated toy checkpoint and runs the identical machinery.
"""

from .aot_declaration import DECLARATION, FAMILY
from .main import Generate, MicroDefaults, MicroIn, MicroOut, Size
from .pipeline import MicroPipeline

__all__ = [
    "DECLARATION",
    "FAMILY",
    "Generate",
    "MicroDefaults",
    "MicroIn",
    "MicroOut",
    "MicroPipeline",
    "Size",
]
