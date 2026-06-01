"""opt — 优化管线：配置、场景组装、问题定义、求解器。"""

from .config import BaseConfig, BeamformingConfig, ASMConfig, MSMConfig
from .scenario import assemble_scenario
from .problem import BeamformingProblem, _COMPONENT_REGISTRY
from .problem import main_lobe_pointing, sidelobe, null_steering, directivity, hpbw, difference_beam
from .solver import minimize, CMA, DE, GWO
