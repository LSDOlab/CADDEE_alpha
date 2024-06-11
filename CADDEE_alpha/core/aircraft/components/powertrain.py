from lsdo_function_spaces import FunctionSet
from CADDEE_alpha.core.component import Component
from typing import Union


class Powertrain(Component):
    def __init__(self, geometry: Union[FunctionSet, None] = None, **kwargs) -> None:
        super().__init__(geometry, **kwargs)
        self._skip_ffd = True