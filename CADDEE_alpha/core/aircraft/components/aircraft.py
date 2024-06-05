from lsdo_function_spaces import FunctionSet
from CADDEE_alpha.core.component import Component
from typing import Union


class Aircraft(Component):
    """Aircraft container component"""
    def __init__(self, geometry: Union[FunctionSet, None] = None, **kwargs) -> None:
        kwargs["do_not_remake_ffd_block"] = True
        super().__init__(geometry, **kwargs)
        self._skip_ffd = True
        