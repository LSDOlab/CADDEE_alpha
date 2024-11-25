from CADDEE_alpha.core.configuration import Configuration
from CADDEE_alpha.core.condition import Condition
import csdl_alpha as csdl
from CADDEE_alpha.utils.caddee_dict import CADDEEDict


class ConditionsDict(CADDEEDict):
    def __getitem__(self, key) -> Condition:
        return super().__getitem__(key)

class CADDEE:
    """Top-level CADDEE class.
    
    Attributes:
    -----------
    configurations :  dict
        a dictionary of the configurations

    conditions : dict
        a dictionary of the conditions
    """
    def __init__(self, 
                 base_configuration: Configuration = None,
                 conditions: ConditionsDict = None,
                 ) -> None:
        csdl.check_parameter(base_configuration, "base_configuration", types=Configuration, allow_none=True)
        self._base_configuration = base_configuration
        
        # Initialize a new ConditionsDict if conditions is None
        self._conditions = conditions if conditions is not None else ConditionsDict(types=Condition)

    @property
    def base_configuration(self):
        return self._base_configuration
    
    @base_configuration.setter
    def base_configuration(self, value):
        if not isinstance(value, Configuration):
            raise TypeError(f"'base_configuration' must be of type {Configuration}; received {type(value)}")
        self._base_configuration = value

    @property
    def conditions(self) -> ConditionsDict:
        return self._conditions
    
    @conditions.setter
    def conditions(self, value):
        raise Exception("'conditions' attribute cannot be re-set")
    