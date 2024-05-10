from CADDEE_alpha.core.configuration import Configuration


class Condition:
    """The Condition class"""
    def __init__(self) -> None:
        self.parameters : dict = {}
        self._configuration : Configuration = None

    @property
    def configuration(self):
        return self._configuration
    
    @configuration.setter
    def configuration(self, value):
        if not isinstance(value, Configuration):
            raise TypeError(f"'base_configuration' must be of type {Configuration}")
        self._configuration = value

    def finalize_meshes(self):
        raise NotImplementedError(f"'finalize_meshes' has not been implemented for condition of type {type(self)}")
    
    def assemble_forces_and_moments(self):
        raise NotImplementedError(f"'assemble_forces_and_moments' has not been implemented for condition of type {type(self)}")

