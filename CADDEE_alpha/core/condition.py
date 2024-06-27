from CADDEE_alpha.core.configuration import Configuration, VectorizedConfig


class Condition:
    """The Condition class"""
    def __init__(self) -> None:
        self.parameters : dict = {}
        self._configuration : Configuration = None
        self._vectorized_configuration : VectorizedConfig = None

    @property
    def configuration(self):
        if not hasattr(self, "_configuration"):
            self._configuration = None
            return self._configuration
        return self._configuration
    
    @configuration.setter
    def configuration(self, value):
        if not isinstance(value, Configuration):
            raise TypeError(f"'configuration' must be of type {Configuration}, received {type(value)}")
        if self.vectorized_configuration is not None:
            raise ValueError("cannot set 'configuration' and 'vectorized' configuration at the same time.")
        self._configuration = value

    @property
    def vectorized_configuration(self):
        if not hasattr(self, "_vectorized_configuration"):
            self._vectorized_configuration = None
            return self._vectorized_configuration
        return self._vectorized_configuration
    
    @vectorized_configuration.setter
    def vectorized_configuration(self, value):
        if not isinstance(value, VectorizedConfig):
            raise TypeError(f"'vectorized_configuration' must be of type {VectorizedConfig}, received {type(value)}")
        if self.configuration is not None:
            raise ValueError("cannot set 'configuration' and 'vectorized' configuration at the same time.")
        self._vectorized_configuration = value

    def finalize_meshes(self):
        raise NotImplementedError(f"'finalize_meshes' has not been implemented for condition of type {type(self)}")
    
    def assemble_forces_and_moments(self):
        raise NotImplementedError(f"'assemble_forces_and_moments' has not been implemented for condition of type {type(self)}")

