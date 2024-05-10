import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union, List, Dict
from CADDEE_alpha.utils.caddee_dict import CADDEEDict


@dataclass
class Discretization(csdl.VariableGroup):
    nodal_coordinates: Union[csdl.Variable, None]
    nodal_velocities: Union[csdl.Variable, None] = None
    mesh_quality = None
    _has_been_expanded = False

    def __post_init__(self):
        csdl.check_parameter(self.nodal_coordinates, "nodal_coordinates", types=csdl.Variable, allow_none=True)
        csdl.check_parameter(self.nodal_velocities, "nodal_velocities", types=csdl.Variable, allow_none=True)

class DiscretizationsDict(CADDEEDict):
    def __init__(self, types=Discretization, *args, **kwargs):
        super().__init__(types, *args, **kwargs)
    
    def __setitem__(self, key, value):
        # Check if value is a mesh
        if not isinstance(value, Discretization):
            raise TypeError(f"Can only add meshes to mesh dict; Received object of type {type(value)}.")
        
        elif key in self:
            print(f"Overwriting/updating mesh {key}")
            super().__setitem__(key, value, allow_overwrite=True)
        
        else:
            super().__setitem__(key, value)
    
    def __getitem__(self, key) -> Discretization:
        return super().__getitem__(key)

class SolverMesh:
    discretizations : DiscretizationsDict = DiscretizationsDict()


class MeshContainer(CADDEEDict):
    def __init__(self, types=SolverMesh, *args, **kwargs):
        super().__init__(types, *args, **kwargs)
    
    def __getitem__(self, key) -> SolverMesh:
        return super().__getitem__(key)
   
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    # def __setitem__(self, key, value) -> None:
    #     # Check if value is a mesh
    #     if not isinstance(value, SolverMesh):
    #         raise TypeError(f"Can only add meshes to mesh dict; Received object of type {type(value)}.")
    
    #     # Check if key has already been defined 
    #     elif key in self:
    #         print(f"overwriting/updating mesh {key} in mesh container")
    #         # raise KeyError(f"Mesh '{key}' has already been specified.")
        
    #     # Proceed with __setitem__ 
    #     else:
    #         super().__setitem__(key, value)

    # def __getitem__(self, key, identifier: str = None) -> Discretization:
    #     csdl.check_parameter(identifier, "identifier", types=str, allow_none=True)
    #     from CADDEE_alpha.core.component import Component
        
    #     # Access mesh container directly through components
    #     if isinstance(key, tuple):
    #         comp, identifier = key
    #         if not isinstance(comp, Component):
    #             raise Exception(f"Can only access mesh container through instances of components. Received type {type(comp)}")
    #     else:
    #         if not isinstance(key, Component):
    #             raise Exception(f"Can only access mesh container through instances of components. Received type {type(key)}")
    #         else:
    #             comp = key

    #     # Check if there are any meshes associated with the component
    #     comp_name = comp._name
    #     mesh_names = []
    #     for mesh_name in self.keys():
    #         if identifier is not None:
    #             if comp_name in mesh_name and identifier in mesh_name:
    #                 mesh_names.append(mesh_name)
    #         else:
    #             if comp_name in mesh_name:
    #                 mesh_names.append(mesh_name)
    
    #     if not mesh_names:
    #         if identifier is not None:
    #             raise Exception(f"No meshes associated with component {comp} and identifier '{identifier}'. Existing meshes are {list(self.keys())}")
    #         else:
    #             raise Exception(f"No meshes associated with component {comp}. Existing meshes are {list(self.keys())}")

    #     # Return a list of all meshes associated with a component.
    #     meshes = []
    #     for mesh_name in mesh_names:
    #         meshes.append(super().__getitem__(mesh_name))
    
    #     return meshes
    

    
