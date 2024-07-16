__version__ = '0.1.4'


from CADDEE_alpha.core.caddee import CADDEE
from CADDEE_alpha.core.configuration import Configuration, VectorizedConfig
from CADDEE_alpha.core.component import Component
import CADDEE_alpha.core.aircraft as aircraft
import CADDEE_alpha.core.mesh as mesh
from CADDEE_alpha.utils.import_geometry import import_geometry
from CADDEE_alpha.utils.units import Units
from CADDEE_alpha.utils.loading import load_var
import caddee_materials as materials
import CADDEE_alpha.utils.mesh_utils as mesh_utils