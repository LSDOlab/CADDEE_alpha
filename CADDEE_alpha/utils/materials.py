import lsdo_function_spaces as fs
import numpy as np
import csdl_alpha as csdl
from csdl_alpha.utils.typing import VariableLike
from typing import Union

import numpy as np
import xml.etree.ElementTree as ET
import sys

# TODO: ditch thickness

class Material():
    def __init__(self, name:str=None, density:VariableLike=None, 
                 compliance:VariableLike=None, 
                 thickness:Union[VariableLike, fs.FunctionSet]=None, 
                 strength:VariableLike=None):
        """Initialize a Material object.

        Parameters
        ----------
        name : str, optional
            The name of the material. Defaults to None.
        density : VariableLike, optional
            The density of the material. Defaults to None.
        compliance : VariableLike, optional
            The compliance matrix of the material. Defaults to None.
        thickness : Union[VariableLike, fs.FunctionSet], optional
            The thickness of the material. Can be a single value or a function set. Defaults to None.
        strength : VariableLike, optional
            The strength matrix of the material. Defaults to None.
        """
        self.name = name
        self.density = density
        self.compliance = compliance
        self.thickness = thickness
        self.strength = strength
        
    # https://docs.python.org/3/library/xml.etree.elementtree.html
    def import_xml(self, fname:str):
        """Import material properties from an XML file.

        Parameters
        ----------
        fname : str
            The name of the file to import from.
        """
        tree = ET.parse(fname)
        root = tree.getroot()

        self.name = root.attrib['name']

        if root.find('density') is not None: 
            self.density = float(root.find('density').text)
            
        if root.find('compliance') is not None:
            self.compliance = np.array(
                [[float(x) for x in i.text.split()] 
                for i in root.find('compliance')]
                )
        
        if root.find('thickness') is not None: 
            self.thickness = float(root.find('thickness').text)

        if root.find('strength') is not None:
            self.strength = np.array(
                [[float(x) for x in i.text.split()] 
                for i in root.find('strength')]
                )

    def export_xml(self, fname):
        """Export material properties to an XML file.

        Parameters
        ----------
        fname : str
            The name of the file to export to.
        """
        root = ET.Element('material')
        root.set('type', self.__class__.__name__)
        root.set('name', self.name)

        if self.density is not None:
            ET.SubElement(root, 'density').text = str(self.density)

        if self.compliance is not None:
            compliance_el = ET.SubElement(root, 'compliance')
            for row in self.compliance:
                ET.SubElement(compliance_el, 'row').text = ' '.join(map(str, row))

        if self.thickness is not None:
            ET.SubElement(root, 'thickness').text = str(self.thickness)

        if self.strength is not None:
            strength_el = ET.SubElement(root, 'strength')
            for row in self.strength:
                ET.SubElement(strength_el, 'row').text = ' '.join(map(str, row))

        tree = ET.ElementTree(root)
        if sys.version_info[1] >= 9:
            ET.indent(tree) # makes it pretty, new for Python3.9
        tree.write(fname)

class IsotropicMaterial(Material):
    def __init__(self, name:str=None, density:VariableLike=None, thickness:VariableLike=None,
                 E:VariableLike=None, nu:VariableLike=None, G:VariableLike=None,
                 Ft:VariableLike=None, Fc:VariableLike=None, F12:VariableLike=None):
        """Initialize an isotropic material object.

        Parameters
        ----------
        name : str, optional
            The name of the material. Defaults to None.
        density : VariableLike, optional
            The density of the material. Defaults to None.
        thickness : VariableLike, optional
            The thickness of the material. Defaults to None.
        E : VariableLike, optional
            The Young's modulus of the material. Defaults to None.
        nu : VariableLike, optional
            The Poisson's ratio of the material. Defaults to None.
        G : VariableLike, optional
            The shear modulus of the material. Defaults to None.
        Ft : VariableLike, optional
            The tensile strength of the material. Defaults to None.
        Fc : VariableLike, optional
            The compressive strength of the material. Defaults to None.
        F12 : VariableLike, optional
            The shear strength of the material. Defaults to None.
        """
        super().__init__(name=name, density=density, thickness=thickness)

        if E is None and nu is None and G is None:
            pass
        else:
            self.set_compliance(E=E, nu=nu, G=G)
        if Ft is None and Fc is None and F12 is None:
            pass
        else:
            if Ft is None or Fc is None or F12 is None:
                raise Exception('Material strength properties are uderdefined')
            self.set_strength(Ft=Ft, Fc=Fc, F12=F12)


    def set_compliance(self, E = None, nu = None, G = None):
            if not None in [E, nu]:
                pass
            elif not None in [G, nu]:
                E = G*2*(1+nu)
            elif not None in [E, G]:
                nu = E/(2*G)-1
            else:
                raise Exception('Material properties are uderdefined')

            self.compliance = 1/E*np.array(
                [[1, -nu, -nu, 0, 0, 0],
                [-nu, 1, -nu, 0, 0, 0],
                [-nu, -nu, 1, 0, 0, 0],
                [0, 0, 0, 1+nu, 0, 0],
                [0, 0, 0, 0, 1+nu, 0],
                [0, 0, 0, 0, 0, 1+nu]]
            )

    def from_compliance(self):
        E = 1/self.compliance[0,0]
        nu = -self.compliance[0,1]*E
        G = E/(2*(1+nu))
        return E, nu, G

    def set_strength(self, Ft, Fc, F12):
        self.strength = np.array([[Ft, Ft, Ft],[Fc, Fc, Fc],[F12, F12, F12]])

class TransverseMaterial(Material):
    def set_compliance(self, EA, ET, vA, GA, vT = None, GT = None):
            # E1 = EA
            # E2 = E3 = ET
            # v12 = v13 = vA
            # v23 = vT
            # G12 = G13 = GA

            if vT is not None and GT is None:
                GT = ET/(2*(1+vT)) # = G23
            elif GT is not None and vT is None:
                vT = ET/(2*GT)-1
            else:
                raise Exception('Material is underdefined')

            self.compliance = np.array(
                [[1/ET, -vT/ET, -vA/EA, 0, 0, 0],
                [-vT/ET, 1/ET, -vA/EA, 0, 0, 0],
                [-vA/EA, -vA/EA, 1/EA, 0, 0, 0],
                [0, 0, 0, 1/GA, 0, 0],
                [0, 0, 0, 0, 1/GA, 0],
                [0, 0, 0, 0, 0, 1/GT]]
            )

    def from_compliance(self):
        ET = 1/self.compliance[0,0]
        EA = 1/self.compliance[5,5]
        vT = -self.compliance[1,0]*ET
        vA = -self.compliance[2,0]*EA
        GA = 1/self.compliance[3,3]        
        return EA, ET, vA, vT, GA

    def set_strength(self, F1t, F1c, F2t, F2c, F12, F23):
        self.strength = np.array(
            [[F1t, F2t, F2t],
            [F1c, F2c, F2c],
            [F12, F12, F23]]
            )
        

def import_material(fname:str) -> Material:
    """Import material from an XML file.

    Parameters
    ----------
    fname : str
        The name of the file to import from.

    Returns
    -------
    Material
        The material object.
    """


    tree = ET.parse(fname)
    root = tree.getroot()
    name = root.attrib['name']

    mat_type = root.attrib['type']
    if mat_type == 'IsotropicMaterial':
        material = IsotropicMaterial()
    elif mat_type == 'TransverseMaterial':
        material = TransverseMaterial()
    else:
        material = Material()

    material.name = name
    
    if root.find('density') is not None: 
        material.density = float(root.find('density').text)
    
    if root.find('compliance') is not None:
        material.compliance = np.array(
            [[float(x) for x in i.text.split()] 
            for i in root.find('compliance')]
            )
    
    if root.find('thickness') is not None: 
        material.thickness = float(root.find('thickness').text)
    
    if root.find('strength') is not None:
        material.strength = np.array(
            [[float(x) for x in i.text.split()] 
            for i in root.find('strength')]
            )
    
    return material