import os
from pathlib import Path
import lsdo_geo as lg
import numpy as np
import lsdo_function_spaces as lfs


_REPO_ROOT_FOLDER = Path(__file__).parents[0]
TEST_GEOMETRY_FOLDER = _REPO_ROOT_FOLDER / '..'/ '..' / 'examples'/ 'test_geometries'

def import_geometry(
        file_name: str,
        file_path = TEST_GEOMETRY_FOLDER,
        refit = False,
        refit_num_coefficients: tuple = (40, 40), 
        refit_b_spline_order: tuple = (4, 4),
        refit_resolution: tuple = (200, 200),
        rotate_to_body_fixed_frame: bool = True,
        scale: float = 1.0
    ) -> lg.Geometry:
        """Import a (OpenVSP) .stp file.

        Parameters
        ----------
        file_name : str
            name of the file (.stp support only at this time)

        file_path : _type_, optional
            file path, by default TEST_GEOMETRY_FOLDER

        refit_num_coefficients : tuple, optional
            number coefficients, by default (40, 40)
        
        refit_b_spline_order : tuple, optional
            order of the b-spline refit, by default (4, 4)
        
        refit_resolution : tuple, optional
            drive the quality of the refit, by default (200, 200)
        
        rotate_to_body_fixed_frame : bool, optional
            apply a 180 deg rotation about the z-axis followed by a 180 deg rotation about the x-axis, by default True

        Returns
        -------
        lg.BSplineSet
            the geometry function

        Raises
        ------
        Exception
            If file is not .stp format
        Exception
            If unknown path or file path
        """
        
        if not file_name.endswith(".stp"):
            raise Exception(f"Can only import '.stp' files at the moment. Received {file_name}")
        
        if not os.path.isfile(file_path / file_name):
            raise Exception(f"Unknown file path or file. File path: {file_path}/{file_name}")


        geometry = lg.import_geometry(TEST_GEOMETRY_FOLDER / file_name, parallelize=False, scale=scale)

        if refit:
            refit_space = lfs.BSplineSpace(2, refit_b_spline_order, refit_num_coefficients)
            geometry.refit(
                refit_space,
                grid_resolution=refit_resolution,
            )

        # if rotate_to_body_fixed_frame:
        if rotate_to_body_fixed_frame:
            geometry.rotate(
                axis_origin=np.array([0., 0., 0]),
                axis_vector=np.array([0., 0., 1]),
                angles=180.,
            )

            geometry.rotate(
                axis_origin=np.array([0., 0., 0]),
                axis_vector=np.array([1., 0., 0]),
                angles=180.,
            )


        return geometry