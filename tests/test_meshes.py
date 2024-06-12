import CADDEE_alpha as cd
import csdl_alpha as csdl
import numpy as np
import pytest


@pytest.fixture(scope="class")
def setup_test_class():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    wing_geometry = cd.import_geometry("simple_wing.stp")

    wing = cd.aircraft.components.Wing(
        S_ref=45.,
        AR=7.2,
        taper_ratio=0.2,
        geometry=wing_geometry,
        tight_fit_ffd=False,
    )

    return {"wing" : wing}

@pytest.mark.usefixtures("setup_test_class")
class TestMeshes:
    @pytest.fixture(autouse=True)
    def _setup(self, setup_test_class):
        self.wing = setup_test_class["wing"]

    def test_chord_surface(self):
        """Test vlm chord surface mesh."""
        
        mesh_nodes_norm_desired = 66.21782142
        chord_surface = cd.mesh.make_vlm_surface(
                self.wing,
                num_spanwise=14,
                num_chordwise=5,
                ignore_camber=True,
            )
        mesh_nodes_norm_actual = csdl.norm(chord_surface.nodal_coordinates).value

        np.testing.assert_almost_equal(
            mesh_nodes_norm_actual,
            mesh_nodes_norm_desired,
            decimal=7
        )

    def test_camber_surface(self):
        """Test vlm camber surface mesh."""
        mesh_nodes_norm_desired = 47.29397405
        
        chord_surface = cd.mesh.make_vlm_surface(
            self.wing,
            num_spanwise=10,
            num_chordwise=3,
            ignore_camber=False,
        )
        mesh_nodes_norm_actual = csdl.norm(chord_surface.nodal_coordinates).value

        np.testing.assert_almost_equal(
            mesh_nodes_norm_actual,
            mesh_nodes_norm_desired,
            decimal=7
        )

    def test_spar_rib_helper(self):
        """Test the helper function for making ribs and spars"""
        desired_coeff_norm_sum_before = 1357.73155717
        desired_coeff_norm_sum_after = 1964.7352484
        
        actual_coeff_norm_sum_before = 0
        for fun in self.wing.geometry.functions.values():
            actual_coeff_norm_sum_before += csdl.norm(fun.coefficients).value

        self.wing.construct_ribs_and_spars(
            self.wing.geometry,
            num_ribs=6,
            spar_locations=np.array([0.25, 0.75]),
            rib_locations=np.array([0., 0.2, 0.5, 0.75, 0.9, 1.])
        )
        
        actual_coeff_norm_sum_after = 0
        for fun in self.wing.geometry.functions.values():
            actual_coeff_norm_sum_after += csdl.norm(fun.coefficients).value

        np.testing.assert_almost_equal(
            actual_coeff_norm_sum_before,
            desired_coeff_norm_sum_before,
            decimal=7
        )

        np.testing.assert_almost_equal(
            actual_coeff_norm_sum_after,
            desired_coeff_norm_sum_after,
            decimal=7
        )

