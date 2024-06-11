import CADDEE_alpha as cd
import csdl_alpha as csdl
import pytest


recorder =  csdl.Recorder(inline=True)
recorder.start()

def test_setting_of_caddee_attributes():
    caddee = cd.CADDEE()
    comp = cd.Component()
    
    # Check to make sure only Configuration instance can be set
    with pytest.raises(Exception) as exc_info:
        caddee.base_configuration = comp

    assert exc_info.type is TypeError
    assert str(exc_info.value) == f"'base_configuration' must be of type {cd.Configuration}; received {cd.Component}"

    # Check to make sure conditions cannot be re-set
    with pytest.raises(Exception) as exc_info:
        caddee.conditions = {}

    assert exc_info.type is Exception
    assert str(exc_info.value) == "'conditions' attribute cannot be re-set"


def test_setting_of_comps():
    """Test description: if not isinstance(Componet.comps["name"], Component): raise TypeError
    """
    aircraft = cd.aircraft.components.Aircraft()
    airframe_str = "airframe"

    # Check to make sure only components can be added
    # Test case adding string instead of Component
    with pytest.raises(Exception) as exc_info:
        aircraft.comps["airframe"] = airframe_str

    assert exc_info.type is TypeError
    assert str(exc_info.value) == f"Components must be of type(s) {cd.Component}; received {str}"
