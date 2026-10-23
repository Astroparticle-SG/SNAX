import numpy as np
import pytest

from snudd import config
from snudd.binding import binding_xe
from snudd.models import SM, GeneralNSI
from snudd.rrpa import rrpa_scaling
from snudd.targets import Electron, Nucleus

NSI_MUTAU = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.1, 0.0]])


@pytest.fixture
def e_nus():
    """Neutrino energies in GeV."""
    return np.geomspace(1e-4, 1e-2, 5)


@pytest.fixture
def nsi_model():
    return GeneralNSI(NSI_MUTAU, np.pi / 4, 0)


@pytest.fixture(params=["SM", "NSI"])
def model(request):
    if request.param == "SM":
        return SM()
    return GeneralNSI(NSI_MUTAU, np.pi / 4, 0)


@pytest.fixture
def xe_nucleus():
    return Nucleus(54, 132, mass=131.9041535 * config.u)


@pytest.fixture
def xe_electron(xe_nucleus):
    return Electron(xe_nucleus, binding_xe, rrpa_scaling)
