import numpy as np
import pytest
from phantom_sim.tests.fixtures.make_synth_config import straight_bundle


@pytest.fixture
def synth_config():
    return straight_bundle()
