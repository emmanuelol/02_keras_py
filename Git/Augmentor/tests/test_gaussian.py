import pytest

# Context
import os
import sys
import tempfile
sys.path.insert(0, os.path.abspath('.'))

import Augmentor
from Augmentor import Operations


def test_create_gaussian_distortion_object():
    g = Augmentor.Operations.GaussianDistortion(1, 8, 8, 8, "true", "true", 1.0, 1.0, 1.0, 1.0)
    assert g is not None


def test_add_gaussian_to_pipeline():
    tmp_dir = tempfile.mkdtemp()

    p = Augmentor.Pipeline(tmp_dir)
    p.gaussian_distortion(1, 8, 8, 8, "true", "true")

    assert p is not None
