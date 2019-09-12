# Context
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pytest

# Imports
import Augmentor
import Augmentor.Operations


def test_add_rotate_operation():
    p = Augmentor.Pipeline()

    assert len(p.augmentor_images) == 0

    with pytest.raises(ValueError):
        p.rotate(probability=1, max_left_rotation=50, max_right_rotation=50)
        p.rotate(probability=1.1, max_left_rotation=10, max_right_rotation=10)
        p.rotate(probability='a string', max_left_rotation=10, max_right_rotation=10)

    assert len(p.operations) == 0

    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)

    assert len(p.operations) == 1
    assert isinstance(p.operations[0], Augmentor.Operations.Operation)
