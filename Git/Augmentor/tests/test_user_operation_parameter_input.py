import Augmentor
import Augmentor.ImageUtilities


def test_user_param_parsing():

    # Scalar input should return itself, as a integer
    scalar_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=5)
    assert scalar_input == 5
    assert type(scalar_input) == int

    # A float input should return itself as a float.
    float_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=1.01)
    assert float_input == 1.01
    assert type(float_input) == float

    # Lists are interpreted as [from, to, step] and should return a value between from and to
    # while respecting the step parameter.
    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[-10, 10, 0.5])
    assert -10 <= list_input <= 10

    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[-10, 10, 0.01])
    assert -10 <= list_input <= 10

    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[-10, 10, 5])
    assert -10 <= list_input <= 10
    assert list_input % 5 == 0

    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[-10, 10, 2])
    assert -10 <= list_input <= 10
    assert list_input % 2 == 0

    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[2])
    assert 0 <= list_input <= 2
    assert list_input in [0, 1, 2]

    list_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=[10, 12])
    assert 10 <= list_input <= 12
    assert list_input in [10, 11, 12]

    # Tuples are interpreted as meaning a number of discrete values
    tuple_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=(2, 4, 6))
    assert tuple_input in (2, 4, 6)

    tuple_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=(2.1, 4.2, 6.3))
    assert tuple_input in (2.1, 4.2, 6.3)

    tuple_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=(2,))  # Ensure it is interpreted
    assert tuple_input == 2                                                       # as a tuple and not a scalar.
    assert type(tuple_input) == int

    tuple_input = Augmentor.ImageUtilities.parse_user_parameter(user_param=(2.2,))
    assert tuple_input == 2.2
    assert type(tuple_input) == float
