from hnne import HNNE
import numpy as np
import pytest

data = np.random.random(size=(1000, 256))


# Test if n_components is set correctly without any arguments
def test_default_n_components():
    hnne = HNNE()
    assert hnne.n_components == 2


# Test the default projection shape
def test_default_projection_shape():
    hnne = HNNE()
    projection = hnne.fit_transform(data)
    assert projection.shape == (1000, 2)


# This raises a DeprecationWarning
def test_hnne_deprecation_warning():
    with pytest.warns(DeprecationWarning, match='The argument `dim` is being deprecated'):
        hnne = HNNE(dim=3)
        assert hnne.n_components == 3


# Test dimensionality with non-default argument
def test_dimensionality_3d():
    hnne = HNNE(n_components=3)
    projection = hnne.fit_transform(data)
    assert projection.shape == (1000, 3)


# Really the same as test_default_projection_shape but with explicitly stated n_components argument
def test_dimensionality_2d():
    hnne = HNNE(n_components=2)
    projection = hnne.fit_transform(data)
    assert projection.shape == (1000, 2)


# Raise a warning if both dim and n_components are used
def test_both_arguments_specified():
    with pytest.warns(UserWarning, match='It is sufficient to specify `n_components`'):
        hnne = HNNE(n_components=3, dim=3)
        assert hnne.n_components == 3


# ... if they are different values
def test_argument_mismatch():
    with pytest.raises(ValueError, match='Conflicting values:'):
        _ = HNNE(n_components=3, dim=2)


# Test the deprecation warning of dim in fit_transform
def test_fit_transform_dim_deprecation():
    hnne = HNNE(n_components=3)
    with pytest.warns(DeprecationWarning, match='The argument `dim` is being deprecated in favor of `override_dim`'):
        projection = hnne.fit_transform(data, dim=2)
        assert projection.shape == (1000, 2)


# Check if output dimensions are correctly overridden
def test_override_dim():
    hnne = HNNE(n_components=3)
    projection = hnne.fit_transform(data, override_dim=2, verbose=True)
    assert projection.shape == (1000, 2)


# ... and its default behaviour
def test_no_arguments_fit_transform():
    hnne = HNNE(n_components=3)
    projection = hnne.fit_transform(data)
    assert projection.shape == (1000, 3)
