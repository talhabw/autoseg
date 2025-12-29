"""
Basic tests to verify the project structure is working.
"""

import pytest


def test_import_app():
    """Test that app module can be imported."""
    import app
    assert app.__version__ == "0.1.0"


def test_import_core():
    """Test that core module can be imported."""
    import core
    assert core is not None


def test_import_ml():
    """Test that ml module can be imported."""
    import ml
    assert ml is not None
