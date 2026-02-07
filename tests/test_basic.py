"""
Basic tests to verify the project structure is working.
"""

import pytest


def test_import_app():
    """Test that backend module can be imported."""
    import backend.main

    assert backend.main.app is not None


def test_import_core():
    """Test that core module can be imported."""
    import core

    assert core is not None


def test_import_ml():
    """Test that ml module can be imported."""
    import ml

    assert ml is not None
