import pytest

from flense.config import FlenseConfig
from flense.app import create_app


@pytest.fixture
def config():
    return FlenseConfig()


@pytest.fixture
def app(config):
    return create_app(config)
