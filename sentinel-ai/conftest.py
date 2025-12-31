"""
Pytest configuration and fixtures for smoke tests

This file mocks the vertexai module so tests can run without
the google-cloud-aiplatform package installed.
"""

import sys
from unittest.mock import MagicMock


# Mock the vertexai module and its submodules before any imports
vertexai_mock = MagicMock()
sys.modules['vertexai'] = vertexai_mock
sys.modules['vertexai.preview'] = MagicMock()
sys.modules['vertexai.preview.language_models'] = MagicMock()
sys.modules['vertexai.preview.generative_models'] = MagicMock()
