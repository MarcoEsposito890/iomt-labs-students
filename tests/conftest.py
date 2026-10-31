"""Test configuration.

Tests that use the data_dir or df fixtures are skipped, with a message, when
the dataset has not been downloaded.
"""
import pytest

from iomtlab.config import DATA_DIR


@pytest.fixture(scope="session")
def data_dir():
    """The dataset folder; the tests that use it are skipped when it is empty."""
    return DATA_DIR


def pytest_collection_modifyitems(config, items):
    """Skip the tests that need the dataset when it is missing."""
    if DATA_DIR.is_dir() and any(DATA_DIR.glob("s*/*.csv")):
        return
    skip_data = pytest.mark.skip(reason=f"dataset assente in {DATA_DIR}: "
                                        f"lancia `python scripts/get_data.py`")
    for item in items:
        if "data_dir" in item.fixturenames or "df" in item.fixturenames:
            item.add_marker(skip_data)
