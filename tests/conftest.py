import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def cfg():
    from edgegen.config import load_config
    return load_config(ROOT / "config" / "default.yaml")
