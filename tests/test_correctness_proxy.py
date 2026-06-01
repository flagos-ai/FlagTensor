from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_CTESTS = ROOT / "ctests"

def test_correctness_proxy_relocated():
    assert LEGACY_CTESTS.exists()
