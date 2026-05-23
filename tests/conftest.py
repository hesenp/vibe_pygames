import sys
from pathlib import Path

# Make `from _shared.star_common import ...` resolvable in tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "games"))
