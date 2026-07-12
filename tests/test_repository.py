import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_notebooks_are_clean() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_datasets_and_checkpoints_are_not_committed() -> None:
    tracked = subprocess.run(
        ("git", "ls-files"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any(path.startswith("data/") for path in tracked)
    assert not any(path.endswith((".pth", "-ubyte")) for path in tracked)
