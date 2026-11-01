import csv

from r0.cli import main
from tests.conftest import CONFIGS


def test_cli_deterministic_with_csv(tmp_path, capsys):
    out = tmp_path / "traj.csv"
    assert main(["simulate", str(CONFIGS / "seir.yaml"), "--days", "100", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "R0 = 2.5" in text
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 101
    assert {"S", "E", "I", "R", "new_E"} <= set(rows[0])


def test_cli_stochastic(capsys):
    args = ["simulate", str(CONFIGS / "seir.yaml"), "--days", "50", "--engine", "stochastic", "--replicates", "3"]
    assert main(args) == 0
    assert "intervalle 90 %" in capsys.readouterr().out
