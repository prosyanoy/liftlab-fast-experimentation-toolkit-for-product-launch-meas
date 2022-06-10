from liftlab.cli import main


def test_cli_commands(capsys):
    assert main(["demo", "--users", "5000"]) == 0
    assert main(["power"]) == 0
    assert main(["aa", "--sims", "20"]) == 0
    assert main(["bench", "--users", "2000", "--resamples", "100"]) == 0
    out = capsys.readouterr().out
    assert "per arm" in out and "speedup_vs_numpy" in out
