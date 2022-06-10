"""DuckDB-backed storage and SQL summaries of experiment data."""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import Settings


def connect(path: str | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path or Settings.from_env().db_path)


def save_experiment(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Replace the rows of experiment `name` in the `assignments` table."""
    con.execute("""CREATE TABLE IF NOT EXISTS assignments (
        experiment VARCHAR, user_id BIGINT, arm VARCHAR, pre_value DOUBLE, value DOUBLE)""")
    con.execute("DELETE FROM assignments WHERE experiment = ?", [name])
    frame = df.assign(experiment=name)
    if "pre_value" not in frame:
        frame["pre_value"] = float("nan")
    frame = frame[["experiment", "user_id", "arm", "pre_value", "value"]]
    con.register("_incoming", frame)
    con.execute("INSERT INTO assignments SELECT * FROM _incoming")
    con.unregister("_incoming")


def load_experiment(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(
        "SELECT user_id, arm, pre_value, value FROM assignments WHERE experiment = ? ORDER BY user_id",
        [name]).df()


def arm_summary(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Per-arm users, mean, standard deviation and percentiles of order value."""
    return con.execute("""
        SELECT arm,
               COUNT(*)                         AS users,
               AVG(value)                       AS mean_value,
               STDDEV_SAMP(value)               AS sd_value,
               QUANTILE_CONT(value, 0.5)        AS median_value,
               QUANTILE_CONT(value, 0.95)       AS p95_value,
               CORR(value, pre_value)           AS pre_post_corr
        FROM assignments WHERE experiment = ? GROUP BY arm ORDER BY arm""", [name]).df()
