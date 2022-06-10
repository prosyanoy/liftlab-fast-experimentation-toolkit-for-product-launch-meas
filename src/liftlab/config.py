"""Runtime configuration read from environment variables (all optional)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    alpha: float = 0.05
    seed: int = 42
    threads: int = 0
    db_path: str = ":memory:"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            alpha=float(env.get("LIFTLAB_ALPHA", cls.alpha)),
            seed=int(env.get("LIFTLAB_SEED", cls.seed)),
            threads=int(env.get("LIFTLAB_THREADS", cls.threads)),
            db_path=env.get("LIFTLAB_DB", cls.db_path),
        )
