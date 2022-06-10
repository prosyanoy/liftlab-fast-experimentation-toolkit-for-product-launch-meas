"""LiftLab: design, analyse and monitor A/B tests for restaurant product launches."""
from .analysis import AnalysisResult, analyze
from .guardrails import aa_simulation, srm_check
from .power import minimum_detectable_effect, sample_size_per_arm
from .simulate import ExperimentConfig, simulate_experiment

__all__ = [
    "AnalysisResult", "analyze", "aa_simulation", "srm_check",
    "minimum_detectable_effect", "sample_size_per_arm",
    "ExperimentConfig", "simulate_experiment",
]
__version__ = "0.1.0"
