"""
BBAC Framework - Experiments Module
"""

from .run import ExperimentRunner
from .scenarios import ScenarioManager, BASELINE_SCENARIOS
from .metrics import MetricsCollector
from .ablation import AblationStudy
from .baseline_comparison import BaselineComparison

__all__ = [
    'ExperimentRunner',
    'ScenarioManager',
    'BASELINE_SCENARIOS',
    'MetricsCollector',
    'AblationStudy',
    'BaselineComparison'
]
