"""
BBAC Framework - Core Modules

This package contains the core decision-making layers of the BBAC framework:
- Layer 1: Rule-based Access Control
- Layer 2: Behavioral Analysis (Markov Chains)
- Layer 3: ML Anomaly Detection (Isolation Forest)
"""

__version__ = "0.1.0"
__author__ = "BBAC Framework Contributors"

from .rule_engine import RuleEngine
from .behavioral_analysis import BehavioralAnalyzer
from .ml_detection import MLAnomalyDetector

__all__ = [
    "RuleEngine",
    "BehavioralAnalyzer",
    "MLAnomalyDetector"
]
