"""
BBAC Framework - ROS2 Nodes

This package contains ROS2 node implementations:
- BBAC Controller: Main decision-making node
- Robot Agents: Simulated robot agents
- Human Agents: Simulated human agents
"""

__version__ = "0.1.0"

from .bbac_controller import BBACController
from .robot_agents import RobotAgentNode
from .human_agents import HumanAgentNode

__all__ = [
    "BBACController",
    "RobotAgentNode",
    "HumanAgentNode"
]
