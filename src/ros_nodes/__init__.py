"""
BBAC Framework - ROS2 Nodes

This package contains ROS2 node implementations:
- BBAC Controller: Main decision-making node
- Robot Agents: Simulated robot agents
- Human Agents: Simulated human agents
"""

__version__ = "0.1.0"

from .controller import BBACControllerfrom
from .human_agents import OperatorNode, SupervisorNode, TechnicianNode
from .robot_agents import AssemblyRobotNode, CameraRobotNode, TransportRobotNode

__all__ = [
    'BBACController',
    'AssemblyRobotNode',
    'CameraRobotNode', 
    'OperatorNode',
    'SupervisorNode',  
    'TechnicianNode',
    'TransportRobotNode'
]
