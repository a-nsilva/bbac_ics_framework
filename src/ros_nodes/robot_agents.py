"""
BBAC Framework - Robot Agent Nodes

Simulates robot agents with predictable behavior patterns.
Each robot follows a defined sequence of actions.
"""

# 1. Biblioteca padrão
import json
import random
import uuid
from datetime import datetime

# 2. Bibliotecas de terceiros
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotAgentNode(Node):
    """
    Simulated Robot Agent with predictable behavior.
    
    Robots have:
    - Fixed action sequences
    - Predictable temporal patterns
    - Consistent resource access
    """
    
    def __init__(self, robot_id: str, robot_type: str, 
                 action_sequence: list, zone: str = 'production'):
        """
        Initialize robot agent.
        
        Args:
            robot_id: Unique robot identifier
            robot_type: Type of robot (assembly_robot, camera_robot, etc.)
            action_sequence: List of actions to perform cyclically
            zone: Operating zone
        """
        super().__init__(f'{robot_id.lower()}_node')
        
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.action_sequence = action_sequence
        self.zone = zone
        self.current_action_index = 0
        
        # State
        self.waiting_for_decision = False
        self.current_request_id = None
        self.actions_completed = 0
        self.actions_denied = 0
        
        # ROS2 Communication
        self.request_publisher = self.create_publisher(
            String,
            '/access_requests',
            10
        )
        
        self.decision_subscriber = self.create_subscription(
            String,
            '/access_decisions',
            self.handle_decision,
            10
        )
        
        # Work cycle timer (every 3 seconds)
        self.work_timer = self.create_timer(3.0, self.work_cycle)
        
        self.get_logger().info(
            f'Robot Agent {robot_id} ({robot_type}) initialized in zone {zone}'
        )
    
    def work_cycle(self):
        """Execute one work cycle."""
        if not self.waiting_for_decision and self.action_sequence:
            # Get next action
            action_data = self.action_sequence[self.current_action_index]
            
            # Create access request
            request_id = str(uuid.uuid4())
            request = {
                'request_id': request_id,
                'agent_id': self.robot_id,
                'agent_type': 'robot',
                'agent_role': self.robot_type,
                'resource_id': action_data['resource'],
                'action': action_data['action'],
                'timestamp': datetime.now().isoformat(),
                'context': {
                    'zone': self.zone,
                    'priority': 5.0,
                    'emergency': False
                }
            }
            
            # Publish request
            msg = String()
            msg.data = json.dumps(request)
            self.request_publisher.publish(msg)
            
            # Update state
            self.waiting_for_decision = True
            self.current_request_id = request_id
            
            self.get_logger().info(
                f'Requesting: {action_data["action"]} on {action_data["resource"]}'
            )
    
    def handle_decision(self, msg: String):
        """
        Handle access decision from BBAC Controller.
        
        Args:
            msg: Decision message
        """
        try:
            decision = json.loads(msg.data)
            
            # Check if decision is for this robot's request
            if decision.get('request_id') == self.current_request_id:
                self.waiting_for_decision = False
                
                result = decision.get('decision')
                confidence = decision.get('confidence', 0.0)
                latency = decision.get('latency_ms', 0.0)
                
                if result == 'grant':
                    # Execute action
                    action_data = self.action_sequence[self.current_action_index]
                    self.get_logger().info(
                        f'✓ GRANTED: Executing {action_data["action"]} '
                        f'(confidence: {confidence:.2f}, latency: {latency:.2f}ms)'
                    )
                    
                    # Move to next action
                    self.current_action_index = (self.current_action_index + 1) % len(self.action_sequence)
                    self.actions_completed += 1
                    
                else:
                    # Access denied
                    self.get_logger().warn(
                        f'✗ DENIED: {decision.get("explanation", {}).get("reason", "unknown")}'
                    )
                    self.actions_denied += 1
                    
                    # Still move to next action (try again later)
                    self.current_action_index = (self.current_action_index + 1) % len(self.action_sequence)
        
        except Exception as e:
            self.get_logger().error(f'Error handling decision: {e}')


class AssemblyRobotNode(RobotAgentNode):
    """Assembly Robot - predictable manufacturing tasks."""
    
    def __init__(self, robot_id: str = 'robot_assembly_001'):
        action_sequence = [
            {'resource': 'material_storage', 'action': 'read'},
            {'resource': 'assembly_station_A', 'action': 'write'},
            {'resource': 'assembly_station_A', 'action': 'execute'},
            {'resource': 'assembly_log', 'action': 'write'},
        ]
        
        super().__init__(
            robot_id=robot_id,
            robot_type='assembly_robot',
            action_sequence=action_sequence,
            zone='production'
        )


class CameraRobotNode(RobotAgentNode):
    """Camera Robot - quality inspection tasks."""
    
    def __init__(self, robot_id: str = 'robot_camera_001'):
        action_sequence = [
            {'resource': 'inspection_station', 'action': 'read'},
            {'resource': 'vision_analysis', 'action': 'execute'},
            {'resource': 'quality_db', 'action': 'write'},
        ]
        
        super().__init__(
            robot_id=robot_id,
            robot_type='camera_robot',
            action_sequence=action_sequence,
            zone='quality_control'
        )


class TransportRobotNode(RobotAgentNode):
    """Transport Robot - material handling tasks."""
    
    def __init__(self, robot_id: str = 'robot_transport_001'):
        action_sequence = [
            {'resource': 'material_storage', 'action': 'read'},
            {'resource': 'assembly_station_A', 'action': 'transport'},
            {'resource': 'transport_log', 'action': 'write'},
            {'resource': 'material_storage', 'action': 'transport'},
        ]
        
        super().__init__(
            robot_id=robot_id,
            robot_type='transport_robot',
            action_sequence=action_sequence,
            zone='production'
        )


def main(args=None):
    """Main function - launch multiple robot agents."""
    rclpy.init(args=args)
    
    # Create multiple robot agents
    robots = [
        AssemblyRobotNode('robot_assembly_001'),
        CameraRobotNode('robot_camera_001'),
        TransportRobotNode('robot_transport_001'),
    ]
    
    # Use MultiThreadedExecutor for concurrent execution
    from rclpy.executors import MultiThreadedExecutor
    
    executor = MultiThreadedExecutor()
    
    for robot in robots:
        executor.add_node(robot)
    
    try:
        print('Robot agents started. Press Ctrl+C to stop.')
        executor.spin()
    except KeyboardInterrupt:
        print('\nShutting down robot agents...')
    finally:
        for robot in robots:
            robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
