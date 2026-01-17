"""
BBAC Framework - Human Agent Nodes

Simulates human agents with variable behavior patterns.
Humans have less predictable patterns compared to robots.
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


class HumanAgentNode(Node):
    """
    Simulated Human Agent with variable behavior.
    
    Humans have:
    - Variable action patterns
    - Less predictable timing
    - Occasional unusual requests
    """
    
    def __init__(self, human_id: str, role: str, 
                 typical_actions: list, zones: list):
        """
        Initialize human agent.
        
        Args:
            human_id: Unique human identifier
            role: Role (supervisor, operator, technician)
            typical_actions: List of typical actions
            zones: List of accessible zones
        """
        super().__init__(f'{human_id.lower()}_node')
        
        self.human_id = human_id
        self.role = role
        self.typical_actions = typical_actions
        self.zones = zones
        
        # State
        self.waiting_for_decision = False
        self.current_request_id = None
        self.actions_completed = 0
        self.actions_denied = 0
        
        # Behavior variability (10% chance of unusual behavior)
        self.unusual_behavior_chance = 0.1
        
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
        
        # Work cycle timer (variable timing: 2-8 seconds)
        timer_period = random.uniform(2.0, 8.0)    ## 4.0, 8.0
        self.work_timer = self.create_timer(timer_period, self.work_cycle)
        
        self.get_logger().info(
            f'Human Agent {human_id} ({role}) initialized'
        )
    
    def work_cycle(self):
        """Execute one work cycle with variable behavior."""
        if not self.waiting_for_decision:
            # Decide on action (mostly typical, sometimes unusual)
            if random.random() < self.unusual_behavior_chance:
                # Unusual behavior
                action_data = self._generate_unusual_request()
                self.get_logger().debug('Generating unusual request')
            else:
                # Typical behavior
                action_data = random.choice(self.typical_actions)
            
            # Select zone
            zone = random.choice(self.zones)
            
            # Create access request
            request_id = str(uuid.uuid4())
            request = {
                'request_id': request_id,
                'agent_id': self.human_id,
                'agent_type': 'human',
                'agent_role': self.role,
                'resource_id': action_data['resource'],
                'action': action_data['action'],
                'timestamp': datetime.now().isoformat(),
                'context': {
                    'zone': zone,
                    'priority': action_data.get('priority', 5.0),
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
    
    def _generate_unusual_request(self):
        """Generate an unusual/anomalous request."""
        unusual_actions = [
            {'resource': 'emergency_stop', 'action': 'execute', 'priority': 8.0},
            {'resource': 'safety_system', 'action': 'write', 'priority': 9.0},
            {'resource': 'admin_panel', 'action': 'read', 'priority': 7.0},
            {'resource': 'network_gateway', 'action': 'write', 'priority': 8.0},
        ]
        return random.choice(unusual_actions)
    
    def handle_decision(self, msg: String):
        """
        Handle access decision from BBAC Controller.
        
        Args:
            msg: Decision message
        """
        try:
            decision = json.loads(msg.data)
            
            # Check if decision is for this human's request
            if decision.get('request_id') == self.current_request_id:
                self.waiting_for_decision = False
                
                result = decision.get('decision')
                confidence = decision.get('confidence', 0.0)
                latency = decision.get('latency_ms', 0.0)
                
                if result == 'grant':
                    self.get_logger().info(
                        f'✓ GRANTED (confidence: {confidence:.2f}, '
                        f'latency: {latency:.2f}ms)'
                    )
                    self.actions_completed += 1
                    
                else:
                    explanation = decision.get('explanation', {})
                    reason = explanation.get('reason', 'unknown')
                    
                    self.get_logger().warn(
                        f'✗ DENIED: {reason} (confidence: {confidence:.2f})'
                    )
                    self.actions_denied += 1
        
        except Exception as e:
            self.get_logger().error(f'Error handling decision: {e}')


class SupervisorNode(HumanAgentNode):
    """Supervisor - high privileges, monitors all zones."""
    
    def __init__(self, human_id: str = 'human_supervisor_001'):
        typical_actions = [
            {'resource': 'production_schedule', 'action': 'read', 'priority': 6.0},
            {'resource': 'production_schedule', 'action': 'write', 'priority': 7.0},
            {'resource': 'quality_metrics', 'action': 'read', 'priority': 6.0},
            {'resource': 'robot_status', 'action': 'read', 'priority': 5.0},
            {'resource': 'access_control', 'action': 'override', 'priority': 9.0},
        ]
        
        zones = ['production', 'quality_control', 'maintenance', 'admin']
        
        super().__init__(
            human_id=human_id,
            role='supervisor',
            typical_actions=typical_actions,
            zones=zones
        )


class OperatorNode(HumanAgentNode):
    """Operator - production floor worker."""
    
    def __init__(self, human_id: str = 'human_operator_001'):
        typical_actions = [
            {'resource': 'assembly_station_A', 'action': 'read', 'priority': 5.0},
            {'resource': 'assembly_station_A', 'action': 'write', 'priority': 5.0},
            {'resource': 'assembly_station_B', 'action': 'read', 'priority': 5.0},
            {'resource': 'material_storage', 'action': 'read', 'priority': 4.0},
            {'resource': 'production_log', 'action': 'write', 'priority': 5.0},
        ]
        
        zones = ['production', 'quality_control']
        
        super().__init__(
            human_id=human_id,
            role='operator',
            typical_actions=typical_actions,
            zones=zones
        )


class TechnicianNode(HumanAgentNode):
    """Technician - maintenance tasks."""
    
    def __init__(self, human_id: str = 'human_technician_001'):
        typical_actions = [
            {'resource': 'robot_assembly_001', 'action': 'maintenance', 'priority': 7.0},
            {'resource': 'robot_camera_001', 'action': 'read', 'priority': 6.0},
            {'resource': 'diagnostic_tools', 'action': 'execute', 'priority': 7.0},
            {'resource': 'maintenance_log', 'action': 'write', 'priority': 6.0},
            {'resource': 'equipment_status', 'action': 'read', 'priority': 5.0},
        ]
        
        zones = ['production', 'maintenance']
        
        # Higher unusual behavior chance (maintenance is less predictable)
        super().__init__(
            human_id=human_id,
            role='technician',
            typical_actions=typical_actions,
            zones=zones
        )
        self.unusual_behavior_chance = 0.15


def main(args=None):
    """Main function - launch multiple human agents."""
    rclpy.init(args=args)
    
    # Create multiple human agents
    humans = [
        SupervisorNode('human_supervisor_001'),
        OperatorNode('human_operator_001'),
        TechnicianNode('human_technician_001'),
    ]
    
    # Use MultiThreadedExecutor for concurrent execution
    from rclpy.executors import MultiThreadedExecutor
    
    executor = MultiThreadedExecutor()
    
    for human in humans:
        executor.add_node(human)
    
    try:
        print('Human agents started. Press Ctrl+C to stop.')
        executor.spin()
    except KeyboardInterrupt:
        print('\nShutting down human agents...')
    finally:
        for human in humans:
            human.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
