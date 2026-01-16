"""
BBAC Framework - Scenario Definitions
Predefined scenarios for experiments
"""


class ScenarioManager:
    """Manages experiment scenarios."""
    
    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios = {
            'normal_operation': self._normal_operation(),
            'anomaly_detection': self._anomaly_detection(),
            'emergency_response': self._emergency_response(),
            'adversarial_attack': self._adversarial_attack(),
        }
    
    def get_scenario(self, name: str) -> dict:
        """
        Get scenario configuration.
        
        Args:
            name: Scenario name
            
        Returns:
            Scenario configuration dictionary
        """
        if name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {name}")
        return self.scenarios[name].copy()
    
    def list_scenarios(self) -> list:
        """List available scenarios."""
        return list(self.scenarios.keys())
    
    def _normal_operation(self) -> dict:
        """
        Normal operation scenario.
        
        Tests system under typical conditions:
        - Predictable robot behavior
        - Normal human patterns
        - No anomalies
        - No emergencies
        """
        return {
            'name': 'Normal Operation',
            'description': 'System under normal operating conditions',
            'agents': {
                'robots': 3,  # Assembly, Camera, Transport
                'humans': 2,  # Operator, Supervisor
            },
            'layers_enabled': {
                'rule_engine': True,
                'behavioral_analysis': True,
                'ml_detection': True
            },
            'modifications': {
                'inject_anomalies': False,
                'trigger_emergency': None,
                'attack_simulation': False
            },
            'expected_behavior': {
                'grant_rate': (0.90, 1.0),  # 90-100%
                'avg_latency': (0, 10),     # 0-10ms
                'anomalies_detected': 0
            }
        }
    
    def _anomaly_detection(self) -> dict:
        """
        Anomaly detection scenario.
        
        Tests system's ability to detect abnormal behavior:
        - Inject unusual access patterns
        - Out-of-sequence requests
        - Unusual timing
        - Unexpected resource access
        """
        return {
            'name': 'Anomaly Detection',
            'description': 'System detecting abnormal access patterns',
            'agents': {
                'robots': 3,
                'humans': 3,  # Include technician for variable behavior
            },
            'layers_enabled': {
                'rule_engine': True,
                'behavioral_analysis': True,
                'ml_detection': True
            },
            'modifications': {
                'inject_anomalies': True,
                'anomaly_types': [
                    'unusual_sequence',
                    'timing_anomaly',
                    'unexpected_resource',
                    'high_frequency_access'
                ],
                'anomaly_rate': 0.15,  # 15% of requests
                'trigger_emergency': None,
                'attack_simulation': False
            },
            'expected_behavior': {
                'grant_rate': (0.70, 0.90),  # 70-90% (some anomalies denied)
                'avg_latency': (0, 15),       # Slightly higher
                'anomalies_detected': (3, 10) # Should detect several
            }
        }
    
    def _emergency_response(self) -> dict:
        """
        Emergency response scenario.
        
        Tests emergency handling:
        - Trigger emergency mid-experiment
        - System should adapt policies
        - Emergency access granted
        - Non-essential access restricted
        """
        return {
            'name': 'Emergency Response',
            'description': 'System handling emergency situations',
            'agents': {
                'robots': 3,
                'humans': 3,  # Need supervisor for emergency override
            },
            'layers_enabled': {
                'rule_engine': True,
                'behavioral_analysis': True,
                'ml_detection': True
            },
            'modifications': {
                'inject_anomalies': False,
                'trigger_emergency': 'fire_alarm',  # or 'gas_leak', 'power_failure'
                'emergency_delay': 20,  # Trigger after 20 seconds
                'attack_simulation': False
            },
            'expected_behavior': {
                'grant_rate': (0.50, 0.80),  # Reduced during emergency
                'avg_latency': (0, 20),       # May be higher
                'emergency_accesses': (1, 5),  # Emergency overrides
                'escalation_triggered': True
            }
        }
    
    def _adversarial_attack(self) -> dict:
        """
        Adversarial attack scenario.
        
        Tests system against malicious behavior:
        - Privilege escalation attempts
        - Unauthorized resource access
        - Rapid-fire requests (DoS)
        - Session hijacking simulation
        """
        return {
            'name': 'Adversarial Attack',
            'description': 'System defending against malicious behavior',
            'agents': {
                'robots': 2,  # Normal robots
                'humans': 2,  # Normal humans
                'adversarial': 1  # Malicious agent
            },
            'layers_enabled': {
                'rule_engine': True,
                'behavioral_analysis': True,
                'ml_detection': True
            },
            'modifications': {
                'inject_anomalies': True,
                'attack_simulation': True,
                'attack_types': [
                    'privilege_escalation',
                    'unauthorized_access',
                    'dos_attempt',
                    'session_hijack'
                ],
                'attack_intensity': 'high',  # low, medium, high
                'trigger_emergency': None
            },
            'expected_behavior': {
                'grant_rate': (0.60, 0.85),  # Lower due to attacks
                'avg_latency': (0, 25),       # May spike during attacks
                'attacks_blocked': (5, 15),   # Should block attacks
                'false_positives': (0, 2)     # Minimize false positives
            }
        }
    
    def get_scenario_description(self, name: str) -> str:
        """Get detailed scenario description."""
        scenario = self.get_scenario(name)
        
        desc = f"\n{'='*70}\n"
        desc += f"SCENARIO: {scenario['name']}\n"
        desc += f"{'='*70}\n"
        desc += f"Description: {scenario['description']}\n\n"
        
        desc += "Configuration:\n"
        desc += f"  Agents: {scenario['agents']}\n"
        desc += f"  Layers: {scenario['layers_enabled']}\n"
        
        if scenario['modifications'].get('inject_anomalies'):
            desc += "  Anomalies: YES\n"
        
        if scenario['modifications'].get('trigger_emergency'):
            desc += f"  Emergency: {scenario['modifications']['trigger_emergency']}\n"
        
        if scenario['modifications'].get('attack_simulation'):
            desc += "  Attack Simulation: YES\n"
        
        desc += "\nExpected Behavior:\n"
        for key, value in scenario['expected_behavior'].items():
            desc += f"  {key}: {value}\n"
        
        desc += f"{'='*70}\n"
        
        return desc


# Scenario configurations for specific tests
BASELINE_SCENARIOS = {
    'rule_only': {
        'name': 'Rule-based Only',
        'layers_enabled': {
            'rule_engine': True,
            'behavioral_analysis': False,
            'ml_detection': False
        }
    },
    'rule_behavioral': {
        'name': 'Rule + Behavioral',
        'layers_enabled': {
            'rule_engine': True,
            'behavioral_analysis': True,
            'ml_detection': False
        }
    },
    'rule_ml': {
        'name': 'Rule + ML',
        'layers_enabled': {
            'rule_engine': True,
            'behavioral_analysis': False,
            'ml_detection': True
        }
    },
    'full_bbac': {
        'name': 'Full BBAC',
        'layers_enabled': {
            'rule_engine': True,
            'behavioral_analysis': True,
            'ml_detection': True
        }
    }
}


def main():
    """Demo scenario manager."""
    manager = ScenarioManager()
    
    print("Available Scenarios:")
    print("="*70)
    for name in manager.list_scenarios():
        scenario = manager.get_scenario(name)
        print(f"\n{name}:")
        print(f"  {scenario['description']}")
    
    print("\n" + "="*70)
    print("\nDetailed Scenario Information:")
    print(manager.get_scenario_description('normal_operation'))


if __name__ == '__main__':
    main()
