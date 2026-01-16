"""
BBAC Framework - Layer 1: Rule-based Access Control

This module implements the rule-based access control layer with:
- Emergency rules and overrides
- Time-based policies
- Role-based access control (RBAC)
- Safety constraints
- Resource constraints
"""

# 1. Biblioteca padrão
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Layer 1: Rule-based Access Control Engine
    
    Implements deterministic policy enforcement based on:
    - Emergency rules (highest priority)
    - Time-based policies
    - Role-based access control
    - Safety and resource constraints
    """
    
    def __init__(self, policies_path: str = "config/policies.json",
                 emergency_rules_path: str = "config/emergency_rules.json"):
        """
        Initialize the Rule Engine.
        
        Args:
            policies_path: Path to policies configuration file
            emergency_rules_path: Path to emergency rules configuration
        """
        self.policies_path = Path(policies_path)
        self.emergency_rules_path = Path(emergency_rules_path)
        
        self.policies = None
        self.emergency_rules = None
        self.emergency_state = {}  # Track active emergency states
        
        self.load_policies()
        self.load_emergency_rules()
        
        logger.info("RuleEngine initialized")
    
    def load_policies(self) -> bool:
        """
        Load access control policies from configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.policies_path, 'r') as f:
                self.policies = json.load(f)
            logger.info(f"Loaded policies from {self.policies_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
            return False
    
    def load_emergency_rules(self) -> bool:
        """
        Load emergency rules from configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.emergency_rules_path, 'r') as f:
                self.emergency_rules = json.load(f)
            logger.info(f"Loaded emergency rules from {self.emergency_rules_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading emergency rules: {e}")
            return False
    
    def evaluate_access_request(self, request: Dict) -> Tuple[str, float, Dict]:
        """
        Evaluate an access request against rule-based policies.
        
        Args:
            request: Dictionary containing:
                - agent_id: Agent identifier
                - agent_type: 'robot' or 'human'
                - agent_role: Role (supervisor, operator, assembly_robot, etc.)
                - resource_id: Target resource
                - action: Requested action (read, write, execute, etc.)
                - timestamp: Request timestamp
                - context: Additional context (zone, priority, etc.)
        
        Returns:
            Tuple of (decision, confidence, explanation)
            - decision: 'grant', 'deny', or 'require_approval'
            - confidence: 0.0 to 1.0 (rule-based is always 1.0 when matched)
            - explanation: Dictionary with reasoning details
        """
        explanation = {
            'layer': 'rule_engine',
            'rules_checked': [],
            'rules_matched': [],
            'decision_reason': None
        }
        
        # Priority 1: Check for active emergency states
        emergency_decision = self._check_emergency_rules(request, explanation)
        if emergency_decision:
            return emergency_decision
        
        # Priority 2: Check safety constraints
        safety_decision = self._check_safety_constraints(request, explanation)
        if safety_decision:
            return safety_decision
        
        # Priority 3: Check time-based policies
        time_decision = self._check_time_policies(request, explanation)
        if time_decision:
            return time_decision
        
        # Priority 4: Check role-based access control
        rbac_decision = self._check_role_based_access(request, explanation)
        if rbac_decision:
            return rbac_decision
        
        # Priority 5: Check resource constraints
        resource_decision = self._check_resource_constraints(request, explanation)
        if resource_decision:
            return resource_decision
        
        # Default: Deny if no rule explicitly grants access
        explanation['decision_reason'] = 'no_matching_rule_default_deny'
        return ('deny', 1.0, explanation)
    
    def _check_emergency_rules(self, request: Dict, explanation: Dict) -> Optional[Tuple]:
        """
        Check if any emergency rules override normal access control.
        
        Returns:
            Decision tuple if emergency rule applies, None otherwise
        """
        explanation['rules_checked'].append('emergency_rules')
        
        # Check for active emergency states
        for emergency_type, state in self.emergency_state.items():
            if state.get('active', False):
                rule = self.emergency_rules['emergency_rules'].get(emergency_type)
                if rule:
                    explanation['rules_matched'].append(f'emergency:{emergency_type}')
                    
                    # Check if agent is allowed under emergency
                    override = rule.get('override', {})
                    access_control = override.get('access_control', 'disabled')
                    
                    if access_control == 'disabled':
                        explanation['decision_reason'] = f'emergency_override:{emergency_type}'
                        return ('deny', 1.0, explanation)
                    
                    elif access_control == 'emergency_only':
                        if request.get('context', {}).get('emergency_personnel', False):
                            explanation['decision_reason'] = f'emergency_personnel_allowed:{emergency_type}'
                            return ('grant', 1.0, explanation)
                        else:
                            explanation['decision_reason'] = f'emergency_denied:{emergency_type}'
                            return ('deny', 1.0, explanation)
        
        return None
    
    def _check_safety_constraints(self, request: Dict, explanation: Dict) -> Optional[Tuple]:
        """
        Check safety constraints (human-robot proximity, etc.).
        
        Returns:
            Decision tuple if safety constraint applies, None otherwise
        """
        explanation['rules_checked'].append('safety_constraints')
        
        if self.policies is None:
            return None
        
        safety = self.policies.get('policies', {}).get('safety_constraints', {})
        
        # Check human-robot proximity
        if request.get('agent_type') == 'robot':
            proximity_rules = safety.get('human_robot_proximity', {})
            if proximity_rules.get('enabled', False):
                # Check if there are humans in proximity (from context)
                humans_nearby = request.get('context', {}).get('humans_in_proximity', [])
                if humans_nearby:
                    explanation['rules_matched'].append('human_robot_proximity')
                    explanation['decision_reason'] = 'safety_human_proximity_violation'
                    return ('deny', 1.0, explanation)
        
        return None
    
    def _check_time_policies(self, request: Dict, explanation: Dict) -> Optional[Tuple]:
        """
        Check time-based access policies.
        
        Returns:
            Decision tuple if time policy applies, None otherwise
        """
        explanation['rules_checked'].append('time_based_policies')
        
        if self.policies is None:
            return None
        
        timestamp = request.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        current_time = timestamp.time()
        current_day = timestamp.strftime('%A').lower()
        action = request.get('action', '')
        
        time_policies = self.policies.get('policies', {}).get('time_based', {})
        
        # Check work hours
        if current_day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            work_hours = time_policies.get('work_hours', {})
            if work_hours.get('enabled', False):
                start = datetime.strptime(work_hours['start_time'], '%H:%M').time()
                end = datetime.strptime(work_hours['end_time'], '%H:%M').time()
                
                if start <= current_time <= end:
                    actions = work_hours.get('actions', {})
                    action_policy = actions.get(action, 'deny')
                    
                    if action_policy == 'allow':
                        explanation['rules_matched'].append('work_hours_allow')
                        explanation['decision_reason'] = 'time_policy_work_hours_grant'
                        return ('grant', 1.0, explanation)
                    elif action_policy == 'deny':
                        explanation['rules_matched'].append('work_hours_deny')
                        explanation['decision_reason'] = 'time_policy_work_hours_deny'
                        return ('deny', 1.0, explanation)
        
        # Check weekend policies
        if current_day in ['saturday', 'sunday']:
            weekend = time_policies.get('weekend', {})
            if weekend.get('enabled', False):
                actions = weekend.get('actions', {})
                action_policy = actions.get(action, 'deny')
                
                explanation['rules_matched'].append('weekend_policy')
                
                if action_policy == 'allow':
                    explanation['decision_reason'] = 'time_policy_weekend_allow'
                    return ('grant', 1.0, explanation)
                elif action_policy == 'require_approval':
                    explanation['decision_reason'] = 'time_policy_weekend_approval'
                    return ('require_approval', 1.0, explanation)
                else:
                    explanation['decision_reason'] = 'time_policy_weekend_deny'
                    return ('deny', 1.0, explanation)
        
        return None
    
    def _check_role_based_access(self, request: Dict, explanation: Dict) -> Optional[Tuple]:
        """
        Check role-based access control (RBAC).
        
        Returns:
            Decision tuple if RBAC rule applies, None otherwise
        """
        explanation['rules_checked'].append('role_based_access')
        
        if self.policies is None:
            return None
        
        agent_role = request.get('agent_role', request.get('agent_type', 'unknown'))
        resource_id = request.get('resource_id', '')
        action = request.get('action', '')
        zone = request.get('context', {}).get('zone', '')
        
        rbac = self.policies.get('policies', {}).get('role_based', {})
        role_policy = rbac.get(agent_role, None)
        
        if role_policy:
            explanation['rules_matched'].append(f'role:{agent_role}')
            
            # Check resource access
            allowed_resources = role_policy.get('resources', [])
            resource_match = False
            
            for pattern in allowed_resources:
                if pattern == '*' or resource_id == pattern:
                    resource_match = True
                    break
                # Check wildcard patterns (e.g., "assembly_station_*")
                if pattern.endswith('*') and resource_id.startswith(pattern[:-1]):
                    resource_match = True
                    break
            
            if not resource_match:
                explanation['decision_reason'] = f'rbac_resource_mismatch:{agent_role}'
                return ('deny', 1.0, explanation)
            
            # Check action permission
            allowed_actions = role_policy.get('actions', [])
            if action not in allowed_actions:
                explanation['decision_reason'] = f'rbac_action_not_allowed:{agent_role}'
                return ('deny', 1.0, explanation)
            
            # Check zone access
            allowed_zones = role_policy.get('zones', [])
            if zone and zone not in allowed_zones:
                explanation['decision_reason'] = f'rbac_zone_mismatch:{agent_role}'
                return ('deny', 1.0, explanation)
            
            # All checks passed
            explanation['decision_reason'] = f'rbac_grant:{agent_role}'
            return ('grant', 1.0, explanation)
        
        return None
    
    def _check_resource_constraints(self, request: Dict, explanation: Dict) -> Optional[Tuple]:
        """
        Check resource-level constraints (concurrent access, critical resources).
        
        Returns:
            Decision tuple if resource constraint applies, None otherwise
        """
        explanation['rules_checked'].append('resource_constraints')
        
        if self.policies is None:
            return None
        
        resource_id = request.get('resource_id', '')
        
        constraints = self.policies.get('policies', {}).get('resource_constraints', {})
        
        # Check if resource is critical
        critical_resources = constraints.get('critical_resources', [])
        if resource_id in critical_resources:
            # Critical resources require special handling
            agent_role = request.get('agent_role', request.get('agent_type'))
            
            # Only high-priority roles can access critical resources
            if agent_role not in ['supervisor', 'admin']:
                explanation['rules_matched'].append('critical_resource_restriction')
                explanation['decision_reason'] = 'resource_critical_insufficient_role'
                return ('deny', 1.0, explanation)
        
        # Check concurrent access limits (would need state tracking in production)
        # For now, we just check if the limit is defined
        concurrent = constraints.get('concurrent_access', {})
        if resource_id in concurrent:
            max_concurrent = concurrent[resource_id]
            # In a full implementation, we would check current access count
            explanation['rules_matched'].append('concurrent_access_limit')
            # For now, allow but note the constraint exists
        
        return None
    
    def trigger_emergency(self, emergency_type: str, context: Dict = None) -> bool:
        """
        Trigger an emergency state.
        
        Args:
            emergency_type: Type of emergency (fire_alarm, gas_leak, etc.)
            context: Additional context about the emergency
        
        Returns:
            True if emergency was triggered successfully
        """
        if emergency_type not in self.emergency_rules.get('emergency_rules', {}):
            logger.error(f"Unknown emergency type: {emergency_type}")
            return False
        
        self.emergency_state[emergency_type] = {
            'active': True,
            'triggered_at': datetime.now(),
            'context': context or {}
        }
        
        logger.warning(f"EMERGENCY TRIGGERED: {emergency_type}")
        return True
    
    def clear_emergency(self, emergency_type: str) -> bool:
        """
        Clear an emergency state.
        
        Args:
            emergency_type: Type of emergency to clear
        
        Returns:
            True if emergency was cleared successfully
        """
        if emergency_type in self.emergency_state:
            self.emergency_state[emergency_type]['active'] = False
            logger.info(f"Emergency cleared: {emergency_type}")
            return True
        return False
    
    def get_active_emergencies(self) -> List[str]:
        """
        Get list of currently active emergencies.
        
        Returns:
            List of active emergency types
        """
        return [
            etype for etype, state in self.emergency_state.items()
            if state.get('active', False)
        ]


if __name__ == "__main__":
    # Test the rule engine
    engine = RuleEngine()
    
    # Test case 1: Normal robot access during work hours
    request1 = {
        'agent_id': 'robot_001',
        'agent_type': 'robot',
        'agent_role': 'assembly_robot',
        'resource_id': 'assembly_station_A',
        'action': 'write',
        'timestamp': datetime(2024, 1, 15, 10, 30),  # Monday 10:30 AM
        'context': {'zone': 'production'}
    }
    
    decision, confidence, explanation = engine.evaluate_access_request(request1)
    print(f"\nTest 1 - Robot access during work hours:")
    print(f"  Decision: {decision} (confidence: {confidence})")
    print(f"  Reason: {explanation['decision_reason']}")
    
    # Test case 2: Supervisor access to critical resource
    request2 = {
        'agent_id': 'human_001',
        'agent_type': 'human',
        'agent_role': 'supervisor',
        'resource_id': 'emergency_stop',
        'action': 'execute',
        'timestamp': datetime(2024, 1, 15, 14, 0),
        'context': {'zone': 'production'}
    }
    
    decision, confidence, explanation = engine.evaluate_access_request(request2)
    print(f"\nTest 2 - Supervisor accessing critical resource:")
    print(f"  Decision: {decision} (confidence: {confidence})")
    print(f"  Reason: {explanation['decision_reason']}")
    
    # Test case 3: Emergency scenario
    engine.trigger_emergency('fire_alarm')
    
    decision, confidence, explanation = engine.evaluate_access_request(request1)
    print(f"\nTest 3 - Robot access during fire emergency:")
    print(f"  Decision: {decision} (confidence: {confidence})")
    print(f"  Reason: {explanation['decision_reason']}")
    print(f"  Active emergencies: {engine.get_active_emergencies()}")
