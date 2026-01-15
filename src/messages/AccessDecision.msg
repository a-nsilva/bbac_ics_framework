# BBAC Framework - AccessDecision Message
# Decision response for an access request

# Header with timestamp
std_msgs/Header header

# Original request reference
string request_id        # ID of the corresponding AccessRequest

# Decision
string decision          # 'grant', 'deny', or 'require_approval'
float64 confidence       # Overall confidence (0.0 to 1.0)
uint32 latency_ms        # Decision latency in milliseconds

# Layer decisions (hybrid approach)
string layer1_decision   # Rule engine decision
float64 layer1_confidence

string layer2_decision   # Behavioral analysis decision
float64 layer2_confidence

string layer3_decision   # ML anomaly detection decision
float64 layer3_confidence

# Explanation
string decision_reason   # Primary reason for the decision
string[] contributing_factors  # Additional factors (e.g., ["time_policy", "rbac"])

# Logging
bool logged              # Whether decision was logged
string log_id            # Log entry identifier
