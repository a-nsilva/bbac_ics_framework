# BBAC Framework - AccessRequest Message
# Request for access to a resource

# Header with timestamp
std_msgs/Header header

# Agent information
string agent_id          # Unique identifier for the agent
string agent_type        # Type: 'robot' or 'human'
string agent_role        # Role: 'supervisor', 'operator', 'assembly_robot', etc.

# Resource information
string resource_id       # Target resource identifier
string action            # Requested action: 'read', 'write', 'execute', 'delete', etc.

# Context information
string zone              # Physical zone: 'production', 'maintenance', 'storage', etc.
float64 priority         # Request priority (0.0 to 10.0)
bool emergency           # Emergency flag
string[] context_data    # Additional context as key-value pairs (e.g., ["key:value"])

# Request tracking
string request_id        # Unique request identifier (UUID)
