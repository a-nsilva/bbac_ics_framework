# BBAC Framework - Behavioral-Based Access Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/bbac_ics_framework)

## 📋 Overview

**BBAC (Behavioral-Based Access Control)** is a novel hybrid access control framework designed for Industrial Control Systems (ICS) that combines three layers of decision-making:

- **Layer 1: Rule-based Access Control** - Emergency rules, time policies, admin override, safety constraints
- **Layer 2: Behavioral Analysis (Markov Chains)** - Pattern learning through transition probabilities
- **Layer 3: ML Anomaly Detection (Isolation Forest)** - Adaptive learning and anomaly scoring

The framework achieves **sub-100ms latency** for real-time decision-making in robotics and ICS environments.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  BBAC Access Control System                 │
├──────────────────┬──────────────────┬───────────────────────┤
│   Layer 1:       │   Layer 2:       │   Layer 3:            │
│   Rule-based     │   Behavioral     │   ML Anomaly          │
│   - Emergency    │   - Markov Chain │   - Isolation Forest  │
│   - Time Policy  │   - Transitions  │   - Feature Extract   │
│   - Admin        │   - Sequences    │   - Anomaly Score     │
└──────────────────┴──────────────────┴───────────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │  ROS2 Communication Layer    │
            │  Topics: /access_requests    │
            │         /access_decisions    │
            │         /emergency_alerts    │
            └──────────────────────────────┘
                           ▼
       ┌─────────────┬──────────────┬─────────────┐
       │   Robot     │    Human     │    BBAC     │
       │   Agents    │    Agents    │  Controller │
       └─────────────┴──────────────┴─────────────┘
```

## 📁 Project Structure

```
bbac-framework/
├── README.md                          # This file
├── LICENSE                            # Apache 2.0 license
├── main.py                            # 
├── requirements.txt                   # Python dependencies
├── setup.sh                           # 
├── .devcontainer/
│   └── devcontainer.json             # GitHub Codespaces configuration
├── config/                           # Configuration files
│   ├── robot_profiles.yaml           # Agent behavioral profiles
│   ├── policies.json                 # Access control policies
│   └── emergency_rules.json          # Emergency scenarios
├── data/                             # Dataset directory
│   ├── README.md                     # Data documentation
│   ├── __init__.py
│   └── dataset_loader.py             # Load bbac_ics_dataset
├── results/                          # Results and outputs
│   ├── metrics/                      # Performance metrics
│   ├── plots/                        # Visualizations
│   └── ablation/                     # Ablation study results
└── src/
    ├── core/                         # Core BBAC modules
    │   ├── __init__.py
    │   ├── behavioral_analysis.py    # Layer 2: Markov Chain
    │   ├── ml_detection.py           # Layer 3: Isolation Forest
    │   └── rule_engine.py            # Layer 1: Rules
    ├── experiment/
    │   ├── __init__.py
    │   ├── ablation.py               #
    |   ├── baseline_comparison.py    #
    |   ├── metrics.py                #
    │   ├── run.py                    # 
    │   └── scenarios.py              #
    ├── messages/                     # ROS2 custom messages
    │   ├── AccessRequest.msg
    │   └── AccessDecision.msg
    └── ros_nodes/                    # ROS2 nodes
        ├── __init__.py
        ├── bbac_controller.py        # BBAC controller
        ├── robot_agents.py           # Robot agent simulation
        └── human_agents.py           # Human agent simulation
    
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ROS2 Humble
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bbac-framework.git
cd bbac-framework
git status

# Install packages
chmod +x setup.sh
./setup.sh

# Install dependencies
pip install -r requirements.txt

# Source ROS2
source /opt/ros/humble/setup.bash

# Clone the dataset (private repository)
git clone https://github.com/a-nsilva/bbac_ics_dataset.git data/bbac_ics_dataset
```

### Running Tests

```bash
# Minimal test (quick validation)
python src/tests/bbac_minimal_test.py

```

### Running ROS2 Nodes

```bash
# Terminal 1: Start BBAC Controller
ros2 run bbac_framework controller

# Terminal 2: Start Robot Agents
ros2 run bbac_framework robot_agents

# Terminal 3: Start Human Agents
ros2 run bbac_framework human_agents
```

## 📊 Dataset

The framework uses the **bbac_ics_dataset** which contains:

- Historical access logs from ICS environments
- Agent behavioral profiles (robots and humans)
- Normal and anomalous access patterns
- Temporal and contextual features

Dataset repository: `https://github.com/a-nsilva/bbac_ics_dataset.git`

## 🧪 Features

- ✅ **Real-time Decision Making** - Sub-100ms latency
- ✅ **Multi-layer Hybrid Approach** - Combines rules, behavior, and ML
- ✅ **ROS2 Integration** - Native support for robotic systems
- ✅ **Adaptive Learning** - Continuous model updates
- ✅ **Multi-agent Support** - Handles robots and humans differently
- ✅ **Emergency Handling** - Priority override mechanisms
- ✅ **Comprehensive Testing** - Minimal, complete, and ablation tests

## 📈 Performance Metrics

The framework tracks:

- **Decision Latency** - Response time per request
- **Accuracy** - Correct decisions vs total decisions
- **False Positive Rate** - Legitimate requests denied
- **False Negative Rate** - Malicious requests granted
- **Layer Contributions** - Individual layer impact

## 🔬 Research Paper

This framework is part of academic research on hybrid access control for ICS environments.

**Citation** (update when published):
```bibtex
@article{bbac2025,
  title={BBAC: A Hybrid Behavioral-Based Access Control Framework for Industrial Control Systems},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations, please contact: [your.email@institution.edu]

## 🙏 Acknowledgments

- ROS2 Community
- Industrial Control Systems security research community
- Dataset contributors

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Last Updated**: January 2025
