#!/bin/bash
# BBAC Framework - Setup Script for GitHub Codespaces
# Ubuntu 22.04 + ROS2 Humble + Python 3.10

set -e

echo "=========================================="
echo "BBAC Framework Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION == 3.10* ]]; then
    echo "WARNING: Python 3.10 required for ROS2 Humble"
    echo "Current version: $PYTHON_VERSION"
fi

# Install ROS2 Humble
echo ""
echo "[2/6] Installing ROS2 Humble..."

if [ ! -d "/opt/ros/humble" ]; then
    echo "Installing ROS2 Humble Hawksbill..."
    
    # Setup locale
    sudo apt update && sudo apt install locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    
    # Setup sources
    sudo apt install software-properties-common
    sudo add-apt-repository universe
    
    sudo apt update && sudo apt install curl -y
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    # Install ROS2 packages
    sudo apt update
    sudo apt upgrade -y
    sudo apt install ros-humble-desktop python3-argcomplete -y
    sudo apt install ros-dev-tools -y
    
    echo "ROS2 Humble installed successfully"
else
    echo "ROS2 Humble already installed"
fi

# Source ROS2
echo ""
echo "[3/6] Sourcing ROS2 environment..."
source /opt/ros/humble/setup.bash

# Add to bashrc if not already there
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "Added ROS2 to ~/.bashrc"
fi

# Install Python dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found, installing..."
    apt-get update
    apt-get install -y python3-pip
fi
pip install --upgrade pip
pip install -r requirements.txt

echo "Python packages installed"

# Verify installations
echo ""
echo "[5/6] Verifying installations..."

echo "Checking numpy..."
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"

echo "Checking pandas..."
python3 -c "import pandas; print(f'  pandas: {pandas.__version__}')"

echo "Checking scikit-learn..."
python3 -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"

echo "Checking tensorflow..."
python3 -c "import tensorflow; print(f'  tensorflow: {tensorflow.__version__}')" || echo "  tensorflow: optional, not critical"

echo "Checking ROS2..."
python3 -c "import rclpy; print('  rclpy: installed')"

# Setup complete
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "BBAC Framework Ready"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  1. Run minimal test:"
echo "     python3 test/minimal.py"
echo ""
echo "  2. Run individual components:"
echo "     python3 src/core/rule_engine.py"
echo "     python3 src/core/behavioral_analysis.py"
echo "     python3 src/core/ml_detection.py"
echo ""
echo "  3. Run with ROS2 (3 terminals):"
echo "     Terminal 1: python3 src/ros_nodes/bbac_controller.py"
echo "     Terminal 2: python3 src/ros_nodes/robot_agents.py"
echo "     Terminal 3: python3 src/ros_nodes/human_agents.py"
echo ""
echo "=========================================="
