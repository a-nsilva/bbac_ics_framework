#!/bin/bash
# BBAC Framework - Setup Script for GitHub Codespaces
# Ubuntu 22.04 + ROS2 Humble + Python 3.10
set -e

echo "=========================================="
echo "BBAC Framework Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
if [[ ! $PYTHON_VERSION == 3.10* ]]; then
    echo "WARNING: Python 3.10 required for ROS2 Humble"
    echo "Current version: $PYTHON_VERSION"
fi

# Install pip if not present
echo ""
echo "[2/5] Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found, installing..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
    echo "pip3 installed successfully"
else
    echo "pip3 already installed"
fi

# Source ROS2 (já vem instalado na imagem)
echo ""
echo "[3/5] Sourcing ROS2 environment..."
source /opt/ros/humble/setup.bash

# Add to bashrc if not already there
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "Added ROS2 to ~/.bashrc"
fi

# Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Python packages installed"

# Verify installations
echo ""
echo "[5/5] Verifying installations..."
echo "Checking numpy..."
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')"
echo "Checking scipy..."
python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')"
echo "Checking pandas..."
python3 -c "import pandas; print(f'  pandas: {pandas.__version__}')"
echo "Checking scikit-learn..."
python3 -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"
echo "Checking ROS2..."
python3 -c "import rclpy; print('  rclpy: installed')"

# Setup complete
echo ""
echo "=========================================="
echo "BBAC Framework Ready"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  python3 main.py"
echo ""
echo "=========================================="
