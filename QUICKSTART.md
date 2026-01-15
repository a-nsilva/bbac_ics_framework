# 🚀 BBAC Framework - Quick Start Guide

## ⚡ GitHub Codespaces Setup (5 minutes)

### Step 1: Install Dependencies

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (installs ROS2 Humble + Python packages)
./setup.sh
```

### Step 2: Run Minimal Test

```bash
# Test all 3 layers + ROS2 integration (30 seconds)
python3 src/tests/bbac_minimal_test.py

# Or custom duration
python3 src/tests/bbac_minimal_test.py --duration 60
```

**Expected output:**
```
✓ Layer 1 (Rules): Working
✓ Layer 2 (Behavioral): Working
✓ Layer 3 (ML): Working
✓ ROS2 Integration: Working
✓ Latency target achieved (<100ms)
```

---

## 🧪 Testing Individual Layers

### Layer 1: Rule Engine
```bash
python3 src/bbac_core/rule_engine.py
```

### Layer 2: Behavioral Analysis (Markov)
```bash
python3 src/bbac_core/behavioral_analysis.py
```

### Layer 3: ML Detection (Isolation Forest)
```bash
python3 src/bbac_core/ml_detection.py
```

---

## 🤖 Running with ROS2

Open **3 terminals**:

### Terminal 1: BBAC Controller
```bash
source /opt/ros/humble/setup.bash
python3 src/ros_nodes/bbac_controller.py
```

### Terminal 2: Robot Agents
```bash
source /opt/ros/humble/setup.bash
python3 src/ros_nodes/robot_agents.py
```

### Terminal 3: Human Agents
```bash
source /opt/ros/humble/setup.bash
python3 src/ros_nodes/human_agents.py
```

**Monitor topics:**
```bash
# In another terminal
ros2 topic list
ros2 topic echo /access_requests
ros2 topic echo /access_decisions
```

---

## 📊 Training Models with Dataset

### Option 1: Use bbac_ics_dataset (when available)

```bash
# Clone dataset
cd data
git clone https://github.com/a-nsilva/bbac_ics_dataset.git
cd ..

# Train models
python3 -c "
from src.data.dataset_loader import DatasetLoader
from src.bbac_core.behavioral_analysis import BehavioralAnalyzer
from src.bbac_core.ml_detection import MLAnomalyDetector

# Load data
loader = DatasetLoader('data/bbac_ics_dataset')
loader.load_all()

# Train Layer 2
behavioral = BehavioralAnalyzer(order=2)
behavioral.train(loader.access_logs)
behavioral.save_models('models/behavioral.json')

# Train Layer 3
ml_detector = MLAnomalyDetector()
ml_detector.train(loader.normal_patterns)
ml_detector.save_models('models/ml_detector.pkl')

print('Models trained successfully!')
"
```

### Option 2: Use Synthetic Data (for testing)

```bash
python3 src/data/dataset_loader.py
# Generates sample data automatically
```

---

## 🔧 Troubleshooting

### Issue: Python version mismatch
```bash
python3 --version  # Must be 3.10.x

# If wrong version, install Python 3.10
sudo apt install python3.10 python3.10-venv
```

### Issue: ROS2 not found
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Add to bashrc permanently
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Issue: Package import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify installations
python3 -c "import numpy, pandas, sklearn; print('OK')"
```

### Issue: numpy version conflicts
```bash
# Critical: numpy must be <2.0 for sklearn 1.3.0
pip install "numpy<2.0" --force-reinstall
```

---

## 📈 Performance Metrics

The system tracks:
- **Decision Latency**: Time per access decision (target: <100ms)
- **Accuracy**: Correct decisions / Total decisions
- **Grant Rate**: Percentage of granted requests
- **Layer Contributions**: Individual layer performance

View statistics in controller logs (auto-published every 10 seconds).

---

## 🎯 Next Steps

1. ✅ Run minimal test
2. ✅ Verify <100ms latency
3. ✅ Test with synthetic data
4. 📝 Train with real dataset (bbac_ics_dataset)
5. 📊 Run ablation study
6. 📄 Collect metrics for paper

---

## 📚 Configuration

Edit these files to customize:

- **`config/policies.json`**: Access control policies
- **`config/emergency_rules.json`**: Emergency scenarios
- **`config/robot_profiles.yaml`**: Agent behavioral profiles

---

## 🐛 Common Errors

**Error:** `ModuleNotFoundError: No module named 'rclpy'`  
**Fix:** Install ROS2 via apt (not pip): `sudo apt install ros-humble-desktop`

**Error:** `numpy has no attribute 'bool'`  
**Fix:** Downgrade numpy: `pip install "numpy<2.0"`

**Error:** `sklearn version mismatch`  
**Fix:** Install exact version: `pip install scikit-learn==1.3.0`

---

## ✅ Verification Checklist

- [ ] Python 3.10.x installed
- [ ] ROS2 Humble installed
- [ ] All Python packages installed
- [ ] Minimal test passes
- [ ] Latency <100ms achieved
- [ ] All 3 layers working

---

**Need help?** Check the main [README.md](README.md) for detailed documentation.
