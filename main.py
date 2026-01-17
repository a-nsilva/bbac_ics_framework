#!/usr/bin/env python3
"""
BBAC Framework - Main Entry Point
Interactive CLI for running experiments
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiments.run_experiment import ExperimentRunner
from experiments.ablation import AblationStudy
from experiments.baseline_comparison import BaselineComparison


def print_banner():
    """Print framework banner."""
    print("\n" + "="*70)
    print("BBAC FRAMEWORK - Behavior-Based Access Control")
    print("Experiment Suite for Industrial Multi-Agent Systems")
    print("="*70 + "\n")


def configure_dataset():
    """
    Configure dataset path.
    Returns the dataset path chosen by user.
    """
    print("\n" + "-"*70)
    print("DATASET CONFIGURATION")
    print("-"*70)
    
    # Check common paths
    common_paths = [
        "data/data_100k",
        "data/data100k",
        "data/data_300k",
        "data/data300k"
    ]
    
    print("\nAvailable datasets found:")
    available = []
    for i, path in enumerate(common_paths, 1):
        if Path(path).exists():
            print(f"  [{i}] {path}")
            available.append(path)
    
    if not available:
        print("  (No datasets found in common locations)")
    
    print(f"  [{len(available) + 1}] Enter custom path")
    print("-"*70)
    
    # Get user choice
    while True:
        choice = input("\nSelect dataset option: ").strip()
        
        # Custom path
        if choice == str(len(available) + 1):
            custom_path = input("Enter dataset path: ").strip()
            dataset_path = Path(custom_path)
            
            if not dataset_path.exists():
                print(f"❌ Path not found: {dataset_path}")
                retry = input("Try again? [Y/n]: ").strip().lower()
                if retry in ['n', 'no']:
                    return None
                continue
            
            # Verify required files
            required_files = ['bbac_train.csv', 'bbac_val.csv', 'bbac_test.csv', 'agents.json']
            missing = [f for f in required_files if not (dataset_path / f).exists()]
            
            if missing:
                print(f"❌ Missing required files: {', '.join(missing)}")
                retry = input("Try again? [Y/n]: ").strip().lower()
                if retry in ['n', 'no']:
                    return None
                continue
            
            print(f"✅ Dataset validated: {dataset_path}")
            return str(dataset_path)
        
        # Available path
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                dataset_path = available[idx]
                print(f"✅ Selected: {dataset_path}")
                return dataset_path
            else:
                print("Invalid option.")
        except ValueError:
            print("Invalid input. Enter a number.")


def print_menu():
    """Print main menu."""
    print("\n" + "-"*70)
    print("MAIN MENU")
    print("-"*70)
    print("[1] Run Ablation Study")
    print("[2] Run Baseline Comparison")
    print("[3] Run Custom Scenario")
    print("[9] Reconfigure Dataset Path")
    print("[0] Exit")
    print("-"*70)


def run_ablation_study(dataset_path: str):
    """Run ablation study with configured dataset."""
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    
    study = AblationStudy(dataset_path=dataset_path)
    
    print("\nThis will test:")
    print("  • L1 only (Rule Engine)")
    print("  • L1 + L2 (Rules + Behavioral)")
    print("  • L1 + L3 (Rules + ML)")
    print("  • Full BBAC (L1 + L2 + L3)")
    
    scenario = input("\nScenario [normal_operation/anomaly_detection]: ").strip()
    if not scenario:
        scenario = 'normal_operation'
    
    confirm = input(f"\nRun ablation on '{scenario}'? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        study.run_ablation(scenario=scenario)
        print(f"\n✓ Ablation study completed!")
    else:
        print("Cancelled.")


def run_baseline_comparison(dataset_path: str):
    """Run baseline comparison with configured dataset."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    comparison = BaselineComparison(dataset_path=dataset_path)
    
    print("\nThis will compare:")
    print("  • Pure Rule-based (RuBAC)")
    print("  • Pure Behavioral (Markov)")
    print("  • Pure ML (Isolation Forest)")
    print("  • Hybrid BBAC (Proposed)")
    
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        comparison.run_comparison()
        print("\n✓ Baseline comparison completed!")
    else:
        print("Cancelled.")


def run_custom_scenario(dataset_path: str):
    """Run custom scenario with configured dataset."""
    print("\n" + "="*70)
    print("CUSTOM SCENARIO")
    print("="*70)
    
    print("\nAvailable scenarios:")
    print("  1. normal_operation")
    print("  2. anomaly_detection")
    print("  3. emergency_response")
    print("  4. adversarial_attack")
    
    scenario = input("\nEnter scenario name: ").strip()
    duration = input("Duration (seconds) [60]: ").strip()
    duration = int(duration) if duration else 60
    
    confirm = input(f"\nRun '{scenario}' for {duration}s? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        runner = ExperimentRunner(dataset_path=dataset_path)
        runner.run_scenario(scenario, duration)
        print(f"\n✓ Scenario completed!")
    else:
        print("Cancelled.")


def main():
    """Main function."""
    print_banner()
    
    # Check ROS2 environment
    if 'ROS_DISTRO' not in os.environ:
        print("⚠️  WARNING: ROS2 environment not sourced!")
        print("Run: source /opt/ros/humble/setup.bash\n")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting.")
            return
    
    # Configure dataset path ONCE at startup
    dataset_path = configure_dataset()
    if not dataset_path:
        print("\n❌ No dataset configured. Exiting.")
        return
    
    # Main loop
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                print("\nExiting BBAC Framework. Goodbye!")
                break
            elif choice == '1':
                run_ablation_study(dataset_path)
            elif choice == '2':
                run_baseline_comparison(dataset_path)
            elif choice == '3':
                run_custom_scenario(dataset_path)
            elif choice == '9':
                # Reconfigure dataset
                new_path = configure_dataset()
                if new_path:
                    dataset_path = new_path
                else:
                    print("Dataset path not changed.")
            else:
                print("Invalid option. Please select 0-3 or 9.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nReturning to menu...")


if __name__ == '__main__':
    main()
