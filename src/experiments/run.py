"""
BBAC Framework - Experiment Runner
Main engine for running experiments and collecting metrics
"""

# 1. Biblioteca padrão
import json
import os
import time
from datetime import datetime
from pathlib import Path

# 2. Bibliotecas de terceiros
import rclpy
from rclpy.executors import MultiThreadedExecutor

# 3. Imports locais
from experiments.metrics import MetricsCollector
from experiments.scenarios import ScenarioManager
from ros_nodes.controller import BBACController
from ros_nodes.human_agents import SupervisorNode, OperatorNode, TechnicianNode
from ros_nodes.robot_agents import (
    AssemblyRobotNode,
    CameraRobotNode,
    TransportRobotNode,
)


class ExperimentRunner:
    """
    Main experiment runner.
    Manages scenarios, agents, and metrics collection.
    """
    
    def __init__(self, dataset_path: str, results_base_dir='results'):
        """Initialize experiment runner."""
        self.dataset_path = dataset_path  # ← Armazenar
        self.results_base_dir = results_base_dir
        self.results_dir = None
        self.scenario_manager = ScenarioManager()
        self.metrics_collector = MetricsCollector()
        
        # Create results directory
        self._setup_results_dir()
    
    def _setup_results_dir(self):
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(self.results_base_dir, f'experiment_{timestamp}')
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        Path(os.path.join(self.results_dir, 'metrics')).mkdir(exist_ok=True)
        Path(os.path.join(self.results_dir, 'plots')).mkdir(exist_ok=True)
        Path(os.path.join(self.results_dir, 'logs')).mkdir(exist_ok=True)
    
    def run_scenario(self, scenario_name: str, duration: int = 60, 
                     config: dict = None):
        """
        Run a single scenario.
        
        Args:
            scenario_name: Name of scenario to run
            duration: Duration in seconds
            config: Optional configuration override
        """
        print(f"\n{'='*70}")
        print(f"RUNNING SCENARIO: {scenario_name}")
        print(f"{'='*70}")
        print(f"Duration: {duration}s")
        print(f"Results: {self.results_dir}")
        print(f"{'='*70}\n")
        
        # Get scenario configuration
        scenario_config = self.scenario_manager.get_scenario(scenario_name)
        if config:
            scenario_config.update(config)
        
        # Initialize ROS2
        rclpy.init()
        
        try:
            # Create nodes
            print("[1/4] Initializing BBAC Controller...")
            controller = BBACController()
            
            # Apply scenario configuration to controller
            if 'layers_enabled' in scenario_config:
                self._configure_layers(controller, scenario_config['layers_enabled'])
            
            print("[2/4] Initializing Agents...")
            agents = self._create_agents(scenario_config)
            
            print("[3/4] Starting Executor...")
            executor = MultiThreadedExecutor()
            executor.add_node(controller)
            for agent in agents:
                executor.add_node(agent)
            
            # Start metrics collection
            self.metrics_collector.start_collection(controller)
            
            # Apply scenario modifications (anomalies, emergencies, etc.)
            if 'modifications' in scenario_config:
                self._apply_scenario_modifications(
                    controller, agents, scenario_config['modifications']
                )
            
            print("[4/4] Running Experiment...")
            print(f"{'='*70}\n")
            
            # Run for specified duration
            import threading
            executor_thread = threading.Thread(target=executor.spin, daemon=True)
            executor_thread.start()
            
            # Monitor progress
            self._monitor_experiment(duration, controller)
            
            # Stop
            print(f"\n{'='*70}")
            print("Stopping experiment...")
            executor.shutdown()
            
            # Collect final metrics
            print("Collecting metrics...")
            metrics = self.metrics_collector.stop_collection(controller)
            
            # Save results
            print("Saving results...")
            self._save_results(scenario_name, metrics, scenario_config)
            
            # Generate plots
            print("Generating plots...")
            self._generate_plots(scenario_name, metrics)
            
            # Cleanup
            controller.destroy_node()
            for agent in agents:
                agent.destroy_node()
            
            print(f"{'='*70}")
            print("✓ Scenario completed successfully!")
            self._print_summary(metrics)
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n❌ Error during experiment: {e}")
            import traceback
            traceback.print_exc()
        finally:
            rclpy.shutdown()
    
    def run_all_scenarios(self):
        """Run all predefined scenarios."""
        scenarios = [
            'normal_operation',
            'anomaly_detection', 
            'emergency_response',
            'adversarial_attack'
        ]
        
        print(f"\n{'='*70}")
        print("RUNNING ALL SCENARIOS")
        print(f"{'='*70}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Estimated duration: {len(scenarios) * 2} minutes")
        print(f"{'='*70}\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[Scenario {i}/{len(scenarios)}]")
            self.run_scenario(scenario, duration=60)
            time.sleep(2)  # Brief pause between scenarios
        
        print(f"\n{'='*70}")
        print("✓ ALL SCENARIOS COMPLETED")
        print(f"Results directory: {self.results_dir}")
        print(f"{'='*70}\n")
    
    def _create_agents(self, config: dict) -> list:
        """Create agent nodes based on configuration."""
        agents = []
        
        agent_config = config.get('agents', {})
        
        # Robots
        num_robots = agent_config.get('robots', 3)
        if num_robots >= 1:
            agents.append(AssemblyRobotNode('robot_assembly_001'))
        if num_robots >= 2:
            agents.append(CameraRobotNode('robot_camera_001'))
        if num_robots >= 3:
            agents.append(TransportRobotNode('robot_transport_001'))
        
        # Humans
        num_humans = agent_config.get('humans', 3)
        if num_humans >= 1:
            agents.append(OperatorNode('human_operator_001'))
        if num_humans >= 2:
            agents.append(SupervisorNode('human_supervisor_001'))
        if num_humans >= 3:
            agents.append(TechnicianNode('human_technician_001'))
        
        return agents
    
    def _configure_layers(self, controller: BBACController, layers_config: dict):
        """Configure which layers are enabled."""
        # This would modify the controller's layer usage
        # For now, just log the configuration
        print(f"  Layer configuration: {layers_config}")
    
    def _apply_scenario_modifications(self, controller, agents, modifications: dict):
        """Apply scenario-specific modifications."""
        # Handle anomaly injection
        if 'inject_anomalies' in modifications:
            print(f"  Will inject anomalies: {modifications['inject_anomalies']}")
        
        # Handle emergency triggers
        #if 'trigger_emergency' in modifications:
        #    emergency_type = modifications['trigger_emergency']
        #    delay = modifications.get('emergency_delay', 30)
        #    print(f"  Will trigger emergency '{emergency_type}' after {delay}s")
        #    
        #    import threading
        #    def trigger_later():
        #        time.sleep(delay)
        #        controller.trigger_emergency(emergency_type)
        #    
        #    threading.Thread(target=trigger_later, daemon=True).start()

        if 'trigger_emergency' in modifications:
            emergency_type = modifications['trigger_emergency']
            
            if emergency_type is not None:  # Validar aqui
                delay = modifications.get('emergency_delay', 30)
                print(f"  Will trigger emergency '{emergency_type}' after {delay}s")
                
                import threading
                def trigger_later():
                    time.sleep(delay)
                    controller.trigger_emergency(emergency_type)
                    
                threading.Thread(target=trigger_later, daemon=True).start()
    
    def _monitor_experiment(self, duration: int, controller):
        """Monitor experiment progress."""
        start_time = time.time()
        last_update = 0
        
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            
            # Print progress every 10 seconds
            if elapsed - last_update >= 10:
                remaining = duration - elapsed
                progress = (elapsed / duration) * 100
                print(f"Progress: {progress:.0f}% | "
                      f"Elapsed: {elapsed}s | "
                      f"Remaining: {remaining}s | "
                      f"Requests: {controller.requests_processed}")
                last_update = elapsed
            
            time.sleep(1)
    
    def _save_results(self, scenario_name: str, metrics: dict, config: dict):
        """Save experiment results."""
        results = {
            'scenario': scenario_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'metrics': metrics
        }
        
        filename = os.path.join(
            self.results_dir, 
            'metrics',
            f'{scenario_name}_results.json'
        )
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _generate_plots(self, scenario_name: str, metrics: dict):
        """Generate plots for results."""
        try:
            import matplotlib.pyplot as plt
            
            # Latency distribution
            if 'latency_samples' in metrics:
                plt.figure(figsize=(10, 6))
                plt.hist(metrics['latency_samples'], bins=50, edgecolor='black')
                plt.xlabel('Latency (ms)')
                plt.ylabel('Frequency')
                plt.title(f'Latency Distribution - {scenario_name}')
                plt.axvline(100, color='r', linestyle='--', label='Target (100ms)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                filename = os.path.join(
                    self.results_dir,
                    'plots',
                    f'{scenario_name}_latency.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Decision distribution
            if 'decisions' in metrics:
                decisions = metrics['decisions']
                plt.figure(figsize=(8, 6))
                plt.bar(decisions.keys(), decisions.values(), color=['green', 'red'])
                plt.xlabel('Decision')
                plt.ylabel('Count')
                plt.title(f'Decision Distribution - {scenario_name}')
                plt.grid(True, alpha=0.3, axis='y')
                
                filename = os.path.join(
                    self.results_dir,
                    'plots',
                    f'{scenario_name}_decisions.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")
    
    def _print_summary(self, metrics: dict):
        """Print experiment summary."""
        print("\nExperiment Summary:")
        print(f"  Total Requests: {metrics.get('total_requests', 0)}")
        print(f"  Granted: {metrics.get('granted', 0)}")
        print(f"  Denied: {metrics.get('denied', 0)}")
        print(f"  Grant Rate: {metrics.get('grant_rate', 0):.1f}%")
        print(f"  Avg Latency: {metrics.get('avg_latency', 0):.2f}ms")
        print(f"  Max Latency: {metrics.get('max_latency', 0):.2f}ms")
        print(f"  Min Latency: {metrics.get('min_latency', 0):.2f}ms")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BBAC Experiment Runner')
    parser.add_argument('--scenario', type=str, default='normal_operation',
                       help='Scenario to run')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    runner.run_scenario(args.scenario, duration=args.duration)


if __name__ == '__main__':
    main()
