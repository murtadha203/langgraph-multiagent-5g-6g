
import numpy as np
import collections
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from typing import Dict, List, Any

from src.simulation import NetworkSimulation
from src.orchestrator.orchestrator import StrategicOrchestrator
from src.orchestrator.schemas import ControlMode

# --- CONFIGURATION ---
CHECKPOINT_PATH = "models/ho_policy.pth"
LOG_DIR = "logs"
JSON_LOG_FILE = os.path.join(LOG_DIR, "StrategicOrchestrator_reasoning.json")
PLOT_DIR = "plots"

# --- SCENARIO DEFINITIONS ---
class StrategicOrchestratorScenario:
    def __init__(self, name: str, duration: int, trigger_step: int, reset_step: int):
        self.name = name
        self.duration = duration
        self.trigger_step = trigger_step
        self.reset_step = reset_step

    def apply(self, sim: NetworkSimulation, step: int):
        if step == self.trigger_step:
            self._trigger(sim)
        elif step == self.reset_step:
            sim.clear_anomalies()

    def _trigger(self, sim: NetworkSimulation):
        pass

class NormalOperation(StrategicOrchestratorScenario):
    def _trigger(self, sim: NetworkSimulation):
        pass # Do nothing

class URLLCStorm(StrategicOrchestratorScenario):
    def _trigger(self, sim: NetworkSimulation):
        sim.inject_traffic_surge(multiplier=5.0)

class EnergyBlackout(StrategicOrchestratorScenario):
    def _trigger(self, sim: NetworkSimulation):
        sim.inject_battery_drop(target_level_percent=10.0)

class CellFailure(StrategicOrchestratorScenario):
    def _trigger(self, sim: NetworkSimulation):
        # Fail the serving cell typically, or cell 1
        sim.inject_cell_failure(cell_id=0)

# --- METRIC EXPORTERS ---
def save_reasoning_log(history: List[Dict], filename: str):
    """Save structured JSON log for explainability audit."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[StrategicOrchestrator] Explainability Log saved to {filename}")

def plot_results(history: List[Dict], metrics_log: Dict[str, List], output_dir: str, scenario_name: str):
    """Generate performance visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    steps = range(len(metrics_log['step']))
    
    # 1. Pareto Trajectory (Reliability vs Energy)
    # Ideally a scatter plot where color = Mode
    plt.figure(figsize=(10, 6))
    
    # Map modes to colors
    mode_colors = {
        ControlMode.BALANCED.value: 'blue',
        ControlMode.SURVIVAL.value: 'red',
        ControlMode.GREEN.value: 'green'
    }
    
    # We need mode per step. 
    # History has mode per strategic interval. We need to expand it? 
    # Or just plot step-by-step metrics.
    # Let's plot RSRP vs Battery
    
    # Create step-wise mode array
    step_modes = []
    current_mode = ControlMode.BALANCED.value
    hist_idx = 0
    strategic_interval = 50 # Retrieve this from main config logic ideally
    
    for s in steps:
        # Find if we updated mode
        # This is approximate reconstruction
        if hist_idx < len(history) and s >= history[hist_idx]['step']:
            current_mode = history[hist_idx].get('proposed_mode', ControlMode.BALANCED.value)
            # Actually history log uses 'proposed_mode' key as per reasoning_log structure
            # Wait, reasoning_log has 'proposed_mode' which is Enum or Value?
            
            
            # Let's check reasoning_log append
            
            
            
            # But wait, history dict keys are: step, input_metrics, symbolic_state, proposed_mode, reasoning, shield_active, applied_params
            # So key is indeed 'proposed_mode'
            
            # Correction: Use proposed_mode
            current_mode = history[hist_idx].get('proposed_mode', ControlMode.BALANCED.value)
            hist_idx += 1
        step_modes.append(current_mode)
    
    colors = [mode_colors.get(m, 'gray') for m in step_modes]
    
    plt.scatter(metrics_log['battery'], metrics_log['rsrp'], c=colors, alpha=0.5, s=10)
    plt.xlabel("Battery Level (%)")
    plt.ylabel("RSRP (dBm)")
    plt.title(f"StrategicOrchestrator Pareto Trajectory ({scenario_name})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{scenario_name.lower()}_trajectory.png"))
    plt.close()
    
    # 2. Timeline
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('RSRP (dBm)', color='tab:blue')
    ax1.plot(steps, metrics_log['rsrp'], color='tab:blue', alpha=0.6, label='RSRP')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Battery (%)', color='tab:green')
    ax2.plot(steps, metrics_log['battery'], color='tab:green', alpha=0.6, label='Battery')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Add background color for Mode
    # Use fill_between for regions
    # Identify mode change points
    
    plt.title(f"StrategicOrchestrator System Response ({scenario_name})")
    plt.savefig(os.path.join(output_dir, f"{scenario_name.lower()}_timeline.png"))
    plt.close()

# --- MAIN LOOP ---
def run_StrategicOrchestrator_validation(
    scenario_type: str = "URLLC_STORM",
    duration: int = 2000,
    seed: int = 42
):
    print(f"--- [StrategicOrchestrator] Validation Run: {scenario_type} ---")
    
    # 1. Setup Environment
    sim = NetworkSimulation(seed=seed)
    context = sim.reset(service_type="VR")
    
    # 2. Setup Orchestrator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orchestrator = StrategicOrchestrator(CHECKPOINT_PATH, device=device)
    
    # 3. Setup Scenario
    if scenario_type == "URLLC_STORM":
        scenario = URLLCStorm("URLLC Storm", duration, trigger_step=500, reset_step=1500)
    elif scenario_type == "BLACKOUT":
        scenario = EnergyBlackout("Blackout", duration, trigger_step=500, reset_step=1500)
    elif scenario_type == "FAILURE":
        scenario = CellFailure("Cell Failure", duration, trigger_step=500, reset_step=1500)
    else:
        scenario = NormalOperation("Normal", duration, 0, 0)
        
    # 4. Metrics Buffers
    metrics_log = collections.defaultdict(list)
    reasoning_log = []
    
    # Buffer for Strategic Aggregation
    agg_buffer = collections.defaultdict(list)
    strategic_interval = 50
    
    # Initial Params
    active_params = {
        "handover_margin_db": 3.0,
        "time_to_trigger_s": 0.16
    }
    
    # Initial Strategic State
    current_mode_val = ControlMode.BALANCED.value
    
    pbar = tqdm(range(duration), desc="Validation Step")
    
    for step in pbar:
        # A. Apply Scenario Events
        scenario.apply(sim, step)
        
        
        if step > 0 and step % strategic_interval == 0:
            # Aggregate metrics from previous interval
            agg_metrics = {
                "avg_load": np.mean(agg_buffer['load']),
                "avg_velocity_kmh": np.mean(agg_buffer['velocity']) * 3.6,
                "ue_battery_percent": agg_buffer['battery'][-1],
                "avg_latency_ms": np.mean(agg_buffer['latency']) if agg_buffer['latency'] else 0.0,
                "rlf_rate": np.mean(agg_buffer['rlf_active']),
                "rsrp": np.min(agg_buffer['rsrp'])
            }
            
            # Reset buffer
            agg_buffer = collections.defaultdict(list)
            
            # 1. Run Strategic Cycle (LangGraph Step)
            # The orchestrator.step() now expects (metrics, observation)
            # and internally updates its LangGraph state.
            
            # We need a dummy observation for the Tactical Executor which is inside the Orchestrator
            # But in the validation loop, we might not want to execute a tactical action in the same call 
            # if we are just doing the strategic update.
            # However, the current API bundles them. Let's start by getting the observation.
            
            current_obs = orchestrator.tactical.agent.get_observation(context)
            
            # EXECUTE LANGGRAPH
            _ = orchestrator.step(agg_metrics, current_obs, decision_interval=1, verbose=True)
            # Note: decision_interval=1 because we already gated this execution with "if step % 50 == 0"
            
            # 2. Extract New Parameters (from Orchestrator State)
            # The params are stored in orchestrator.state["applied_params"] or directly on the agent?
            # The ConfiguratorNode applies them to the agent properties.
            # So reading from agent properties is correct.
            active_params["handover_margin_db"] = orchestrator.tactical.agent.handover_margin_db
            active_params["time_to_trigger_s"] = orchestrator.tactical.agent.time_to_trigger_s
            
            # 3. Log Explainability Data
            debug_info = orchestrator.get_debug_info()
            current_mode_val = debug_info.get('mode', ControlMode.BALANCED.value)
            
            reasoning_log.append({
                "step": step,
                "input_metrics": {k: float(v) for k, v in agg_metrics.items()},
                "symbolic_state": {k: str(v) for k, v in debug_info.get('symbolic_state', {}).items()},
                "proposed_mode": debug_info.get('mode', 'UNKNOWN'),
                "reasoning": debug_info.get('reasoning', ''),
                "shield_active": debug_info.get('shield_active', False),
                "applied_params": active_params.copy()
            })
            
            pbar.set_postfix({"Mode": current_mode_val, "RSRP": f"{context['serving_rsrp_dbm']:.1f}"})
            
        # C. Tactical Action (Deterministic)
        # We need to get observation again because state might have changed? 
        # Actually observation is from start of step.
        obs = orchestrator.tactical.agent.get_observation(context)

        # Deterministic Action from Frozen Agent
        action_idx = orchestrator.tactical.act(obs, deterministic=True)
        
        step_decision = {
            "handover_target": int(action_idx),
            "handover_margin_db": active_params["handover_margin_db"],
            "time_to_trigger_s": active_params["time_to_trigger_s"]
        }
        
        # D. Simulation Step
        context, info = sim.step(step_decision)
        
        # E. Live Metric Collection
        sim_metrics = {
            "load": sim.base_stations[sim.serving_cell_id].load_factor,
            "velocity": context['ue_speed_mps'],
            "battery": context['ue_battery_joules'] / 1000.0 * 100.0,
            "latency": info['task_info']['latency_s'] * 1000.0 if info.get('task_info') else 0.0,
            "rlf_active": 1.0 if sim.is_rlf_active else 0.0,
            "rsrp": context['serving_rsrp_dbm']
        }
        
        # Append to Agg buffer
        for k, v in sim_metrics.items():
            agg_buffer[k].append(v)
               
        # Append to Full Log
        metrics_log["step"].append(step)
        metrics_log["battery"].append(sim_metrics["battery"])
        metrics_log["rsrp"].append(sim_metrics["rsrp"])
        
        # Reconstruct mode history
        # We only updated `current_mode_val` at intervals
        metrics_log["mode"].append(current_mode_val)

    # --- POST-RUN ANALYSIS ---
    log_file = os.path.join(LOG_DIR, f"{scenario_type.lower()}_reasoning.json")
    save_reasoning_log(reasoning_log, log_file)
    plot_results(reasoning_log, metrics_log, PLOT_DIR, scenario_type)

def main():
    print("=== STARTING StrategicOrchestrator SYSTEM VALIDATION ===")
    
    # 1. Traffic Surge (Baseline + Traffic Logic)
    run_StrategicOrchestrator_validation("URLLC_STORM")
    
    # 2. Energy Constraints (Energy Logic)
    run_StrategicOrchestrator_validation("BLACKOUT")
    
    # 3. Hardware Faults (Reliability Logic)
    run_StrategicOrchestrator_validation("FAILURE")
    
    print("=== System Validation Complete. Check plots/ and logs/ ===")

if __name__ == "__main__":
    main()
