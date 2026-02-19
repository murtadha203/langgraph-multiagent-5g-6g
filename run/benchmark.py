"""
Comprehensive Benchmark Suite for Thesis Defense
Tests AI robustness across 4 critical dimensions:
1. Congestion (Network Context)
2. Mobility (User Context)  
3. Application Load (QoS Context)
4. Cell-Edge Distance (Physics Context)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulation import NetworkSimulation
from src.agents.mec_agent_ppo import MECAgentPPO
from src.agents.ho_agent_ppo import HOAgentPPO

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
NUM_EPISODES_PER_POINT = 25
STEPS_PER_EPISODE = 200
MEC_CHECKPOINT = os.path.join(project_root, "models", "mec_policy.pth")
HO_CHECKPOINT = os.path.join(project_root, "models", "ho_policy.pth")

# Test Parameters
CONGESTION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
MOBILITY_SPEEDS = [0, 5, 10, 15, 20, 25, 30]  # m/s (0 to highway speed)
APPLICATION_SIZES = [1e6, 3e6, 5e6, 7e6, 10e6, 12e6, 15e6]  # bits (1 to 15 Mbits)
CELLEDGE_DISTANCES = [50, 100, 150, 200, 250, 300, 400, 500]  # meters from tower

RESULTS_DIR = os.path.join(project_root, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# BASELINE AGENTS (Internal Logic Simulated)
# -------------------------------------------------------------------------
# We use the simulation's built-in baseline by passing None for decisions

# -------------------------------------------------------------------------
# BENCHMARK SUITE
# -------------------------------------------------------------------------
def load_agents():
    """Load trained AI agents."""
    mec_agent = MECAgentPPO()
    ho_agent = HOAgentPPO()
    
    try:
        mec_agent.load(MEC_CHECKPOINT)
        ho_agent.load(HO_CHECKPOINT)
        print("Agents loaded successfully.")
    except Exception as e:
        print(f"Error loading agents: {e}")
        return None, None
    
    # Set to eval mode
    if hasattr(mec_agent.network, 'eval'): mec_agent.network.eval()
    if hasattr(ho_agent.network, 'eval'): ho_agent.network.eval()
        
    return ho_agent, mec_agent

def run_episode(sim, strategy_name, ho_agent, mec_agent, steps=STEPS_PER_EPISODE):
    """Run a single episode and collect metrics."""
    # Reset context done outside
    
    metrics = {
        'tasks_total': 0,
        'tasks_success': 0,
        'total_energy': 0,
        'total_latency': 0,
        'handovers': 0,
        'local_count': 0,
        'edge_count': 0,
        'rsrp_history': [],
        'last_ho_time': -999,  # For ping-pong detection
        'pingpong_count': 0
    }
    
    context = sim.get_context()
    prev_cell = context['serving_cell_id']
    
    # MEC Callback Wrapper
    def mec_callback(task, ctx):
        if strategy_name == "Trained AI":
            obs = mec_agent.get_observation(ctx)
            action = mec_agent.select_action(obs, context=ctx, training=False)
            return mec_agent.action_map.get(action, "local")
        elif strategy_name == "Stay":
            return "local"
        else: # Greedy
            # Replicate greedy logic: Edge if throughput good, else local
            if ctx['serving_throughput_bps'] > 50e6:
                return "edge"
            return "local"

    for step_i in range(steps):
        # HO Decision
        ho_decision = None
        
        if strategy_name == "Trained AI":
            obs_ho = ho_agent.get_observation(context)
            ho_action, _, _ = ho_agent.select_action_with_info(obs_ho, context=context, training=False)
            ho_decision = int(ho_action)
        elif strategy_name == "Stay":
            # Stay on current cell (unless RLF forces it, but we force decision=current)
            ho_decision = context['serving_cell_id']
        else:
            # Greedy: Simulation handles None as Greedy RSRP
            ho_decision = None 
        
        # Step
        next_context, info = sim.step(ho_decision, mec_callback)
        
        # Track metrics
        if info.get('task_info'):
            task_data = info['task_info']
            metrics['tasks_total'] += 1
            metrics['total_energy'] += task_data.get('energy_j', 0)
            metrics['total_latency'] += task_data.get('latency_s', 0)
            if task_data.get('deadline_met'):
                metrics['tasks_success'] += 1
            
            # Track offloading decisions
            if task_data.get('offload_target') == 'local':
                metrics['local_count'] += 1
            else:
                metrics['edge_count'] += 1
        
        # Track handovers and ping-pong
        current_cell = next_context['serving_cell_id']
        if current_cell != prev_cell:
            metrics['handovers'] += 1
            # Ping-pong: HO within 1 second of last HO
            if (sim.current_time_s - metrics['last_ho_time']) < 1.0:
                metrics['pingpong_count'] += 1
            metrics['last_ho_time'] = sim.current_time_s
        
        metrics['rsrp_history'].append(next_context['serving_rsrp_dbm'])
        
        context = next_context
        prev_cell = current_cell
    
    return metrics

def test_congestion():
    """Test 1: Network Congestion Stress Test."""
    print("\n" + "="*80)
    print("TEST 1: CONGESTION STRESS TEST (Network Context)")
    print("="*80)
    
    ho_agent, mec_agent = load_agents()
    sim = NetworkSimulation()
    
    strategies = ["Greedy", "Stay", "Trained AI"]
    
    results = []
    
    for congestion in CONGESTION_LEVELS:
        print(f"\n[Congestion={congestion:.2f}]")
        
        for strat_name in strategies:
            deadline_rates = []
            energies = []
            
            for ep in range(NUM_EPISODES_PER_POINT):
                sim.reset(seed=random.randint(0, 100000))
                for bs in sim.base_stations:
                    sim.set_load_factor(bs.id, congestion)
                
                metrics = run_episode(sim, strat_name, ho_agent, mec_agent)
                
                success_rate = (metrics['tasks_success'] / max(metrics['tasks_total'], 1)) * 100
                avg_energy = metrics['total_energy'] / max(metrics['tasks_total'], 1)
                
                deadline_rates.append(success_rate)
                energies.append(avg_energy)
            
            results.append({
                'Test': 'Congestion',
                'Strategy': strat_name,
                'Parameter': congestion,
                'Deadline_Rate': np.mean(deadline_rates),
                'Deadline_Std': np.std(deadline_rates),
                'Energy': np.mean(energies),
                'Energy_Std': np.std(energies)
            })
            
            print(f"  {strat_name:12s}: Success={np.mean(deadline_rates):.1f}% Energy={np.mean(energies):.2f}J")
    
    return pd.DataFrame(results)

def test_mobility():
    """Test 2: User Mobility Stress Test."""
    print("\n" + "="*80)
    print("TEST 2: MOBILITY STRESS TEST (User Context)")
    print("="*80)
    
    ho_agent, mec_agent = load_agents()
    
    strategies = ["Greedy", "Stay", "Trained AI"]
    
    results = []
    
    for speed in MOBILITY_SPEEDS:
        print(f"\n[Speed={speed} m/s]")
        
        for strat_name in strategies:
            handover_counts = []
            pingpong_rates = []
            
            for ep in range(NUM_EPISODES_PER_POINT):
                # Create sim with modified UE speed
                sim = NetworkSimulation()
                sim.reset(seed=random.randint(0, 100000))
                # Override UE speed - CRITICAL for mobility test
                sim.ue.speed_mps = speed
                if sim.mobility: 
                     sim.mobility.min_speed = speed # Ensure random waypoint respects it
                     sim.mobility.current_speed = speed

                metrics = run_episode(sim, strat_name, ho_agent, mec_agent)
                
                handover_counts.append(metrics['handovers'])
                # Ping-pong rate = pingpongs / total handovers (%)
                pp_rate = (metrics['pingpong_count'] / max(metrics['handovers'], 1)) * 100
                pingpong_rates.append(pp_rate)
            
            results.append({
                'Test': 'Mobility',
                'Strategy': strat_name,
                'Parameter': speed,
                'Handovers': np.mean(handover_counts),
                'Handovers_Std': np.std(handover_counts),
                'PingPong_Rate': np.mean(pingpong_rates),
                'PingPong_Std': np.std(pingpong_rates)
            })
            
            print(f"  {strat_name:12s}: HO={np.mean(handover_counts):.1f} PingPong={np.mean(pingpong_rates):.1f}%")
    
    return pd.DataFrame(results)

def test_application():
    """Test 3: Application Load Stress Test."""
    print("\n" + "="*80)
    print("TEST 3: APPLICATION STRESS TEST (QoS Context)")
    print("="*80)
    
    ho_agent, mec_agent = load_agents()
    sim = NetworkSimulation()
    
    strategies = ["Greedy", "Stay", "Trained AI"]
    
    results = []
    
    # We'll override task size by modifying SERVICE_PROFILES temporarily
    from src.simulation import SERVICE_PROFILES
    
    for task_size in APPLICATION_SIZES:
        print(f"\n[Task Size={task_size/1e6:.1f} Mbits]")
        
        # Temporarily modify VR profile task size
        original_size = SERVICE_PROFILES['VR'].task_data_bits_mean
        SERVICE_PROFILES['VR'].task_data_bits_mean = task_size
        
        for strat_name in strategies:
            latencies = []
            success_rates = []
            
            for ep in range(NUM_EPISODES_PER_POINT):
                sim.reset(seed=random.randint(0, 100000))
                
                metrics = run_episode(sim, strat_name, ho_agent, mec_agent)
                
                success_rate = (metrics['tasks_success'] / max(metrics['tasks_total'], 1)) * 100
                avg_latency_ms = (metrics['total_latency'] / max(metrics['tasks_total'], 1)) * 1000
                
                latencies.append(avg_latency_ms)
                success_rates.append(success_rate)
            
            results.append({
                'Test': 'Application',
                'Strategy': strat_name,
                'Parameter': task_size / 1e6,  # Store as Mbits
                'Latency_ms': np.mean(latencies),
                'Latency_Std': np.std(latencies),
                'Success_Rate': np.mean(success_rates),
                'Success_Std': np.std(success_rates)
            })
            
            print(f"  {strat_name:12s}: Latency={np.mean(latencies):.1f}ms Success={np.mean(success_rates):.1f}%")
        
        # Restore original size
        SERVICE_PROFILES['VR'].task_data_bits_mean = original_size
    
    return pd.DataFrame(results)

def test_celledge():
    """Test 4: Cell-Edge Distance Stress Test."""
    print("\n" + "="*80)
    print("TEST 4: CELL-EDGE STRESS TEST (Physics Context)")
    print("="*80)
    
    ho_agent, mec_agent = load_agents()
    
    strategies = ["Greedy", "Stay", "Trained AI"]
    
    results = []
    
    for distance in CELLEDGE_DISTANCES:
        print(f"\n[Distance={distance}m from Cell 0]")
        
        for strat_name in strategies:
            energies = []
            offload_ratios = []
            
            for ep in range(NUM_EPISODES_PER_POINT):
                sim = NetworkSimulation()
                sim.reset(seed=random.randint(0, 100000))
                
                # Override UE position to be at specific distance from Cell 0
                cell0_x = sim.base_stations[0].x
                cell0_y = sim.base_stations[0].y
                
                sim.ue.x = cell0_x + distance
                sim.ue.y = cell0_y
                sim.ue.speed_mps = 0.0 # Static for distance test
                
                metrics = run_episode(sim, strat_name, ho_agent, mec_agent)
                
                avg_energy = metrics['total_energy'] / max(metrics['tasks_total'], 1)
                # Offload ratio = edge / (local + edge)
                total_decisions = metrics['local_count'] + metrics['edge_count']
                offload_pct = (metrics['edge_count'] / max(total_decisions, 1)) * 100
                
                energies.append(avg_energy)
                offload_ratios.append(offload_pct)
            
            results.append({
                'Test': 'CellEdge',
                'Strategy': strat_name,
                'Parameter': distance,
                'Energy': np.mean(energies),
                'Energy_Std': np.std(energies),
                'Offload_Ratio': np.mean(offload_ratios),
                'Offload_Std': np.std(offload_ratios)
            })
            
            print(f"  {strat_name:12s}: Energy={np.mean(energies):.2f}J Offload={np.mean(offload_ratios):.1f}%")
    
    return pd.DataFrame(results)

def plot_results(df_all):
    """Generate plots for all benchmark tests."""
    print("\nGenerating Plots...")
    
    # Set style if available, else default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('5G Network Benchmark Suite Results', fontsize=16)
    
    # 1. Congestion (Top Left)
    df_cong = df_all[df_all['Test'] == 'Congestion']
    ax = axes[0, 0]
    for strat in df_cong['Strategy'].unique():
        data = df_cong[df_cong['Strategy'] == strat]
        # Sort by parameter to ensure line plot connects correctly
        data = data.sort_values('Parameter')
        ax.plot(data['Parameter'], data['Deadline_Rate'], marker='o', label=strat)
        ax.fill_between(data['Parameter'], 
                       data['Deadline_Rate'] - data['Deadline_Std'],
                       data['Deadline_Rate'] + data['Deadline_Std'], alpha=0.2)
    ax.set_title('Network Congestion Impact')
    ax.set_xlabel('Cell Load Factor (0-1)')
    ax.set_ylabel('Task Success Rate (%)')
    ax.legend()
    ax.grid(True)
    
    # 2. Mobility (Top Right)
    df_mob = df_all[df_all['Test'] == 'Mobility']
    ax = axes[0, 1]
    for strat in df_mob['Strategy'].unique():
        data = df_mob[df_mob['Strategy'] == strat]
        data = data.sort_values('Parameter')
        ax.plot(data['Parameter'], data['Handovers'], marker='o', label=strat)
    ax.set_title('Mobility Impact on Handovers')
    ax.set_xlabel('UE Speed (m/s)')
    ax.set_ylabel('Average Handovers')
    ax.legend()
    ax.grid(True)

    # 3. Application (Bottom Left)
    df_app = df_all[df_all['Test'] == 'Application']
    ax = axes[1, 0]
    for strat in df_app['Strategy'].unique():
        data = df_app[df_app['Strategy'] == strat]
        data = data.sort_values('Parameter')
        ax.plot(data['Parameter'], data['Latency_ms'], marker='o', label=strat)
    ax.set_title('Application Load Impact')
    ax.set_xlabel('Task Size (Mbits)')
    ax.set_ylabel('Average Latency (ms)')
    ax.legend()
    ax.grid(True)

    # 4. Cell-Edge (Bottom Right)
    df_edge = df_all[df_all['Test'] == 'CellEdge']
    ax = axes[1, 1]
    for strat in df_edge['Strategy'].unique():
        data = df_edge[df_edge['Strategy'] == strat]
        data = data.sort_values('Parameter')
        ax.plot(data['Parameter'], data['Energy'], marker='o', label=strat)
    ax.set_title('Cell-Edge Energy Consumption')
    ax.set_xlabel('Distance from Tower (m)')
    ax.set_ylabel('Avg Energy per Task (J)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, "benchmark_suite_plots_new.png")
    plt.savefig(plot_path)
    print(f">> Plots saved to {plot_path}")

def main():
    """Run complete benchmark suite."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUITE - NEW ARCHITECTURE")
    print("="*80)
    print(f"Episodes per data point: {NUM_EPISODES_PER_POINT}")
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    
    # Run all tests
    df_congestion = test_congestion()
    df_mobility = test_mobility()
    df_application = test_application()
    df_celledge = test_celledge()
    
    # Combine all results
    df_all = pd.concat([df_congestion, df_mobility, df_application, df_celledge], ignore_index=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, "benchmark_suite_results.csv"), index=False)
    print("\n>> Raw data saved to benchmark_suite_results.csv")
    
    # Generate Plots
    #df_all = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_suite_results_new.csv"))
    plot_results(df_all)
    
    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
