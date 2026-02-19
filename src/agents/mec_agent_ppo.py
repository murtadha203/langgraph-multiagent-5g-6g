from .ppo_core import BasePPOAgent

from typing import Dict, Any, Optional
import numpy as np

class MECAgentPPO(BasePPOAgent):
    """
    PPO Version of MEC Agent.
    """
    def __init__(self, agent_id="mec_agent_ppo", config=None, num_actions=3, lr=3e-4, **kwargs):
        self.num_actions = num_actions
        self.obs_dim = 19 # Scale-invariant observation dimension
        
        BasePPOAgent.__init__(self, obs_dim=self.obs_dim, action_dim=num_actions,
                              lr=lr, gamma=0.99, batch_size=64)
        
        self.agent_id = agent_id
        self.config = config or {}
        
        # State tracking
        self.last_offload_target = "local"
        self.last_task_success = True
        self.last_task_time = 0.0
        self.task_counter = 0
        
        self.action_map = {0: "local", 1: "edge", 2: "cloud"}

    # Observation Processing
    
    def get_observation(self, context: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract and normalize features from simulation context and task.
        
        Observation Vector (19 features):
        - Task data size (1): Normalized by 100 Mbits
        - Task CPU cycles (1): Normalized by 5 Gigacycles
        - Task deadline (1): Normalized by 1 second
        - Service type (3): One-hot [VR, EV, IoT]
        - Serving throughput (1): Normalized by 1 Gbps
        - Serving RSRP (1): Normalized to [-1, 1]
        - Serving SINR (1): Normalized to [-1, 1]
        - Battery level (1): Normalized by 1000 J
        - UE speed (1): Normalized by 30 m/s
        - Time since last task (1): Normalized by 10 seconds
        - Latency weight (1): Alpha from ContextAgent
        - Energy weight (1): Beta from ContextAgent
        - Reliability weight (1): Gamma from ContextAgent
        # - Serving cell (removed for scale invariance)
        - Last offload target (3): One-hot [local, edge, cloud]
        - Last task success (1): Binary {0, 1}
        """
        features = []
        
        # 1. Task characteristics (6 features)
        if task is not None:
            # Actual task data
            task_data_bits = task.get("data_size_bits", 0.0)
            task_cpu_cycles = task.get("cpu_cycles", 0.0)
            task_deadline_s = task.get("deadline_s", 0.0) - context.get("time_s", 0.0)
            service_type = task.get("service_type", "VR")
            
            # Normalize
            task_data_norm = task_data_bits / 100e6  # / 100 Mbits
            task_cpu_norm = task_cpu_cycles / 5e9    # / 5 Gigacycles
            task_deadline_norm = min(task_deadline_s / 1.0, 2.0)  # / 1 second, clip at 2
            
            # Service type one-hot
            service_onehot = [
                1.0 if service_type == "VR" else 0.0,
                1.0 if service_type == "EV" else 0.0,
                1.0 if service_type == "IoT" else 0.0
            ]
        else:
            # No task: use default/zero values
            task_data_norm = 0.0
            task_cpu_norm = 0.0
            task_deadline_norm = 1.0
            service_onehot = [0.0, 0.0, 0.0]
        
        features.extend([task_data_norm, task_cpu_norm, task_deadline_norm])
        features.extend(service_onehot)
        
        # 2. Radio conditions (3 features)
        serving_throughput_bps = context.get("serving_throughput_bps", 0.0)
        serving_rsrp_dbm = context.get("serving_rsrp_dbm", -100.0)
        serving_sinr_db = context.get("serving_sinr_db", 0.0)
        
        thr_norm = serving_throughput_bps / 1e9  # / 1 Gbps
        rsrp_norm = (serving_rsrp_dbm + 100.0) / 50.0  # [-150, -50] -> [-1, 1]
        sinr_norm = (serving_sinr_db + 10.0) / 30.0    # [-10, 20] -> [-0.33, 1]
        
        features.extend([thr_norm, rsrp_norm, sinr_norm])
        
        # 3. UE context (3 features)
        battery = context.get("ue_battery_joules", 1000.0)
        ue_speed = context.get("ue_speed_mps", 0.0)
        current_time = context.get("time_s", 0.0)
        
        battery_norm = battery / 1000.0
        speed_norm = min(ue_speed / 30.0, 1.0)
        time_since_last_task = min(current_time - self.last_task_time, 10.0)
        time_since_task_norm = time_since_last_task / 10.0
        
        features.extend([battery_norm, speed_norm, time_since_task_norm])
        
        # 4. Intent Weights (3 features)
        # [PHASE 4] Explicit IBN Support: w_lat, w_eng, w_thr
        intent = context.get("intent_weights")
        if intent:
            # Check keys
            alpha = intent.get('latency', 0.33)
            beta = intent.get('energy', 0.33)
            gamma = intent.get('throughput', 0.34)
        elif getattr(self, "context_weights", None):
            alpha = self.context_weights.get("alpha", 0.5)
            beta = self.context_weights.get("beta", 0.5)
            gamma = self.context_weights.get("gamma", 0.0)
        else:
            user_pref = context.get("user_pref", {})
            alpha = user_pref.get("latency_weight", 0.5)
            beta = user_pref.get("energy_weight", 0.5)
            gamma = 1.0 - alpha - beta if (alpha + beta) < 1.0 else 0.0
        
        features.extend([alpha, beta, gamma])
        
        # 5. Serving cell one-hot (Removed for Renaissance Scale Invariance)
        # serving_cell_id = context.get("serving_cell_id", 0)
        # features.extend(...) 
        # We rely on radio metrics (RSRP/Throughput) rather than Cell ID.
        
        # 6. Last decision context (4 features)
        last_target_onehot = [
            1.0 if self.last_offload_target == "local" else 0.0,
            1.0 if self.last_offload_target == "edge" else 0.0,
            1.0 if self.last_offload_target == "cloud" else 0.0
        ]
        last_success = 1.0 if self.last_task_success else 0.0
        
        features.extend(last_target_onehot)
        features.append(last_success)
        
        # Convert to numpy array
        observation = np.array(features, dtype=np.float32)
        
        # Sanity check
        assert len(observation) == self.obs_dim, f"Observation size mismatch: {len(observation)} != {self.obs_dim}"
        
        return observation

    def select_action(self, observation: np.ndarray, context=None, training=True):
        # Physics Guardrail (Safety Override)
        if context is not None:
             # Check Throughput directly (captures both RSRP and SINR/Interference issues)
             throughput = context.get("serving_throughput_bps", 0.0)
             
             # If throughput is trash (< 1 Mbps), force LOCAL.
             # Saving 15 Mbits at 0.1 Mbps takes 150 seconds = HUGE energy.
             if throughput < 1e6: 
                 return 0 # Local
        
        action, _, _ = BasePPOAgent.select_action(self, observation, training=training)
        return action

    def select_action_with_info(self, observation: np.ndarray, context=None, training=True):
        """
        Select action and return policy information.
        Includes a safety guardrail for low Throughput conditions.
        """
        # 1. Always get the "true" network output first for valid log_prob/value
        action, log_prob, value = BasePPOAgent.select_action(self, observation, training=training)

        # 2. Apply Guardrail Override
        if context is not None:
             throughput = context.get("serving_throughput_bps", 0.0)
             # Safety Override: Force Local (0) if Throughput < 1 Mbps
             if throughput < 1e6:
                 # Override ACTION, but keep LOG_PROB and VALUE from the network.
                 # This ensures PPO math remains valid (on-policy assumption mostly holds for value/prob)
                 # even if the executed action was forced.
                 return 0, log_prob, value
        
        return action, log_prob, value

    def get_metrics(self):
        return {
            "tasks_processed": self.task_counter,
            "buffer_size": len(self.obs_buffer)
        }

    def update(self, rollout_data, last_val=0.0):
        self.obs_buffer = rollout_data['obs']
        self.action_buffer = rollout_data['act']
        self.logprob_buffer = rollout_data['logprob']
        self.reward_buffer = rollout_data['rew']
        self.value_buffer = rollout_data['val']
        self.done_buffer = rollout_data['done']
        return self.finish_episode(last_val=last_val)
    
    def load(self, path):
         BasePPOAgent.load(self, path)
         print(f"MECAgentPPO loaded from {path}")
         
    def save(self, path):
         BasePPOAgent.save(self, path)
         print(f"MECAgentPPO saved to {path}")
