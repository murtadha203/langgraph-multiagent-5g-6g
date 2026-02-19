from .ppo_core import BasePPOAgent

import numpy as np
import math
from collections import deque
from typing import Dict, Any

class HOAgentPPO(BasePPOAgent):
    """
    PPO implementation for the Handover (HO) Agent.
    Combines mobility-aware observation logic with a Proximal Policy Optimization core.
    Supports frame stacking for temporal feature perception (e.g., velocity).
    """
    def __init__(self, agent_id="ho_agent_ppo", config=None, num_cells=7, lr=3e-4, frame_stack=1, **kwargs):
        
        # Initialize Base params
        self.num_cells = num_cells
        # Observation dimensions (includes throughput, RSRP, and intent weights)
        self.raw_obs_dim = 5 * num_cells + 7 
        self.frame_stack = frame_stack
        self.obs_dim = self.raw_obs_dim * frame_stack
        
        # Stacking Buffer
        self.obs_queue = deque(maxlen=frame_stack)
        
        # Call BasePPOAgent init
        BasePPOAgent.__init__(self, obs_dim=self.obs_dim, action_dim=num_cells,
                              lr=lr, gamma=0.99, batch_size=64, **kwargs)
        
        # Call HOAgent init (mostly to set config and other state)
        # Note: We overwrite select_action and update, so DQN parts are ignored
        self.agent_id = agent_id
        self.config = config or {}
        
        # Initialize state tracking from ho_agent
        self.last_handover_time = 0.0
        self.handover_margin_db = 3.0
        self.time_to_trigger_s = 0.16
        
        # Metrics to mimic HOAgent interface for logging
        self.loss_history = []
        self.update_counter = 0
        self.episode_counter = 0

    def reset_stack(self):
        """Clear observation history (Call at start of episode)."""
        self.obs_queue.clear()
        
    def get_observation(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract scale-invariant features from simulation context.
        
        Feature Vector (Size = 5 * num_cells + 6):
        
        Per-Cell Features (for each cell i):
        1. RSRP (norm): (dbm + 100) / 50
        2. SINR (norm): (db + 10) / 30
        3. Rel Dist: distance_to_ue / current_isd (clipped to 3.0)
        4. Rel Bearing: angle_to_ue / pi [-1, 1]
        5. Is Serving: 1.0 if serving, else 0.0
        
        Global Features:
        1. UE Speed (norm): mps / 30
        2. UE Battery (norm): joules / 1000
        3. Time since HO (norm): sec / 5
        4. Recent HO Count (norm): count / 5
        5. Alpha (Latency weight)
        6. Beta (Energy weight)
        """
        # 1. Context Extraction
        rsrp_dbm = context["rsrp_dbm"]
        sinr_db = context.get("sinr_db")
        if sinr_db is None:
             # Backward compat if sinr not in context
             sinr_db = [-20.0] * self.num_cells

        ue_pos = context.get("ue_position", (0, 0))
        bs_positions = context.get("bs_positions", [])
        current_isd = context.get("current_isd", 500.0)  # Default if missing
        serving_id = context["serving_cell_id"]
        
        
        # For robustness, if missing, assume 3-cell line or generic
        if not bs_positions and self.num_cells == 3:
            # Legacy fallback
            bs_positions = [(0, 0), (500, 0), (1000, 0)] 
            
        features = []
        
        # 2. Per-Cell Features
        for i in range(self.num_cells):
            # Safe access with padding if mismatch
            r_val = rsrp_dbm[i] if i < len(rsrp_dbm) else -140.0
            s_val = sinr_db[i] if i < len(sinr_db) else -20.0
            
            # RSRP Norm
            rsrp_norm = (r_val + 100.0) / 50.0
            
            # SINR Norm
            sinr_norm = (s_val + 10.0) / 30.0
            
            
            if i < len(bs_positions):
                bx, by = bs_positions[i]
                dx = bx - ue_pos[0]
                dy = by - ue_pos[1]
                dist = math.hypot(dx, dy)
                angle = math.atan2(dy, dx)
                
                # Normalize distance by ISD (Scale Invariance!)
                # Clip to 3.0 (e.g. 3 ISDs away is "far")
                dist_norm = min(dist / max(current_isd, 1.0), 3.0) 
                
                # Normalize bearing to [-1, 1]
                bearing_norm = angle / math.pi
            else:
                dist_norm = 3.0
                bearing_norm = 0.0
            
            # Is Serving
            is_serving = 1.0 if i == serving_id else 0.0
            
            features.extend([rsrp_norm, sinr_norm, dist_norm, bearing_norm, is_serving])
            
        # 3. Global Features
        ue_speed = context.get("ue_speed_mps", 0.0)
        speed_norm = min(ue_speed / 30.0, 1.0)
        
        battery = context.get("ue_battery_joules", 1000.0)
        batt_norm = battery / 1000.0
        
        # HO History
        current_time = context.get("time_s", 0.0)
        ho_history = context.get("handover_history", [])
        
        if ho_history:
            t_since = min(current_time - ho_history[-1], 5.0)
        else:
            t_since = 5.0
        time_ho_norm = t_since / 5.0
        
        recent_count = sum(1 for t in ho_history if current_time - t <= 2.0)
        count_ho_norm = min(recent_count / 5.0, 1.0)
        
        # Context Weights (Intent Weights)
        intent = context.get("intent_weights")
        if intent:
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
            
        features.extend([speed_norm, batt_norm, time_ho_norm, count_ho_norm, alpha, beta, gamma])
        
        single_obs = np.array(features, dtype=np.float32)
        
        # 2. Stack Handling
        if self.frame_stack > 1:
            if len(self.obs_queue) == 0:
                # First step: Repeat frame
                for _ in range(self.frame_stack):
                    self.obs_queue.append(single_obs)
            else:
                self.obs_queue.append(single_obs)
                
            # Flatten to [18*4]
            return np.concatenate(self.obs_queue)
        else:
            return single_obs
       
    def select_action(self, observation: np.ndarray, context: dict = None, training=True):
        """
        PPO Action Selection: Sample from policy.
        Returns: action (int)
        
        Note: PPO needs log_prob and value for update.
        In training loop, we must call select_action_with_info() instead.
        This method is kept for compatibility with evaluating code.
        """
        action, _, _ = self.select_action_with_info(observation, context, training=training)
        return action

    def select_action_with_info(self, observation: np.ndarray, context: dict = None, training=True):
        """
        Returns action, log_prob, value for training buffer.
        """
        # Always run network first
        action, log_prob, value = BasePPOAgent.select_action(self, observation, training=training)
        
        # --- CONTROL OVERRIDE ---
        # If the agent is in a critical connection state, force a handover to the best cell.
        if context:
            # Use SINR for decision (Connection Quality) instead of just RSRP (Signal Strength)
            sinr_list = context.get('sinr_db', [])
            serving_id = context.get('serving_cell_id', 0)
            
            if sinr_list and 0 <= serving_id < len(sinr_list):
                current_sinr = sinr_list[serving_id]
                
                # Threshold: -5.0 dB SINR (Poor Quality/Outage Risk)
                if current_sinr < -5.0:
                    best_cell = int(np.argmax(sinr_list))
                    best_sinr = sinr_list[best_cell]
                    
                    # Switch if there is a significantly better option (> 0 dB or +5 dB better)
                    if best_sinr > current_sinr + 5.0:
                        action = best_cell
                        # We keep log_prob/value as is, treating this as environment dynamics
        # -----------------------
        
        return action, log_prob, value
        
    def get_metrics(self):
        return {
            "episode": self.episode_counter,
            "updates": self.update_counter,
            "buffer_size": len(self.obs_buffer)
        }

    def update(self, rollout_data, last_val=0.0):
        """
        PPO Update Step.
        Accepts aggregated rollout dictionary and runs optimization.
        Overwrites HOAgent.update (DQN).
        """
        self.obs_buffer = rollout_data['obs']
        self.action_buffer = rollout_data['act']
        self.logprob_buffer = rollout_data['logprob']
        self.reward_buffer = rollout_data['rew']
        self.value_buffer = rollout_data['val']
        self.done_buffer = rollout_data['done']
        
        metrics = self.finish_episode(last_val=last_val)
        self.update_counter += 1
        return metrics

    # Re-use save/load from BasePPOAgent (overwrites HOAgent's save/load)
    
    def load(self, path):
         BasePPOAgent.load(self, path)
         print(f"HOAgentPPO loaded from {path}")
         
    def save(self, path):
         BasePPOAgent.save(self, path)
         print(f"HOAgentPPO saved to {path}")
