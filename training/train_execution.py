"""
FORTRESS v5 - train_execution.py
Path: training/train_execution.py

MARL Execution Training Loop.
Trains the Stealth PPO agent using Reinforcement Learning against a simulated Limit Order Book.
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.optim import AdamW

from models.execution.stealth_ppo import StealthPPO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionTrainer")

class ExecutionTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('stealth_ppo', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StealthPPO(self.config).to(self.device)
        
        # PPO requires optimizing both Actor and Critic
        self.optimizer = AdamW([
            {'params': self.model.actor.parameters(), 'lr': 3e-4},
            {'params': self.model.critic.parameters(), 'lr': 1e-3}
        ])
        
        self.gamma = 0.99       # Discount factor
        self.clip_ratio = 0.2   # PPO clipping parameter
        self.ppo_epochs = 10
        self.episodes = 5000

    def _simulate_order_book_step(self, action: np.ndarray, remaining_inventory: float) -> tuple:
        """
        Scaffold for the LOB Environment. 
        In production, this replays Level-2 tick data to calculate exact slippage.
        """
        # Action: [Price Offset, Size Fraction]
        size_fraction = np.clip(action[1], 0.0, 1.0)
        executed_qty = remaining_inventory * size_fraction
        
        # Calculate reward: Negative implementation shortfall
        # If they place a passive limit (negative offset), they capture spread but risk not filling
        price_offset = action[0] 
        base_slippage = 0.0005 # 5 bps
        shortfall = base_slippage + (price_offset * 0.001)
        
        reward = -shortfall 
        
        # Next state representation
        next_state = np.random.randn(self.config.get('stealth_state_dim', 12)).astype(np.float32)
        
        return next_state, reward, executed_qty

    def train(self):
        logger.info(f"Initiating PPO Training for Stealth Agent over {self.episodes} episodes...")
        
        for episode in range(1, self.episodes + 1):
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            # Reset Environment for a new parent order
            state = np.random.randn(self.config.get('stealth_state_dim', 12)).astype(np.float32)
            remaining_inventory = 1.0 # 100% of parent order
            
            # Step through the execution window (e.g., 20 time slices)
            for t in range(20):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get action and value from the networks
                with torch.no_grad():
                    action_mean, action_std = self.model.actor(state_tensor)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = self.model.critic(state_tensor).squeeze(-1)
                
                act_np = action.squeeze(0).cpu().numpy()
                next_state, reward, executed = self._simulate_order_book_step(act_np, remaining_inventory)
                remaining_inventory -= executed
                
                states.append(state)
                actions.append(act_np)
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                state = next_state
                if remaining_inventory <= 0.01:
                    break # Order complete
            
            # Penalize agent if the parent order wasn't fully executed
            if remaining_inventory > 0.01:
                rewards[-1] -= (remaining_inventory * 0.05) # Severe penalty for missing the fill
                
            self._update_ppo(states, actions, rewards, log_probs, values)
            
            if episode % 100 == 0:
                avg_reward = np.mean(rewards)
                logger.info(f"Episode [{episode}/{self.episodes}] | Avg Slice Reward: {avg_reward:.5f} | Remaining Inv: {remaining_inventory:.2%}")

        self._save_weights()

    def _update_ppo(self, states, actions, rewards, old_log_probs, values):
        """PPO Policy Update Phase using Clipped Surrogate Objective."""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Calculate Returns and Advantages (Simplified Monte Carlo)
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            
            # Recalculate probabilities and values with updated weights
            curr_log_probs, state_values, entropy = self.model.evaluate_actions(states, actions)
            
            # Ratio of new probabilities vs old probabilities
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic MSE Loss
            critic_loss = torch.nn.functional.mse_loss(state_values, returns)
            
            # Total Loss (Subtract entropy to encourage exploration)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/stealth_ppo_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Stealth MARL weights saved to {save_path}")

if __name__ == "__main__":
    trainer = ExecutionTrainer()
    trainer.train()