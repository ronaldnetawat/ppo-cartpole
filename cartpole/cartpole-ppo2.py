import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from tqdm import tqdm


# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, batch_size=64):
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # clipping param
        self.epochs = epochs # number of ppo epochs per update
        self.batch_size = batch_size
        
        # Initialize actor-critic network
        self.policy = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """Select action using the current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, next_value):
        """Compute discounted returns (rewards-to-go)"""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update(self):
        """Update policy using PPO"""
        # Compute last state value for bootstrapping
        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
        with torch.no_grad():
            _, last_value = self.policy(last_state)
            last_value = last_value.item() if not self.dones[-1] else 0.0
        
        # Compute returns and advantages
        returns = self.compute_returns(last_value)
        returns = torch.FloatTensor(returns)
        
        # Convert buffers to tensors
        old_states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        
        # Normalize advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(self.epochs):
            # Generate random indices for mini-batch training
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy predictions
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                
                # Policy loss (negative because we want to maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Total loss (policy loss + value loss - entropy bonus)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        # Clear memory
        self.clear_memory()
        
        return total_policy_loss / self.epochs, total_value_loss / self.epochs
    
    def clear_memory(self):
        """Clear memory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []


# Training Loop
def train_ppo(episodes=1000, max_steps=500, update_frequency=2048):
    """Train PPO agent on CartPole environment"""
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = PPOAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    running_reward = 0
    
    # Progress bar
    pbar = tqdm(range(episodes), desc="Training PPO")
    
    step_count = 0
    
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Update policy every update_frequency steps
            if step_count % update_frequency == 0:
                policy_loss, value_loss = agent.update()
        
        # Update metrics
        episode_rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        # Update progress bar
        if episode % 10 == 0:
            pbar.set_postfix({
                'Episode': episode,
                'Reward': f'{episode_reward:.2f}',
                'Avg Reward': f'{running_reward:.2f}'
            })
        
        # Check if solved (average reward > 475 over last 100 episodes)
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"\nSolved at episode {episode}! Average reward: {np.mean(episode_rewards[-100:]):.2f}")
            break
    
    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    print("Training CartPole with PPO...")
    print("=" * 50)
    
    # Train the agent
    agent, rewards = train_ppo(episodes=1000, max_steps=500, update_frequency=2048)
    
    print("\nTraining complete!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    for test_episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Test Episode {test_episode + 1}: Reward = {total_reward}")
    
    env.close()

