import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class BasketEnv(ParallelEnv):
    """MAS для корзины: 3 агента оптимизируют по запросу."""
    metadata = {"render_modes": ["human"], "name": "basket_mas_v0"}
    
    def __init__(self, budget=1500.0, max_steps=10):
        super().__init__()
        self._budget = float(budget)
        self._max_steps = int(max_steps)
        
        self.possible_agents = ["budget_agent", "compat_agent", "profile_agent"]
        self.agents = self.possible_agents.copy()
        
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=2000, shape=(12,), dtype=np.float32) 
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(11)
            for agent in self.possible_agents
        }
        
        self.current_sum = 0.0
        self.cart = []
        self.steps = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.current_sum = 0.0
        self.cart = []
        self.steps = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        obs_dict = {}
        for agent in self.agents:
            # ✅ Используем self._budget
            obs_array = np.array(
                [self._budget, 0.0] + [0.0] * 10,
                dtype=np.float32
            )
            obs_dict[agent] = obs_array
        
        infos = {agent: {"budget": self._budget} for agent in self.agents}
        return obs_dict, infos
    
    def step(self, actions):
        for agent, action in actions.items():
            if action > 0:
                price = 100 + action * 50
                self.cart.append(price)
                self.current_sum += price
        
        self.steps += 1
        
        budget_diff = abs(self.current_sum - self._budget)
        rewards = {
            "budget_agent": -budget_diff / 100,
            "compat_agent": len(self.cart) * 0.2 if len(self.cart) > 1 else 0,
            "profile_agent": 0.5 if self.cart else 0
        }
        
        for agent, r in rewards.items():
            self._cumulative_rewards[agent] += r
        
        obs = {}
        for agent in self.agents:
            obs_array = np.array(
                [self._budget - self.current_sum, self.current_sum] 
                + (np.random.rand(10) * 100).tolist(),
                dtype=np.float32
            )
            obs[agent] = obs_array
        
        done = self.steps >= self._max_steps
        terms = {agent: done for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        infos = {
            agent: {"cart_sum": self.current_sum, "cart_size": len(self.cart)} 
            for agent in self.agents
        }
        
        if done:
            self.agents = []
        
        return obs, rewards, terms, truncs, infos
    
    def render(self):
        print(f"Step {self.steps}: Cart={self.cart}, Sum={self.current_sum:.2f}")
    
    def close(self):
        pass


def create_basket_env(budget=1500.0, max_steps=10):
    """Factory для создания среды."""
    return BasketEnv(budget=float(budget), max_steps=int(max_steps))
