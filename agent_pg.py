import torch
import copy
from torch import nn
import random, numpy as np
from pathlib import Path
from torch.distributions import Categorical
from collections import deque


class MarioPG:
    """
    Policy Gradient Agent class
    """
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 10
        self.gamma = 0.95

        #Useless stuffs for metrics consistency
        self.epsilon = -99
        self.exploration_rate = -99
        self.curr_step = -99

        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.reset()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action (int): An integer representing which action Mario will perform
        """
        state = torch.FloatTensor(state.__array__()).cuda() if self.use_cuda else torch.FloatTensor(state.__array__())
        state = state.unsqueeze(0)
        distribution = Categorical(self.net(state))
        action = distribution.sample()
        return action.item()

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        self.memory.append((torch.tensor(state.__array__().to(self.device)), torch.tensor(next_state.__array__().to(self.device)),
                            torch.tensor([action].to(self.device)), torch.tensor([reward]).to(self.device), torch.tensor([done]).to(self.device)))

        distribution = Categorical(self.net(state.unsqueeze(0)))
        self.episode_actions = torch.cat([self.episode_actions, distribution.log_prob(action).reshape(1)])
        self.episode_rewards.append(reward)




    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
  
    def reset(self):
        self.episode_actions = torch.tensor([], requires_grad=True).cuda() if self.use_cuda else torch.tensor([], requires_grad=True)
        self.episode_rewards = []

  


    def learn(self):   
        future_reward = 0
        rewards = []
        #print(len(self.episode_rewards),self.episode_rewards[::-1])
        for r in self.episode_rewards[::-1]:
            future_reward = r + self.gamma * future_reward
            rewards.append(future_reward)
        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda() if self.use_cuda else torch.tensor(rewards[::-1], dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()

        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        current_Q = self.net(state)[np.arange(0, self.batch_size), action]
        #return current_Q, loss.item()
        return None, None #Might change in future if we want to train


    def save(self):
        save_path = self.save_dir / f"mario_net_"+self.__class__.__name__+".pth"
        torch.save(
            dict(
                model=self.net.state_dict()),
            save_path
        )
        print(f"MarioNet saved to {save_path}")



    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        
        state_dict = torch.load(load_path, map_location=self.device)
        print(f"Loading model at {load_path}")
        self.net.model.load_state_dict(state_dict)


class MarioNet(nn.Module):
    '''mini cnn structure for Policy Gradient
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=-1)
        )



    def forward(self, x):
        return self.model(x)