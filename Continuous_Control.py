#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


# For linux env, install python.
# get_ipython().run_line_magic('%bash', 'pip -q install ./python')


# In[2]:


from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import importlib
import torch
import matplotlib.pyplot as plt


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Reacher.app"`
# - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
# - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
# - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
# - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
# - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
# - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Reacher.app")
# ```

# In[3]:


# Use this to load from linux.
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')


# In[4]:


# Use this to load from Mac application.
#env = UnityEnvironment(file_name='./Reacher 2.app')
env = UnityEnvironment(file_name='/home/dagriff2/p2_continuous_control/Reacher_Linux_NoVis/Reacher.x86_64')

# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[5]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
# 
# The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.
# 
# Run the code cell below to print some information about the environment.

# In[6]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agent's performance, if it selects an action at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  
# 
# Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# In[7]:


# env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
# states = env_info.vector_observations                  # get the current state (for each agent)
# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# while True:
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[8]:


# env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

# # DDPG Implementation
# 

# In[9]:

'''
get_ipython().run_cell_magic('writefile', 'model.py', '\nimport numpy as np\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\ndef hidden_init(layer):\n    fan_in = layer.weight.data.size()[0]\n    lim = 1. / np.sqrt(fan_in)\n    return (-lim, lim)\n\nclass Actor(nn.Module):\n    """Actor (Policy) Model."""\n\n    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=280):\n        """Initialize parameters and build model.\n        Params\n        ======\n            state_size (int): Dimension of each state\n            action_size (int): Dimension of each action\n            seed (int): Random seed\n            fc1_units (int): Number of nodes in first hidden layer\n            fc2_units (int): Number of nodes in second hidden layer\n        """\n        super(Actor, self).__init__()\n        self.seed = torch.manual_seed(seed)\n        self.fc1 = nn.Linear(state_size, fc1_units)\n        self.bn1 = nn.BatchNorm1d(num_features=fc1_units)\n        self.fc2 = nn.Linear(fc1_units, fc2_units)\n#         self.bn2 = nn.BatchNorm1d(num_features=fc2_units)\n        self.fc3 = nn.Linear(fc2_units, action_size)\n        self.reset_parameters()\n\n    def reset_parameters(self):\n        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))\n        self.fc2.weight.data.uniform_(-3e-3, 3e-3)\n\n    def forward(self, state):\n        """Build an actor (policy) network that maps states -> actions."""\n        x = F.relu(self.fc1(state))\n        x = self.bn1(x)\n        x = F.relu(self.fc2(x))\n#         x = self.bn2(x)\n        return F.tanh(self.fc3(x))\n\n\nclass Critic(nn.Module):\n    """Critic (Value) Model."""\n\n    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=280):\n        """Initialize parameters and build model.\n        Params\n        ======\n            state_size (int): Dimension of each state\n            action_size (int): Dimension of each action\n            seed (int): Random seed\n            fcs1_units (int): Number of nodes in the first hidden layer\n            fc2_units (int): Number of nodes in the second hidden layer\n        """\n        super(Critic, self).__init__()\n        self.seed = torch.manual_seed(seed)\n        self.fcs1 = nn.Linear(state_size, fcs1_units)\n        self.bn1 = nn.BatchNorm1d(num_features=fcs1_units)\n        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)\n        self.fc3 = nn.Linear(fc2_units, 1)\n#         self.fc4 = nn.Linear(fc3_units, 1)\n        self.reset_parameters()\n\n    def reset_parameters(self):\n        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))\n        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))\n        self.fc3.weight.data.uniform_(-3e-3, 3e-3)\n#         self.fc4.weight.data.uniform_(-3e-3, 3e-3)\n\n    def forward(self, state, action):\n        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""\n        xs = F.leaky_relu(self.fcs1(state))\n        xs = self.bn1(xs)\n        x = torch.cat((xs, action), dim=1)\n        x = F.leaky_relu(self.fc2(x))\n#         x = F.leaky_relu(self.fc3(x))\n        return self.fc3(x)\n')
'''

# In[10]:

'''
get_ipython().run_cell_magic('writefile', 'ddpg_agent.py', '\nimport numpy as np\nimport random\nimport copy\nfrom collections import namedtuple, deque\n\nfrom model import Actor, Critic\n\nimport torch\nimport torch.nn.functional as F\nimport torch.optim as optim\n\nBUFFER_SIZE = int(1e6)  # replay buffer size\nBATCH_SIZE = 128        # minibatch size\nGAMMA = 0.99            # discount factor\nTAU = 1e-3              # for soft update of target parameters\nLR_ACTOR = 1e-3         # learning rate of the actor \nLR_CRITIC = 1e-3        # learning rate of the critic\nWEIGHT_DECAY = 0 #0.0001   # L2 weight decay\n\nUPDATE_TIMES = 10 # Update the network this many times when the learn function is called.\nSTEPS_TO_UPDATE = 20 # Call the learn function \'UPDATE_TIMES\' every \'STEPS_TO_UPDATE\'\n\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n\nclass Agent():\n    """Interacts with and learns from the environment."""\n    \n    def __init__(self, state_size, action_size, random_seed):\n        """Initialize an Agent object.\n        \n        Params\n        ======\n            state_size (int): dimension of each state\n            action_size (int): dimension of each action\n            random_seed (int): random seed\n        """\n        self.steps = 0\n        \n        self.state_size = state_size\n        self.action_size = action_size\n        self.seed = random.seed(random_seed)\n\n        # Actor Network (w/ Target Network)\n        self.actor_local = Actor(state_size, action_size, random_seed).to(device)\n        self.actor_target = Actor(state_size, action_size, random_seed).to(device)\n        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)\n\n        # Critic Network (w/ Target Network)\n        self.critic_local = Critic(state_size, action_size, random_seed).to(device)\n        self.critic_target = Critic(state_size, action_size, random_seed).to(device)\n        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)\n\n        # Noise process\n        self.noise = OUNoise(action_size, random_seed)\n\n        # Replay memory\n        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)\n    \n    def step(self, state, action, reward, next_state, done):\n        """Save experience in replay memory, and use random sample from buffer to learn."""\n        # Save experience / reward\n        self.memory.add(state, action, reward, next_state, done)\n\n        # Increment step counter.\n        self.steps += 1\n        \n        # Learn, if enough samples are available in memory\n        if len(self.memory) > BATCH_SIZE and self.steps % STEPS_TO_UPDATE == 0:\n            for i in range(UPDATE_TIMES):\n                experiences = self.memory.sample()\n                self.learn(experiences, GAMMA)\n\n    def act(self, state, add_noise=True):\n        """Returns actions for given state as per current policy."""\n        state = torch.from_numpy(state).float().to(device)\n        self.actor_local.eval()\n        with torch.no_grad():\n            action = self.actor_local(state).cpu().data.numpy()\n        self.actor_local.train()\n        if add_noise:\n            action += self.noise.sample()\n        return np.clip(action, -1, 1)\n\n    def reset(self):\n        self.noise.reset()\n\n    def learn(self, experiences, gamma):\n        """Update policy and value parameters using given batch of experience tuples.\n        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))\n        where:\n            actor_target(state) -> action\n            critic_target(state, action) -> Q-value\n\n        Params\n        ======\n            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s\', done) tuples \n            gamma (float): discount factor\n        """\n        states, actions, rewards, next_states, dones = experiences\n\n        # ---------------------------- update critic ---------------------------- #\n        # Get predicted next-state actions and Q values from target models\n        actions_next = self.actor_target(next_states)\n        Q_targets_next = self.critic_target(next_states, actions_next)\n        # Compute Q targets for current states (y_i)\n        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))\n        # Compute critic loss\n        Q_expected = self.critic_local(states, actions)\n        critic_loss = F.mse_loss(Q_expected, Q_targets)\n        # Minimize the loss\n        self.critic_optimizer.zero_grad()\n        critic_loss.backward()\n        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)\n        self.critic_optimizer.step()\n\n        # ---------------------------- update actor ---------------------------- #\n        # Compute actor loss\n        actions_pred = self.actor_local(states)\n        actor_loss = -self.critic_local(states, actions_pred).mean()\n        # Minimize the loss\n        self.actor_optimizer.zero_grad()\n        actor_loss.backward()\n        self.actor_optimizer.step()\n\n        # ----------------------- update target networks ----------------------- #\n        self.soft_update(self.critic_local, self.critic_target, TAU)\n        self.soft_update(self.actor_local, self.actor_target, TAU)                     \n\n    def soft_update(self, local_model, target_model, tau):\n        """Soft update model parameters.\n        θ_target = τ*θ_local + (1 - τ)*θ_target\n\n        Params\n        ======\n            local_model: PyTorch model (weights will be copied from)\n            target_model: PyTorch model (weights will be copied to)\n            tau (float): interpolation parameter \n        """\n        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):\n            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)\n\nclass OUNoise:\n    """Ornstein-Uhlenbeck process."""\n\n    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):\n        """Initialize parameters and noise process."""\n        self.mu = mu * np.ones(size)\n        self.theta = theta\n        self.sigma = sigma\n        self.seed = random.seed(seed)\n        self.reset()\n\n    def reset(self):\n        """Reset the internal state (= noise) to mean (mu)."""\n        self.state = copy.copy(self.mu)\n\n    def sample(self):\n        """Update internal state and return it as a noise sample."""\n        x = self.state\n        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])\n        self.state = x + dx\n        return self.state\n\nclass ReplayBuffer:\n    """Fixed-size buffer to store experience tuples."""\n\n    def __init__(self, action_size, buffer_size, batch_size, seed):\n        """Initialize a ReplayBuffer object.\n        Params\n        ======\n            buffer_size (int): maximum size of buffer\n            batch_size (int): size of each training batch\n        """\n        self.action_size = action_size\n        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)\n        self.batch_size = batch_size\n        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])\n        self.seed = random.seed(seed)\n    \n    def add(self, state, action, reward, next_state, done):\n        """Add a new experience to memory."""\n        e = self.experience(state, action, reward, next_state, done)\n        self.memory.append(e)\n    \n    def sample(self):\n        """Randomly sample a batch of experiences from memory."""\n        experiences = random.sample(self.memory, k=self.batch_size)\n\n        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)\n        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)\n        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)\n        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)\n        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)\n\n        return (states, actions, rewards, next_states, dones)\n\n    def __len__(self):\n        """Return the current size of internal memory."""\n        return len(self.memory)')
'''

# In[11]:


import ddpg_agent
importlib.reload(ddpg_agent)


# In[12]:


agent = ddpg_agent.Agent(state_size=env_info.vector_observations[0].shape[0], 
              action_size=brain.vector_action_space_size, 
              random_seed=10)


# In[ ]:


def ddpg(n_episodes=500, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores_list = []
    max_score = -np.Inf
    
    for i_episode in range(1, n_episodes+1):
        # Reset the agent's state.
        env_info = env.reset(train_mode=True)[brain_name]
        # Get the current state.
        states = env_info.vector_observations
        # Reset the agent's noise parameter.
        agent.reset()
        scores = np.zeros(num_agents)
        # For each time step in episode, get action, act, and learn.
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)         # select an action
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = env_info.vector_observations          # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)             
            states = next_states
            scores += rewards        
            if np.any(dones):                                   # exit loop when episode ends
                break
#             # Get actions
#             actions = agent.act(states)
# #             actions = np.clip(actions, -1, 1)
#             # Act in the environment
#             env_info = env.step(actions)[brain_name]
            
#             # Get next states, rewards, dones, and scores.
#             next_states = env_info.vector_observations         # get next state (for each agent)
#             rewards = env_info.rewards                        # get reward (for each agent)
#             dones = env_info.local_done                        # see if episode finished
#             scores += env_info.rewards                         # update the score (for each agent)
            
            
#             for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
#                 agent.step(state, action, reward, next_state, done)  
                
#             states = next_states
#             scores += rewards        
#             if np.any(dones):                                   # exit loop when episode ends
#                 break
            
#             # Get next states, rewards, dones, and scores.
#             next_states = env_info.vector_observations         # get next state (for each agent)
#             rewards = env_info.rewards                        # get reward (for each agent)
#             dones = env_info.local_done                        # see if episode finished
#             scores += env_info.rewards                         # update the score (for each agent)
            
#             # Update the agent.
#             agent.step(states, actions, rewards, next_states, dones)
            
#             # Update the next state.
#             states = next_states                               # roll over states to next time step
            
#             if np.any(dones):                                  # exit loop if episode finished
#                 break
        # Add final score to the scores queue.
        score = np.mean(scores)
        scores_deque.append(score)
        scores_list.append(score)
        with open('scores_list.txt', 'a') as f:
            f.write("{}\n".format(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        # If checkpoint reached, store agent's score.

        if len(scores_deque) >= 100 and np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'final_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'final_checkpoint_critic.pth')
            break


        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
            with open('progress.txt', 'a') as f:
                f.write('\n\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores

# Run ddpg algorithm, and save final output.
scores = ddpg()
torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

# Plot performance
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('performance.png')
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # D4PG Implementation
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




