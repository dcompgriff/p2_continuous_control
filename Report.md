# P2 Conitnuous Control Project Report

Author: Daniel Griffin

## Problem Overview

The goal of this project was to develop an agent that could learn to control 20 robotic arms. Specifically, the goal was to develop a reinforcement learning algorithm that would hold each of 20 arms in a particular specified position. To do this, we required an agent that could accept a relatively high dimensional and complex environment, and could propose actions on the continuous real line from -1 to +1. The agent needed to be complex enough to understand the environment and popose actions in the environment, while being sample effecient enough to train in a reasonable amount of time (less than a few hours using a GPU). The problem was defined by the Unity ML agent 'Reacher' environment. I solved the problem using the DDPG algorithm as described in [this](https://arxiv.org/pdf/1509.02971.pdf) paper, with a few minor alterations to the algorithm, described in the 'Algorithm' section below.

## Learning Environment

States: 33 dimension vector with continuour real values describing the arm positions, angular momentums and velocities, etc...
Actions: 4 dimension vector with continuous real values for controling the joints of an arm.
Rewards: On each step +0.1 reward for the arm keeping its position in the proper location.
Number of Agents: 20 agents, 1 brain.
Solved Criteria: +30 averaged across all 20 agents for 100 consecutive episodes. 

## Algorithm

### DDPG

The main algorithm I used can be found described in [this](https://arxiv.org/pdf/1509.02971.pdf) paper. The basic principle of this algorithm was to combine two areas of major advancement in the deep reinforcement learning space. The first was to utilize the 'actor-critic' method of framing the learning problem, where the actor is a parameterized probabalistic model of a policy, and the critic is a parameterized model of the value function Q(s,a). The sedond advancement used was the optimization algorithm and structures used by the DQN algorithm. Specifically, the use of a 'replay buffer' which is periodically sampled from to create training data set batches for stochastically updating a deep neural netowrk, and the use of 'target' and 'local' networks which decouple the parameters used for generating a set of target labels from the parameters of the network being updated. 

DDPG is a:
1. Model Free
2. Off Policy
3. Policy Based Method
4. Actor-Critic Method (Models policy, as well as Q(s,a) function for optimal updates)
5. Neural Network Function Approximators
6. With Correlated Continuous Noise Exploration (Basically a Correlated Random Walk for Exploration.) (Similar to MCMC exploration kind of thing)

Needed to use policy based methods to model complex continuous action space policy. 
Could attempt to write the problem in terms of DQN networks, but this would be highly complicated, especially since the Q-Learning approach requires an argmax over the Q(s,a) values of the environment (which can be done but is a complex optimization problem, which has it's own complexeties, and would also slow down learning)

### Model Free

DDPG is a 'model free' method, meaning that the algorithm does not have a model of the environment dynamics. This means that classical dynamic programing methods can not be used to solve the problem. Instead, methods that either build a model of the environment over time, or implicitly side step the need for learning a model over time must be used. DDPG side steps the need for building a model by developing a function that estimates the value of state action pairs given past experience. By using this value estimate function Q(s,a) along with the bellman update equations, we can use what's known as a 'temporal difference' optimization method where Q(s,a) is recursively defined to provide a directed update to the value estimate function for any particular state action pair.

### Off Policy

DDPG is an 'off policy' method meaning that it uses one policy to generate experiences, and then uses those generated experiences to update a target policy. Reinforcement algorithms need to have some component of exploration to ensure that the policies they learn will conitnue to converge towards an optimal policy. This means that the algorithm's policy for generating experiences must sometimes explore non optimal policies. Yet, the optimal policy may be totally deterministic and never explores. It could even be a stochastic policy, yet stochastically optimal. The DDPG algorithm uses the widely applied off policy method known as Q-Learning, albeit with some modifications (Specifically the actor implicitly performs the 'argmax' by simply learning to output a single action, and adjusting that action output through a parameterized model, guided by the value functions.)


### Actor-Critic Method

An actor critic method is defined by 3 major components. The first is that it uses a parameterized model to represent the agent's policy. The second is that the value function Q(s,a) is also represented by a parameterized model. The third is that the two are combined through the use of an expected loss function that uses a baseline, where that baseline is the Q(s,a) value function, effectively reducing the optimization update step to something similar to a temporal difference, bootstraped estimate. 

### Neural Network Function Approximators

The DDPG algorithm uses a method of function approximation analogous to that used on the DQN reinforcement learning framework. Neural networks are the function approximators, a 'memory buffer' is randomly sampled from to lessen the serial correlation between data points (But not totally), and two networks are used for updates, target and local neural network approximation method. The DDPG algorithm re-uses both of these ideas for representing and optimizing both the actor and the critic.

### Continuous Action Space Exploration

The final major component of the algorithm is how exploration is defined and performed. The paper uses a correlated random noise parameter to generate a special kind of random walk in the action space. The algorithm creates a mean function for each state, and updates the mean to be the current mean plus a random noise parameter. This effectively generates a correlated noise process that has an element of 'inertia' to keep the process exploring in a direction correlated to the previous example. In alignment with the original paper, the Ornstein-Uhlenbeck process was used to generate a temporally correlated exploration of the action space. 


## Tweaks to the DDPG Algorithm

### Cliped Gradients

It has been shown in previous research that gradient explosion can happen when performing updates to an algorithm. To reduce the effect of this 'exploding gradient' problem, gradient magnitdes are often clipped to have a maximum and minimum value. This gradient clipping can also be useful for mitigating the use of data generated by an earlier policy to update the current policy. In this implementation, I added gradient clipping to both the actor and the critic networks.

### Decayed Noise Exploration

The noise component was a critical factor to the performance of the system. Modifying the noise magnitude had the single greatest effect on the performance of the agent. In p1 and the classical algorithms, we would typically decay the exploration parameter over time to ensure that our learned agent would have good performance. Using the standard noise process defined in the DDPG paper allowed for the agent to reach around +18 in total score, which would then drop rapidly afterwards. To improve the agent's learning capability, a noise magnitude parameter was added to decay the rate of exploration. Through experimentation, it was found that if the decay was too low, the system would learn quickly and top out, or top out and then decreases in performance. If the decay parameter was too high, the system would never get to the right level of performance. But when set just right, the system learns quickly, tops out in performance, but then maintains performance over time without degrading. The final example of this behavior is shown in the plot of rewards below.

### Batch Normalization

Just as in the DDPG paper, I added batch normalization. Normalizaiton of inputs is critical when developing neural networks that have inputs varied magnitued, or extremely large or small magnitudes. Without normalization, activations can become saturated and learn extremely slowly. This is even more prevalant when a squeeze based activation is used such as a tanh or sigmoid function (as in the case with this environment). Thus, I added batch normalization to the first layer of both the actor and the critic models. I tried using batch normalization on multiple layers, but learning didn't occur quickly or to the desired level.

### Soft Updates

The final component of the algorithm was the 'soft update' of the target model. In DQN, the target model was always updated by latching in (aka copying) the parameters of the current model to the target model. In DDPG on the other hand, the parameters of the target model are updated in a 'soft' manner. Each weight value in the network is set to be 0.1% of the current model's value for that weight, 99.9% the target model's value for that weight. 

### Periodic Updates

In the DDPG paper implementation, their algorithm performs learning at every step. This is very time consuming, and didn't seem to effect overall network performance. So, we instead use an approach analagous to that used for the DQN network in p1, where we would only update the target network after a fixed number of steps. When an update was performed, 10 mini-batches of data were sampled from the network. This delay and update method helps to further decouple the serial correlation of data points by allowing the replay memory to refill with newer experiences, prodiving a better estimate of the performance of the new policy given the previous gradient update.

### Attempted Variations

Many variations of the algorithm and optimization method were tried including increasing/decreasing model complexity, changing batch size, changing learning rate, changing noise level, and changing the replay buffer size. The most critical choices to the success of the algorithm reaching the performance threshold quickly and maintaining it was the noise level, the batch size, and the learning rate.

### Final Model and Hyperparameters

#### Actor Model

* num layers: 3
* layer 1: FC -> ReLU -> Batch Norm
* layer 2: FC -> ReLU
* layer 3: FC -> Tanh

#### Critic Model

* num layers: 2
* layer 1: FC -> Leaky ReLU -> Batch Norm
* layer 2: FC (with layer1 state state and action concat as input) -> Leaky ReLU
* layer 3: FC

#### DDPG Algorithm Hyperparameters

* Optimizer: Adam
* BUFFER_SIZE = int(1e6)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor 
* LR_CRITIC = 1e-4        # learning rate of the critic
* WEIGHT_DECAY = 0 #0.0001   # L2 weight decay
* EPSILON = 1.0           # explore vs exploit noise
* EPSILON_DECAY = 0.9999 # decay rate for noise process
* UPDATE_TIMES = 10 # Update the network this many times when the learn function is called.
* STEPS_TO_UPDATE = 20 # Call the learn function 'UPDATE_TIMES' every 'STEPS_TO_UPDATE'

## Final Results

![Learning Performance](https://github.com/dcompgriff/p2_continuous_control/blob/master/performance.png "Performance Graph")

The graph above shows that with the network's final parameters, the agent was able to learn very quickly and to maintain it's knowledge for a long period of time. A key to getting the algorithm to maintain the performance over a long period of time was to set the epsilon decay parameter so that it decayed quick enough that the algorithm would not start to loose it's performance bennefits (which would happen if the epsilon decay was too small such as 0.99).

## Ideas for Future Work

* Sensitive to amount of data used for learning (Batch size):
    * One key issue is that the networks were both sensitive to the batch size used for training.  Some methods to reduce this include:
        * Bag/Boost multiple neural network models on the training batch.
        * Prioritize the replay buffer so that smaller batch sizes are more effective.
        * Dynamically adjust the batch size used over time with it being smaller at the beginning, and larger near the end.
* Sensitive to learning on each time step:
    * Another key issue is that the network was sensitive to performing updates after each step made. Methods to try to adjust for this include:
        * Incorporate the decision to learn into the data set. Combine temporal decay with entropy based confusion metric.
        * Online exploration with offline continuous learning. Generating experiences and then training each time step is slow. We can instead take multiple steps, and while generating experiences, perform more time training and evaluating multiple models on the data set.
* Use another method. D4PG, PPO, Deep Bayesian RL:
    * A final method to try to to improve performance is to try another algorithm such as D4PG, PPO, Deep Bayesian RL, or another non-gradient based optimization method such as genetic algorithms are PSO.
        




























