import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):

    def __init__(self, state_space, action_space, dropout):
        super(PolicyNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.layer1 = nn.Linear(in_features=self.state_space, out_features=124)
        self.layer2 = nn.Linear(in_features=124, out_features=24)
        self.layer3 = nn.Linear(in_features=24, out_features=self.action_space)
        self.dropout = dropout

    def forward(self, data):
        data = self.layer1(data)
        data = nn.ReLU()(data)
        data = nn.Dropout(p=self.dropout)(data)
        data = self.layer2(data)
        data = nn.ReLU()(data)
        data = self.layer3(data)
        return nn.Softmax()(data)  # This gives the probability of picking actions


class MCPolicyAgent:

    def __init__(self, state_space, action_space, gamma, lr,
                 dropout):

        # Define Policy Net Info
        self.state_space = state_space
        self.action_space = action_space
        self.policy = PolicyNetwork(state_space, action_space, dropout)

        # Store trajectories for m episodes
        self.policy_history = torch.tensor([])
        self.reward_history = torch.tensor([])

        # Learning parameters
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state):
        # Sample actions under policy θ
        probabilities = self.policy(state)
        c = Categorical(probabilities)
        action = c.sample()
        pi_action = c.log_prob(action)
        return action, pi_action

    def normalize_rewards(self):
        return (self.reward_history - self.reward_history.mean()) / self.reward_history.std()

    def add_to_trajectory(self, probability, reward):
        self.policy_history = torch.cat((self.policy_history, probability))
        self.reward_history = torch.cat((self.reward_history, reward))

    def update_policy(self):

        # Compute discounted reward for entire episode
        discounted_reward = 0

        # Discount future rewards back to the present using gamma
        for idx in range(len(self.reward_history) - 1, -1, -1):
            discounted_reward = self.reward_history[idx] + self.gamma * discounted_reward
            self.reward_history[idx] = discounted_reward

        self.normalize_rewards()  # Advantage
        policy_gradient = torch.dot(self.policy_history, self.reward_history)
        loss = -policy_gradient

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy_history = torch.tensor([])  # Re-initialize after episode
        self.reward_history = torch.tensor([])


def run():
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = MCPolicyAgent(state_space=observation_space,
                          action_space=action_space,
                          gamma=0.99,
                          lr=0.001,
                          dropout=0.6)

    num_episodes = 1
    while True:
        state = env.reset()
        state = torch.tensor([state]).float()
        total_reward = 0
        steps = 0
        agent.policy.zero_grad()
        while True:
            steps += 1
            if num_episodes > 500:
                env.render()

            action, pi_a = agent.act(state)
            state_next, reward, terminal, _ = env.step(int(action))
            reward = reward if not terminal else -reward
            total_reward += reward
            reward = torch.tensor([reward])
            agent.add_to_trajectory(pi_a, reward)
            state = torch.tensor([state_next]).float()

            if terminal:
                break

        agent.update_policy()

        print("Total steps after episode {} is {}".format(num_episodes, steps))
        num_episodes += 1


run()
