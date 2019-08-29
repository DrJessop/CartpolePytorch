import gym
import torch
import torch.nn as nn
from collections import deque
import random


class DQNSolver(nn.Module):

    def __init__(self, state_space, action_space):
        super(DQNSolver, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.layer1 = nn.Linear(in_features=self.state_space, out_features=24)
        self.layer2 = nn.Linear(in_features=24, out_features=24)
        self.layer3 = nn.Linear(in_features=24, out_features=self.action_space)

    def forward(self, data):
        data = self.layer1(data)
        data = nn.ReLU()(data)
        data = self.layer2(data)
        data = nn.ReLU()(data)
        data = self.layer3(data)
        return data


class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, alpha, lr,
                 exploration_max, exploration_min, exploration_decay):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.dqn = DQNSolver(state_space, action_space)
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        # Create memory
        self.memory = deque(maxlen=max_memory_size)
        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.alpha = alpha
        self.l1 = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

    def remember(self, state, action, state2, reward, done):
        self.memory.append((state, action, state2, reward, done))

    def recall(self):
        memories = random.sample(self.memory, self.memory_sample_size)

        STATE = torch.zeros(0, self.state_space)
        ACTION = torch.zeros(0, 1)
        REWARD = torch.zeros(0, 1)
        STATE2 = torch.zeros(0, self.state_space)
        DONE = torch.zeros(0, 1)

        for state, action, reward, state2, done in memories:
            STATE = torch.cat((STATE, state.float()))
            ACTION = torch.cat((ACTION, action.float()))
            REWARD = torch.cat((REWARD, reward.float()))
            STATE2 = torch.cat((STATE2, state2.float()))
            DONE = torch.cat((DONE, done.float()))

        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        return torch.argmax(self.dqn(state)).unsqueeze(0).unsqueeze(0)

    def experience_replay(self):

        if self.memory_sample_size > len(self.memory):
            return

        self.optimizer.zero_grad()

        # Q-Learning update is Q(S, A) <- Q(S, A) + α[r + γ max_a Q(S', a) - Q(S, A)]
        STATE, ACTION, REWARD, STATE2, DONE = self.recall()

        target_values = REWARD + torch.mul(self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1), (~DONE.bool()).float())
        current = self.dqn(STATE)
        target = self.dqn(STATE)

        for idx in range(len(target)):
            target[idx][ACTION[idx].long()] = target_values[idx]

        target = current + self.alpha * (target - current)

        loss = self.l1(current, target)
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


def run():
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=1000000,
                     batch_size=20,
                     gamma=0.95,
                     alpha=0.99,
                     lr=0.001,
                     exploration_max=1.0,
                     exploration_min=0.01,
                     exploration_decay=0.995)

    num_episodes = 1
    while True:
        state = env.reset()
        state = torch.tensor([state]).float()
        total_reward = 0
        steps = 0
        while True:
            steps += 1
            # env.render()
            agent.dqn.zero_grad()
            action = agent.act(state)
            state_next, reward, terminal, _ = env.step(int(action))
            reward = reward if not terminal else -reward
            total_reward += reward
            reward = torch.tensor([reward]).unsqueeze(0)
            state_next = torch.tensor(state_next).float().unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next

            if terminal:
                break

            agent.experience_replay()

        print("Total steps after episode {} is {}".format(num_episodes, steps))
        num_episodes += 1


run()
