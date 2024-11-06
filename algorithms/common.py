from torch import Tensor
import numpy as np


class common_process:
    def __init__(self,memory,args,network):
        self.args = args
        self.memory = memory
        self.network = network
        self.device = 'cuda:0' if self.args.is_GPU == True else 'cpu'
    def sample_data(self):
        batch = self.memory.sample()
        self.batch_size = len(self.memory)
        rewards = Tensor(np.array(batch.reward))
        values = Tensor(batch.value)
        masks = Tensor(np.array(batch.mask))
        self.actions = Tensor(np.array(batch.action))
        self.states = Tensor(np.array(batch.state))
        self.oldlogproba = Tensor(np.array(batch.logproba))
        self.returns = Tensor(self.batch_size)
        deltas = Tensor(self.batch_size)
        self.advantages = Tensor(self.batch_size)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(self.batch_size)):
            self.returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.args.gamma * prev_value * masks[i] - values[i]
            self.advantages[i] = deltas[i] + self.args.gamma * self.args.lamda * prev_advantage * masks[i]
            prev_return = self.returns[i]
            prev_value = values[i]
            prev_advantage = self.advantages[i]
        if self.args.advantage_norm:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + self.args.EPS)
        return self.batch_size, self.oldlogproba, self.returns, self.advantages

    def get_minbatch(self):
        minibatch_ind = np.random.choice(self.batch_size, self.args.minibatch_size, replace=False)
        minibatch_states = self.states[minibatch_ind].to(self.device)
        minibatch_actions = self.actions[minibatch_ind].to(self.device)
        minibatch_oldlogproba = self.oldlogproba[minibatch_ind].to(self.device)
        minibatch_newlogproba = self.network.get_logproba(minibatch_states, minibatch_actions)
        minibatch_advantages = self.advantages[minibatch_ind].to(self.device)
        minibatch_returns = self.returns[minibatch_ind].to(self.device)
        minibatch_newvalues = self.network._forward_critic(minibatch_states).flatten()
        return minibatch_newlogproba,minibatch_oldlogproba,minibatch_advantages,minibatch_returns,minibatch_newvalues
