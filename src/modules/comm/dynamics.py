import math
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, scheme, args):
        super(DynamicsModel, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.n_observations = scheme["obs"]["vshape"]
        self.n_messages = args.msg_dim

        self.input_shape = self._get_input_shape(scheme) + self.n_actions

        """
        Initialization method for the generator
        """
        self.fc1 = nn.Linear(self.input_shape, args.dynamics_hidden_dim)  # input without action (1 hot)
        self.fc2 = nn.Linear(args.dynamics_hidden_dim, self.n_observations)
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initialization for the parameters of the graph generator
        """
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.fc1.weight.data, gain=gain)
        # nn.init.xavier_normal_(self.fc1.bias.data, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight.data, gain=gain)
        # nn.init.xavier_normal_(self.fc2.bias.data, gain=gain)

    def forward(self, inputs, messages, actions):
        if messages is not None and actions is not None:
            # print("Dynamics dimensions:", self.n_observations, self.n_messages, self.n_actions)
            inputs = th.cat([inputs.view(-1, self.n_observations), \
                             messages.contiguous().view(-1, self.n_messages), \
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        
        x = self.fc1(inputs)
        x = th.tanh(x)
        x = self.fc2(x)
        return x

    def _get_input_shape(self, scheme):
        # with communication, add dimensions
        input_shape = scheme["obs"]["vshape"] + scheme["messages"]["vshape"][0]
        return input_shape
