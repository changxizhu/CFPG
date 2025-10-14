import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralCommCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralCommCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        
        self.n_messages = self.n_agents * args.msg_dim
        self.n_state = scheme["state"]["vshape"]


    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None


    def forward(self, inputs, messages, actions, hidden_state=None):
        # Attention!!!! the input includes state, messages, actions
        if messages is not None and actions is not None:
            inputs = th.cat([inputs.view(-1, self.n_state), \
                             messages.contiguous().view(-1, self.n_messages), \
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state


    def _get_input_shape(self, scheme):
        # with communication, using global state and all messages
        input_shape = scheme["state"]["vshape"] + self.n_agents * scheme["messages"]["vshape"][0]
        return input_shape

