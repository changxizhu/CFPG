import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPComm(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPComm, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.comm_rnn_hidden_dim)
        self.fc2 = nn.Linear(args.comm_rnn_hidden_dim, args.comm_rnn_hidden_dim)
        self.fc3 = nn.Linear(args.comm_rnn_hidden_dim, args.msg_dim)

        self.comm_return_logits = getattr(self.args, "comm_return_logits", False)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.comm_rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, messages=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        if self.comm_return_logits:
            messages = self.fc3(x)
        else:
            messages = torch.tanh(self.fc3(x))
        return {"messages": messages, "hidden_state": hidden_state}