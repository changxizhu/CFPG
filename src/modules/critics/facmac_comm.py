import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .facmac import FACMACCritic, FACMACDiscreteCritic

class FACMACCommCritic(FACMACCritic):
    def __init__(self, scheme, args):
        super(FACMACCommCritic, self).__init__(scheme, args)
        self.n_messages = (self.n_agents-1) * args.msg_dim
        self.n_observations = scheme["obs"]["vshape"]
        
    
    def forward(self, inputs, messages, actions, hidden_state=None):
        # Attention!!!! the input only includes observations
        
        if messages is not None and actions is not None:
            inputs = th.cat([inputs.view(-1, self.n_observations), \
                             messages.contiguous().view(-1, self.n_messages), \
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state


    def _get_input_shape(self, scheme):
        # with communication, add dimensions
        input_shape = scheme["obs"]["vshape"] + (self.n_agents-1) * scheme["messages"]["vshape"][0]
        return input_shape