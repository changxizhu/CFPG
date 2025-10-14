from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np
from .mcqmix_controller import MCQMixMAC
from utils.comm_tools import *
from modules.comm import REGISTRY as comm_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from utils.comm_tools import *
from modules.comm.aggregator import Aggregator

# This multi-agent controller shares parameters between agents
class MCQMixAttentionMAC(MCQMixMAC):
    
    def __init__(self, scheme, groups, args):
        super(MCQMixAttentionMAC, self).__init__(scheme, groups, args)
        
        self.aggregator = Aggregator(args)

    
    
    def attention_parameters(self):
        return self.aggregator.parameters()
    

    def _build_inputs(self, batch, t, messages=None, bs=slice(None), use_direct=False, target_mac=False, last_target_action=None):
        ##########################################
        """
        use communication graph or not? decided by the setting
        """
        comm_graph = None
        # use graph from args
        if isinstance(self.args.comm_graph, list):
            comm_grah = th.tensor(self.args.comm_graph).view(self.args.n_agents, self.args.n_agents-1, 1).repeat(batch.batch_size,1, self.args.msg_dim)
            if self.args.use_cuda:
                comm_grah = comm_grah.cuda()
        # use graph from batch data
        if getattr(self.args, "comm_graph", None) == "nearby":
            comm_graph = batch["observable_ids"][:, t].view(batch.batch_size, self.args.n_agents, self.args.n_agents-1, 1).repeat(1, 1, 1, self.args.msg_dim)
        ##########################################
        
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        inputs = []
        inputs.append(batch["obs"][:, t])
        
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])

        """
        add global messages into the input
        """
        if messages is None:
            if getattr(self.args, "allow_messages", False) == True:
                ### using received messages, individual observations; [batch_size, agents, msg_dim]
                inputs.append(batch['messages'][:, t])
        else:
            inputs.append(messages.view(batch.batch_size, self.n_agents, self.args.msg_dim))
            
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1))
            
        inputs = th.cat([x.reshape(batch.batch_size*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            if getattr(self.args, "discretize_actions", False):
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        
        # add the size of messages to the shape
        if getattr(self.args, "allow_messages", False) == True:
            if getattr(self.args, "msg_dtype", "continuous") == "discrete":
                input_shape += scheme["messages_onehot"]["vshape"][0]  #  communication from other agents
            else:
                input_shape += scheme["messages"]["vshape"][0]
        
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)
        self.comm.cuda(device=device)
        self.aggregator.cuda(device=device)
