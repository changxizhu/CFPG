from gym import spaces
import numpy as np
import torch as th


"""
Some tools used for communication

"""

def get_msg_spaces(args):
    msg_spaces = []
    for i in range(args.n_agents):
        msg_spaces.append(spaces.Discrete(args.msg_dim) if getattr(args, "msg_dtype", "continuous") == "discrete" \
            else spaces.Box(low=-int(args.comm_range), high=int(args.comm_range), shape=(args.msg_dim,), dtype=np.float32))
    
    return msg_spaces


def build_msg_input_tensor(batch_size, args, messages, comm_graph):
    assert messages.shape[-2:] == (args.n_agents, args.msg_dim), "Warning!! Wrong message input (%s)!=(%s)!!" % (str(messages.shape), str((-1, args.n_agents, args.msg_dim)))
    ### reshape the messages
    msg = messages.view(batch_size, args.n_agents, args.msg_dim)
    ### using received messages, individual observations
    messages_input = []
    
    for i in range(args.n_agents):
        mask_msg = th.cat((msg[:, :i, :], msg[:, i+1:, :]), 1).view(batch_size, args.n_agents-1, args.msg_dim)
        messages_input.append(mask_msg)
    messages_input = th.stack(messages_input, 1).view(batch_size, args.n_agents, args.n_agents-1, args.msg_dim)
    
    """
    Using communication graph when building message input
    halfcheetah 6: communication with neighboring agents (without itself), 1--2,4; 2--1,3; 3--2; 4--1,5; 5--4,6; 6--5
    """
    if comm_graph is not None:
        messages_input = messages_input * comm_graph.view(batch_size, args.n_agents, args.n_agents-1, args.msg_dim)

    # dimensions: [batch_size, agents, (agents-1)*msg_dim]
    return messages_input.view(batch_size, args.n_agents, (args.n_agents-1) * args.msg_dim)
