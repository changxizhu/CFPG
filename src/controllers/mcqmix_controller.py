from gym import spaces
import torch as th
import torch.distributions as tdist
import numpy as np
from .basic_controller import BasicMAC
from utils.comm_tools import *
from modules.comm import REGISTRY as comm_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from utils.comm_tools import *

# This multi-agent controller shares parameters between agents
class MCQMixMAC(BasicMAC):
    
    def __init__(self, scheme, groups, args):
        super(MCQMixMAC, self).__init__(scheme, groups, args)
        
        # rebuild the agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        
        # build communication
        self.msg_spaces = get_msg_spaces(args)
        self.comm_input_shape = self._get_comm_input_shape(scheme)
        self._build_comm(self.comm_input_shape)

    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        # initialize hiddent states for communication, but comm and action policy may share the same hidden state
        self.comm_hidden_states = self.comm.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
               

    def select_actions(self, ep_batch, t_ep, t_env, messages=None, bs=slice(None), test_mode=False, past_actions=None, critic=None,
                       target_mac=False, explore_agent_ids=None):
        avail_actions = ep_batch["avail_actions"][bs, t_ep]

        if t_ep is not None and t_ep > 0:
            past_actions = ep_batch["actions"][:, t_ep-1]

        if getattr(self.args, "agent", "cqmix") == "cqmix":
            raise Exception("No CQMIX agent selected (naf, icnn, qtopt)!")

        # Note batch_size_run is set to be 1 in our experiments
        if self.args.agent in ["naf", "mlp", "rnn"]:
            chosen_actions = self.forward(ep_batch[bs],
                                          t_ep,
                                          messages=messages,
                                          hidden_states=self.hidden_states[bs],
                                          test_mode=test_mode,
                                          select_actions=True,
                                          bs=bs)["actions"] # just to make sure detach
            chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
            pass
        elif self.args.agent == "icnn":
            inputs = self._build_inputs(ep_batch[bs], t_ep)
            chosen_actions = self.agent.bundle_tuned2(observation=inputs)
            chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()
            pass
        elif self.args.agent in ["cem", "cemrnn"]:
            chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)
        elif self.args.agent in ["cemrand"]:
            N = 64
            agent_inputs = self._build_inputs(ep_batch[bs], t_ep)
            hidden_states = self.hidden_states[bs].repeat(N, 1, 1)

            # Randomly sample N actions from a uniform distribution
            ftype = th.FloatTensor if not next(self.agent.parameters()).is_cuda else th.cuda.FloatTensor
            low = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_() + self.args.action_spaces[0].low[0]
            high = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_() + self.args.action_spaces[0].high[0]
            dist = tdist.Uniform(low.view(-1, self.args.n_actions), high.view(-1, self.args.n_actions))
            actions = dist.sample((N,)).detach()

            # Pick the best sampled action
            out = self.agent(agent_inputs.unsqueeze(0).expand(N, *agent_inputs.shape).contiguous().view(-1, agent_inputs.shape[-1]),
                             hidden_states if hidden_states is not None else self.hidden_states,
                             actions=actions.view(-1, actions.shape[-1]))["Q"].view(N, -1, 1)
            topk, topk_idxs = th.topk(out, 1, dim=0)
            action_prime = th.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            chosen_actions = action_prime.clone().view(ep_batch[bs].batch_size, self.n_agents,
                                                       self.args.n_actions).detach()
            pass
        else:
            raise Exception("No known agent type selected for cqmix! ({})".format(self.args.agent))


        # Now do appropriate noising
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        # Ornstein-Uhlenbeck:
        if not test_mode:  # do exploration
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)

                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args["episode_limit"]*self.args.ou_stop_episode else 0.0
                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env >= start_steps:
                    if explore_agent_ids is None:
                        x = chosen_actions.clone().zero_()
                        chosen_actions += act_noise * x.clone().normal_()
                    else:
                        for idx in explore_agent_ids:
                            x = chosen_actions[:, idx].clone().zero_()
                            chosen_actions[:, idx] += act_noise * x.clone().normal_()
                else:
                    if getattr(self.args.env_args, "scenario_name", None) is None or self.args.env_args["scenario_name"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_actions = th.from_numpy(np.array([[self.args.action_spaces[0].sample() for i in range(self.n_agents)] for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)
                    else:
                        chosen_actions = th.from_numpy(np.array([[self.args.action_spaces[i].sample() for i in range(self.n_agents)] for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)

        # For continuous actions, now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(np.asscalar(self.args.action_spaces[_aid].low[_actid]),
                                                           np.asscalar(self.args.action_spaces[_aid].high[_actid]))
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_spaces]):   # NOTE: This was added to handle scenarios like simple_reference since action space is Tuple
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].spaces[0].shape[0]):
                    chosen_actions[:, _aid, _actid].clamp_(self.args.action_spaces[_aid].spaces[0].low[_actid],
                                                           self.args.action_spaces[_aid].spaces[0].high[_actid])
                for _actid in range(self.args.action_spaces[_aid].spaces[1].shape[0]):
                    tmp_idx = _actid + self.args.action_spaces[_aid].spaces[0].shape[0]
                    chosen_actions[:, _aid, tmp_idx].clamp_(self.args.action_spaces[_aid].spaces[1].low[_actid],
                                                            self.args.action_spaces[_aid].spaces[1].high[_actid])
        return chosen_actions

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()
    

    def forward(self, ep_batch, t, messages=None, actions=None, hidden_states=None, select_actions=False, test_mode=False, use_direct=False, bs=slice(None)):
        agent_inputs = self._build_inputs(ep_batch, t, messages=messages, bs=bs, use_direct=use_direct)
        # hidden_states = self.hidden_states.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)[:ep_batch.batch_size]
        # print("Forward:", agent_inputs.shape, hidden_states.shape)
        if hidden_states is None:
            hidden_states = self.hidden_states
        # print("Forward:", agent_inputs.shape, hidden_states.shape)
        ret = self.agent(agent_inputs, hidden_states.view(ep_batch.batch_size*self.n_agents, -1), actions=actions)
        if select_actions:
            # print("Shape:", bs, ret["hidden_state"].view(ep_batch.batch_size, self.n_agents, -1).shape, self.hidden_states.shape)
            hidden_states = ret["hidden_state"].view(ep_batch.batch_size, self.n_agents, -1)
            if bs == slice(None):
                self.hidden_states = hidden_states
            else:
                self.hidden_states[bs] = hidden_states
            return ret
        agent_outs = ret["Q"]
        self.hidden_states = ret["hidden_state"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions

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
                ### using received messages, individual observations
                msg_input = batch['messages'][:, t]  # [batch_size, agents, msg_dim]
                # print("2. message shape:", msg_input.shape)
                inputs.append(build_msg_input_tensor(batch.batch_size, self.args, msg_input, comm_graph))
        else:
            if use_direct:
                # print("Direct messages used from new but with previous communication graph", t)
                inputs.append(messages.view(batch.batch_size, self.n_agents, (self.n_agents-1) * self.args.msg_dim))
            else:
                inputs.append(build_msg_input_tensor(batch.batch_size, self.args, messages, comm_graph))
            
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, -1, -1))
            
        inputs = th.cat([x.reshape(batch.batch_size*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    
    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    
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
                input_shape += scheme["messages_onehot"]["vshape"][0] * (self.n_agents - 1)  #  communication from other agents
            else:
                input_shape += scheme["messages"]["vshape"][0] * (self.n_agents - 1)
        
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
    
    
    """
    select messages. forward pass, build inputs and get input shape for communication
    """
    
    def select_messages(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, past_actions=None, critic=None,
                       target_mac=False, explore_agent_ids=None):

        if getattr(self.args, "communication", "mcqmix") == "mcqmix":
            raise Exception("No MCQMIX agent selected (naf, icnn, qtopt)!")

        # Note batch_size_run is set to be 1 in our experiments
        if self.args.communication in ["naf", "mlp", "rnn"]:
            # print("RNN:", ep_batch.batch_size, bs, self.comm_hidden_states.shape, self.comm_hidden_states[bs].shape)
            chosen_messages = self.forward_comm(ep_batch[bs],
                                          t_ep,
                                          comm_hidden_states=self.comm_hidden_states[bs],
                                          select_messages=True,
                                          bs=bs)["messages"] # just to make sure detach
            chosen_messages = chosen_messages.view(ep_batch[bs].batch_size, self.n_agents, self.args.msg_dim).detach()
            pass
        else:
            raise Exception("No known agent type selected for cqmix! ({})".format(self.args.agent))

        # Now do appropriate noising
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        # Ornstein-Uhlenbeck:
        if not test_mode:  # do exploration
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_messages.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)

                noise_scale = getattr(self.args, "ou_noise_scale", 0.3) if t_env < self.args.env_args["episode_limit"]*self.args.ou_stop_episode else 0.0
                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_messages += ou_noise
            elif exploration_mode == "gaussian":
                # print("Before using explored messages!!!!!", chosen_messages)
                start_steps = getattr(self.args, "start_steps", 0)
                msg_noise = getattr(self.args, "msg_noise", 0.1)
                # print("Start steps:", start_steps, " t_env:", t_env, " explore_agent_ids:", explore_agent_ids)
                if t_env >= start_steps:
                    if explore_agent_ids is None:
                        x = chosen_messages.clone().zero_()
                        chosen_messages += msg_noise * x.clone().normal_()
                    else:
                        for idx in explore_agent_ids:
                            x = chosen_messages[:, idx].clone().zero_()
                            chosen_messages[:, idx] += msg_noise * x.clone().normal_()
                else:
                    if getattr(self.args.env_args, "scenario_name", None) is None or self.args.env_args["scenario_name"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_messages = th.from_numpy(np.array([[self.msg_spaces[0].sample() for i in range(self.n_agents)] for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)
                    else:
                        chosen_messages = th.from_numpy(np.array([[self.msg_spaces[i].sample() for i in range(self.n_agents)] for _ in range(ep_batch[bs].batch_size)])).float().to(device=ep_batch.device)
                # print("After using explored messages!!!!!", chosen_messages)

        # For continuous messages, now clamp messages to permissible message range (necessary after exploration)
        if all([isinstance(msg_space, spaces.Box) for msg_space in self.msg_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.msg_spaces[_aid].shape[0]):
                    chosen_messages[:, _aid, _actid].clamp_(np.asscalar(self.msg_spaces[_aid].low[_actid]),
                                                           np.asscalar(self.msg_spaces[_aid].high[_actid]))

        return chosen_messages
    
        
    def forward_comm(self, ep_batch, t, messages=None, select_messages=False, comm_hidden_states=None, bs=slice(None)):
        comm_inputs = self._build_comm_inputs(ep_batch, t)
        if comm_hidden_states == None:
            comm_hidden_states = self.comm_hidden_states
        # print("Comm input:", t, comm_inputs.shape, comm_hidden_states.view(ep_batch.batch_size*self.n_agents, -1).shape, ep_batch.batch_size, messages)
        ret = self.comm(comm_inputs, comm_hidden_states.view(ep_batch.batch_size*self.n_agents, -1), messages=messages)
        comm_hidden_states = ret["hidden_state"].view(ep_batch.batch_size, self.n_agents, -1)
        if bs == slice(None):
            self.comm_hidden_states = comm_hidden_states
        else:
            self.comm_hidden_states[bs] = comm_hidden_states
        return ret
    
    
    def _build_comm(self, input_shape):
        self.comm = comm_REGISTRY[self.args.communication](input_shape, self.args)
    
        
    def comm_parameters(self):
        return self.comm.parameters()

    
    def _build_comm_inputs(self, batch, t, target_mac=False, last_target_comm=None):
        # Assumes homogenous communication with flat observations.
        # Other MACs might want to e.g. delegate building inputs for communication to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        # print("Obs shape:", batch["obs"][:, t].shape)
        if self.args.obs_last_act_comm:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id_comm:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs


    def _get_comm_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_act_comm:
            if getattr(self.args, "discretize_actions", "continuous") == "discrete":
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id_comm:
            input_shape += self.n_agents

        return input_shape


    def cem_sampling(self, ep_batch, t, bs, critic=None):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6

        ftype = th.FloatTensor if not next(self.agent.parameters()).is_cuda else th.cuda.FloatTensor
        mu = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_()
        std = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_() + 1.0
        its = 0

        maxits = 2
        agent_inputs = self._build_inputs(ep_batch[bs], t)
        hidden_states = self.hidden_states.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)[bs].repeat(N, 1, 1, 1)

        # Use feed-forward critic here, so it takes only the obs input
        critic_inputs = []
        if critic is not None:
            critic_inputs.append(ep_batch[bs]["obs"][:, t])
            critic_inputs = th.cat([x.reshape(ep_batch[bs].batch_size * self.n_agents, -1) for x in critic_inputs], dim=1)

        while its < maxits:
            dist = tdist.Normal(mu.view(-1, self.args.n_actions), std.view(-1, self.args.n_actions))
            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)

            if critic is None:
                ret = self.agent(agent_inputs.unsqueeze(0).expand(N, *agent_inputs.shape).contiguous().view(-1, agent_inputs.shape[-1]),
                                 hidden_states if hidden_states is not None else self.hidden_states,
                                 actions=actions_prime.view(-1, actions_prime.shape[-1]))
                out = ret["Q"].view(N, -1, 1)
            else:
                out, _ = critic(critic_inputs.unsqueeze(0).expand(N, *critic_inputs.shape).contiguous().view(-1, critic_inputs.shape[-1]),
                                actions=actions_prime.view(-1, actions_prime.shape[-1]))
                out = out.view(N, -1, 1)

            topk, topk_idxs = th.topk(out, Ne, dim=0)
            mu = th.mean(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            std = th.std(actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
            its += 1

        topk, topk_idxs = th.topk(out, 1, dim=0)
        action_prime = th.mean(actions_prime.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()), dim=0)
        chosen_actions = action_prime.clone().view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()

        return chosen_actions


    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)
        self.comm.cuda(device=device)
