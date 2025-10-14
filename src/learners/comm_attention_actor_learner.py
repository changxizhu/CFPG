import copy
from email import message
from components.episode_buffer import EpisodeBatch
from modules.critics.facmac import FACMACCritic
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
from utils.comm_tools import *

"""
The agents will use messages for mixed Q-values and actors
"""

class CommAttentionActorLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())
        self.comm_params = list(mac.comm_parameters()) + list(mac.attention_parameters())  # parameters of communication

        self.critic = FACMACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))
        
        
        
        """
        optimize communication by using two optimizers
        """
        ### optimizers of message function
        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.comm_optimiser = RMSprop(params=self.comm_params, lr=args.comm_lr_one, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.comm_optimiser = Adam(params=self.comm_params, lr=args.comm_lr_one, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        messages = batch["messages"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        pdist = th.nn.PairwiseDistance(p=2)

        """
        Train the critic batched, no gradients backpropagated to action and message functions
        """
        ### Attention!! No target message is used!!!
        ### 1. use current messages and actions for critic
        q_taken = []
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            # use detach which prevent gradients
            critic_out, self.critic.hidden_states = self.critic(inputs, actions[:, t:t + 1].detach(),
                                                                self.critic.hidden_states)
            if self.mixer is not None:
                critic_out = self.mixer(critic_out.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
            q_taken.append(critic_out)

        q_taken = th.stack(q_taken, dim=1)

        ### 2. select target messages
        self.target_mac.init_hidden(batch.batch_size)
        target_msgs = []
        for t in range(batch.max_seq_length):
            # use messages in target mac
            msg_target_outs = self.target_mac.select_messages(batch, t_ep=t, t_env=None, test_mode=True,
                                                              critic=self.target_critic, target_mac=True)
            # aggregate messages
            agg_msg_target_outs, attention_weights = self.target_mac.aggregator(msg_target_outs)

            target_msgs.append(agg_msg_target_outs)
        target_msgs = th.stack(target_msgs, dim=1)  # Concat over time
        
        ### 3. select target actions based on targeted messages
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)   # If it's necessary, I still need to investigate
        for t in range(batch.max_seq_length):
            # use messages in target mac
            agent_target_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, messages=target_msgs[:, t:t+1].detach(), test_mode=True,
                                                               critic=self.target_critic, target_mac=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        ### 4. build messages and actions input for 
        target_vals = []
        self.target_critic.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            target_critic_out, \
            self.target_critic.hidden_states = self.target_critic(target_inputs, target_actions[:, t:t+1].detach(),
                                                                  self.target_critic.hidden_states)
            if self.mixer is not None:
                target_critic_out = self.target_mixer(target_critic_out.view(batch.batch_size, -1, 1),
                                                      batch["state"][:, t:t+1])
            target_vals.append(target_critic_out)
        target_vals = th.stack(target_vals, dim=1)

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets = rewards.expand_as(target_vals) + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals
        td_error = (targets.detach() - q_taken)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.critic_grad_norm_clip)
        self.critic_optimiser.step()


        """
        Train the actor batched, no gradients backpropagated to message functions
        """
        # train actor with fixed messages
        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        actions_norm = []
        for t in range(batch.max_seq_length-1):
            sampled_actions = actions[:, t:t + 1]
            agent_outs = self.mac.forward(batch, t=t, messages=messages[:, t:t+1].detach(), select_actions=True, use_direct=True)["actions"].view(batch.batch_size,
                                                                                           self.n_agents,
                                                                                           self.n_actions)

            actions_norm.append(pdist(sampled_actions, agent_outs).mean().item())

            q, self.critic.hidden_states = self.critic(self._build_inputs(batch, t=t), agent_outs, self.critic.hidden_states)
            if self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1), batch["state"][:, t:t+1])
            mac_out.append(agent_outs)
            chosen_action_qvals.append(q)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out
        
        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()
        
        
        """
        Train the messages batched --- in a parallel way, using the same learning rate!!
        """
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        # 1. select current messages, in order to keep track of the parameters
        # 2. train communication with current messages
        comm_mac_out = []
        mac_out = []
        comm_chosen_action_qvals = []
        agg_messages_norm = []
        actions_comm_norm = []
        for t in range(batch.max_seq_length-1):
            # sampled messages
            sampled_msg_input = messages[:, t:t+1]
            sampled_actions = actions[:, t:t + 1]
            # select current messages 
            msgs_outs = self.mac.forward_comm(batch, t=t)["messages"].view(batch.batch_size, self.n_agents, self.args.msg_dim)
            # aggregate messages
            agg_msgs_outs, attention_weights = self.mac.aggregator(msgs_outs)
            # use current messages to evaluate at this point
            agent_outs = self.mac.forward(batch, t=t, messages=agg_msgs_outs, select_actions=True, use_direct=True)["actions"].view(batch.batch_size,
                                                                                        self.n_agents,
                                                                                        self.n_actions)
            # calculate 2-norm for sampled messages and current messages
            agg_messages_norm.append(pdist(sampled_msg_input.view(batch.batch_size, self.n_agents, self.args.msg_dim), agg_msgs_outs).mean().item())
            actions_comm_norm.append(pdist(sampled_actions.view(batch.batch_size, self.n_agents, self.n_actions), agent_outs).mean().item())

            obs_input = self._build_inputs(batch, t=t)
            q, self.critic.hidden_states = self.critic(obs_input, agent_outs, self.critic.hidden_states)

            if self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1), batch["state"][:, t:t+1])
            comm_mac_out.append(agg_msgs_outs)   # regularize the communication function
            mac_out.append(agent_outs)
            comm_chosen_action_qvals.append(q)
        comm_mac_out = th.stack(comm_mac_out, dim=1)
        mac_out = th.stack(mac_out, dim=1)
        comm_chosen_action_qvals = th.stack(comm_chosen_action_qvals, dim=1)
        comm_pi = comm_mac_out
        pi = mac_out
        
        # Compute the actor loss
        pg_comm_loss = -comm_chosen_action_qvals.mean() + (comm_pi**2).mean() * 1e-3 + (pi**2).mean() * 1e-3

        # Optimise communication
        self.comm_optimiser.zero_grad()
        pg_comm_loss.backward()
        comm_grad_norm = th.nn.utils.clip_grad_norm_(self.comm_params, self.args.grad_norm_clip)
        self.comm_optimiser.step()
        
        if getattr(self.args, "target_update_mode", "hard") == "hard":
            self._update_targets()
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_step/critic_loss", loss.item(), t_env)
            self.logger.log_stat("grad_step/critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss_step/target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("loss_step/q_taken_mean", q_taken.sum().item() / mask_elems, t_env)
            self.logger.log_stat("loss_step/pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("grad_step/agent_grad_norm", agent_grad_norm, t_env)
            self.logger.log_stat("loss_step/pg_comm_loss", pg_comm_loss.item(), t_env)
            self.logger.log_stat("grad_step/comm_grad_norm", comm_grad_norm, t_env)
            
            self.logger.log_stat("grad_step/actions_dist", np.mean(actions_norm), t_env)
            self.logger.log_stat("grad_step/agg_messages_dist", np.mean(agg_messages_norm), t_env)
            self.logger.log_stat("grad_step/actions_comm_dist", np.mean(actions_comm_norm), t_env)
            
            self.log_stats_t = t_env


    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    # build input for the critic
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        
        if self.args.recurrent_critic:
            # The individual Q conditions on the global action-observation history and individual action
            inputs.append(batch["obs"][:, t].repeat(1, self.args.n_agents, 1).view(bs, self.args.n_agents, -1))
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t].repeat(1, self.args.n_agents, 1).
                                                view(bs, self.args.n_agents, -1)))
                else:
                    inputs.append(batch["actions"][:, t - 1].repeat(1, self.args.n_agents, 1).
                                  view(bs, self.args.n_agents, -1))
        else:
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    
    def _build_msg_inputs(self, batch, t, messages=None):        
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
            # print("comm_graph:", comm_graph.numpy())
        ##########################################
        
        ### using received messages, individual observations
        msg_input = batch['messages'][:, t] if messages is None else messages # [batch_size, agents, msg_dim]
        messages_input = build_msg_input_tensor(batch.batch_size, self.args, msg_input, comm_graph)
        return messages_input.view(batch.batch_size, 1, self.n_agents, -1)


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.comm_optimiser.state_dict(), "{}/copt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.comm_optimiser.load_state_dict(
            th.load("{}/copt.th".format(path), map_location=lambda storage, loc: storage))