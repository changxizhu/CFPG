from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th
import numpy as np
import copy
import time
import random
# import matplotlib.pyplot as plt
# from IPython import display

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if 'sc2' in self.args.env:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, **kwargs):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        action_norms = []
        action_means = []
        msg_norms = []
        msg_means = []
        
        while not terminated:
            if self.args.render:
                self.env.render() # just update the data

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            
            ########################################################
            # passing messages between agents, and update message data into the batch
            if getattr(self.args, "allow_messages", False) == True:
                messages =  self.mac.select_messages(self.batch, t_ep=self.t, t_env=self.t_env,
                                                   test_mode=test_mode)  # not explore but can be changed later
                messages_data = {
                    "messages": messages
                }
                self.batch.update(messages_data, ts=self.t)
                
                # norm and mean of messages
                cpu_messages = messages.to("cpu").numpy()
                msg_norms.append(np.sqrt(np.sum(cpu_messages**2)))
                msg_means.append(np.mean(cpu_messages))
                
            # add communication graph
            if getattr(self.args, "comm_graph", None) == "nearby":
                observable_ids = self.env.get_observable_agents()
                # print("observable_ids:", np.shape(observable_ids), observable_ids)
                self.batch.update({"observable_ids": observable_ids}, ts=self.t)
            
            ########################################################

            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                      test_mode=test_mode,
                                                      explore=(not test_mode))
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = th.argmax(actions, dim=-1).long()

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
                episode_return += reward
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())
                episode_return += reward

            # record actions of agents
            cpu_actions = actions.to("cpu").numpy()
            action_norms.append(np.sqrt(np.sum(cpu_actions**2)))
            action_means.append(np.mean(cpu_actions))
            
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # print("Episode return:", episode_return)
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
       
        ########################################################
        # passing messages between agents, and update message data into the batch
        if getattr(self.args, "allow_messages", False) == True:
            messages =  self.mac.select_messages(self.batch, t_ep=self.t, t_env=self.t_env,
                                                    test_mode=test_mode)  # not explore but can be changed later
            messages_data = {
                "messages": messages
            }
            self.batch.update(messages_data, ts=self.t)
        
        # add communication graph
        if getattr(self.args, "comm_graph", None) == "nearby":
            observable_ids = self.env.get_observable_agents()
            # print("observable_ids:", np.shape(observable_ids), observable_ids)
            self.batch.update({"observable_ids": observable_ids}, ts=self.t)
        ########################################################

        # Select actions in the last stored state
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode))
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = th.argmax(actions, dim=-1).long()

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "Perf_evaluate/test_" if test_mode else "Perf/train_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        
        # this is the moving average
        cur_stats["action_norms"] = np.mean(action_norms) + cur_stats.get("action_norms", 0)
        cur_stats["action_means"] = np.mean(action_means) + cur_stats.get("action_means", 0)
        cur_stats["msg_norms"] = np.mean(msg_norms) + cur_stats.get("msg_norms", 0)
        cur_stats["msg_means"] = np.mean(msg_means) + cur_stats.get("msg_means", 0)


        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()