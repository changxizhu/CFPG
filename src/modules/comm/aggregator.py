import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Aggregator(nn.Module):
    def __init__(self, args):
        super(Aggregator, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        # Soft
        self.q = nn.Linear(args.msg_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.msg_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.msg_dim, args.attention_dim)

        # Decoding 输入自己的h_i与x_i，输出自己动作的概率分布
        self.decoding = nn.Linear(args.attention_dim, args.msg_dim)  # decode the messages from attended vectors

    def forward(self, messages):
        # shape for messages: batch_size, n_agents, msg_dim
        q = self.q(messages).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(messages).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(messages)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        x = []
        attention_weights = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # agent i的q，(batch_size, 1, args.attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的k
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的v

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, args.attention_dim， n_agents - 1)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0) # 变成(batch_size, args.attention_dim， n_agents - 1)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim，n_agents - 1) = (batch_size, 1，n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # 归一化
            # scaled_score = score / np.sqrt(self.args.attention_dim)
            scaled_score = score * 10

            # softmax得到权重 and smooth the weights
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)
            soft_weight = torch.clamp(soft_weight, min=0.0001, max=0.9999)
            soft_weight = soft_weight/(soft_weight.sum(-1).unsqueeze(-1))
            
            # print("weights:", soft_weight)
            
            # 加权求和，注意三个矩阵的最后一维是n_agents - 1维度，得到(batch_size, args.attention_dim)
            x_i = (v_i * soft_weight).sum(dim=-1)
            
            # reconstruct weights for agents: insert 0 for index=i
            insert_zeros = torch.zeros(soft_weight.shape[0], soft_weight.shape[1], 1)
            if self.args.use_cuda:
                insert_zeros = insert_zeros.cuda()
            attention_wgh = torch.cat([soft_weight[:, :, :i], insert_zeros, soft_weight[:, :, i:]], dim=-1)
            attention_weights.append(attention_wgh.squeeze())  # average over batch_size: (batch_size, n_agents -1)
            x.append(x_i)

        # 合并每个agent的h与x
        x = torch.stack(x, dim=1).view(-1, self.args.attention_dim)  # (batch_size * n_agents, args.attention_dim)
        attention_weights = torch.stack(attention_weights, dim=0) # (batch_size, n_agents, n_agents-1)
        output = self.decoding(x)  # output (batch_size * n_agents, msg_dim)

        return output.view(-1, self.args.n_agents, self.args.msg_dim), attention_weights.view(-1, self.args.n_agents, self.args.n_agents)

