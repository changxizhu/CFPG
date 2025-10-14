REGISTRY = {}

from .mlp_comm import MLPComm
from .rnn_comm import RNNComm

REGISTRY["mlp"] = MLPComm
REGISTRY["rnn"] = RNNComm