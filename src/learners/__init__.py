from .cq_learner import CQLearner
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner
from .g2anet_learner import G2ANetLearner

from .comm_actor_learner import CommActorLearner
from .comm_attention_actor_learner import CommAttentionActorLearner

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
REGISTRY["g2anet_learner"] = G2ANetLearner

REGISTRY["comm_actor_learner"] = CommActorLearner
REGISTRY["comm_attention_actor_learner"] = CommAttentionActorLearner
