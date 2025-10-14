from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .mcqmix_controller import MCQMixMAC
from .comm_cqmix_controller import CCQMixMAC
from .mcqmix_attention_controller import MCQMixAttentionMAC

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["cqmix_mac"] = CQMixMAC
REGISTRY["mcqmix_mac"] = MCQMixMAC
REGISTRY["comm_cqmix_mac"] = CCQMixMAC
REGISTRY["mcqmix_attention_mac"] = MCQMixAttentionMAC