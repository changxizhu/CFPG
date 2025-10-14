REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner


from .episode_attention_runner import EpisodeAttentionRunner
REGISTRY["episode_attention"] = EpisodeAttentionRunner


from .parallel_attention_runner import ParallelAttentionRunner
REGISTRY["parallel_attention"] = ParallelAttentionRunner
