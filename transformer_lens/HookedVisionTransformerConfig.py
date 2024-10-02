import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from transformer_lens import utils
from transformer_lens.utilities.activation_functions import SUPPORTED_ACTIVATIONS

logger = logging.get_logger(__name__)

@dataclass
class MllamaVisionConfig(PretrainedConfig):
    model_type = "mllama_vision_model"

    def __init__(
        self,
        hidden_size: int = 1280,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 32,
        num_global_layers: int = 8,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5120,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        max_num_tiles: int = 4,
        intermediate_layers_indices: Optional[List[int]] = None,
        supported_aspect_ratios: Optional[List[List[int]]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if supported_aspect_ratios is None:
            if max_num_tiles != 4:
                raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")
            supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if intermediate_layers_indices is None:
            intermediate_layers_indices = [3, 7, 15, 23, 30]

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_global_layers = num_global_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.vision_output_dim = vision_output_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_eps = norm_eps
        self.max_num_tiles = max_num_tiles
        self.intermediate_layers_indices = intermediate_layers_indices
        self.supported_aspect_ratios = supported_aspect_ratios
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> PretrainedConfig:
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning("You are using a model of type %s to instantiate a model of type %s. This is not supported for all configurations of models and can yield errors.",
                config_dict["model_type"], cls.model_type)

        return cls.from_dict(config_dict, **kwargs)

@dataclass
class MllamaTextConfig(PretrainedConfig):
    model_type = "mllama_text_model"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> PretrainedConfig:
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning("You are using a model of type %s to instantiate a model of type %s. This is not supported for all configurations of models and can yield errors.",
                config_dict["model_type"], cls.model_type)

        return cls.from_dict(config_dict, **kwargs)

@dataclass
class MllamaConfig(PretrainedConfig):
    model_type = "mllama"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=128256,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaVisionConfig):
            self.vision_config = vision_config

        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = MllamaTextConfig(**text_config)
        elif isinstance(text_config, MllamaTextConfig):
            self.text_config = text_config

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> PretrainedConfig:
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            vision_config = config_dict["vision_config"]
            text_config = config_dict["text_config"]

        return cls(
            vision_config=MllamaVisionConfig.from_dict(vision_config),
            text_config=MllamaTextConfig.from_dict(text_config),
            **kwargs
        )

'''
    def __post_init__(self):
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    "d_model %d is not divisible by d_head %d."
                    "n_heads was inferred to be %d, rounding down the ratio.",
                    self.d_model,
                    self.d_head,
                    self.n_heads,
                )

        if self.seed is not None:
            self.set_seed_everywhere(self.seed)
        if self.use_local_attn:
            assert self.window_size is not None, "window_size must be specified for local attention"
            assert self.attn_types is not None, "attn_types must be specified for local attention"
        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp: int = self.d_model * 4
            assert self.act_fn is not None, "act_fn must be specified for non-attn-only models"
            assert (
                self.act_fn in SUPPORTED_ACTIVATIONS
            ), f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"
        if self.initializer_range < 0 and self.init_mode == "gpt2":
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)
        if self.initializer_range < 0 and self.init_mode != "gpt2":
            # This is the gain parameter for the weight initialisation
            self.initializer_range = 1.0

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            # If d_vocab is not set, it'll be inferred from tokenizer_name or from a tokenizer
            # explicitly passed to HookedTransformer initialisation.
            self.d_vocab_out = self.d_vocab

        if self.positional_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = self.d_head

        if self.num_experts is not None:
            assert (
                self.experts_per_token is not None
            ), "experts_per_token must be set if num_experts is set"
        if self.experts_per_token is not None:
            assert (
                self.num_experts is not None
            ), "num_experts must be set if experts_per_token is set"

        # The number of parameters in attention layers (ignoring biases and layer norm). 4 because W_Q, W_K, W_V and W_O
        self.n_params = self.n_layers * ((self.d_model * self.d_head * self.n_heads * 4))
        if not self.attn_only:
            assert self.d_mlp is not None  # mypy
            # Number of parameters in MLP layers (ignoring biases and layer norm). 2 because W_in and W_out
            mlp_params_per_layer = self.d_model * self.d_mlp * (2 + self.gated_mlp)

            if self.num_experts:
                # If we are using MoE, we multiply by num_experts, and add the expert gate parameters (d_model * num_experts)
                mlp_params_per_layer = (mlp_params_per_layer + self.d_model) * self.num_experts
            self.n_params += self.n_layers * mlp_params_per_layer

        if self.device is None:
            self.device = utils.get_device()

        if self.n_devices > 1:
            assert (
                torch.cuda.device_count() >= self.n_devices
            ), f"Not enough CUDA devices to support n_devices {self.n_devices}"

        if self.use_attn_scale and self.attn_scale == -1.0:
            self.attn_scale = np.sqrt(self.d_head)

        assert self.default_prepend_bos in [
            True,
            False,
        ], f"padding_side must be either True or False, but {self.default_prepend_bos} is given"

    @classmethod
    def unwrap(cls, config: Union[Dict, "HookedTransformerConfig"]) -> HookedTransformerConfig:
        """
        Convenience function to avoid duplicate code from a common way config is passed to various components
        """
        return HookedTransformerConfig.from_dict(config) if isinstance(config, Dict) else config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedTransformerConfig:
        """
        Instantiates a `HookedTransformerConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedTransformerConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def is_layer_norm_activation(self) -> bool:
        return self.act_fn is not None and self.act_fn.endswith("_ln")

'''