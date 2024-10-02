from typing import cast
import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_mllama_weights(mllama, cfg: HookedTransformerConfig):
    state_dict = {}

    # Embedding weights
    state_dict["embed.W_E"] = mllama.model.embed_tokens.weight

    # Check if the model uses Grouped Query Attention (GQA)
    using_gqa = cfg.n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    # Loop through each layer of the model
    for l in range(cfg.n_layers):
        # Self-attention weights
        W_Q = mllama.model.layers[l].self_attn.q_proj.weight
        W_K = mllama.model.layers[l].self_attn.k_proj.weight
        W_V = mllama.model.layers[l].self_attn.v_proj.weight
        W_O = mllama.model.layers[l].self_attn.o_proj.weight

        # Handle GQA if applicable
        if using_gqa:
            W_K = W_K[:n_kv_heads]
            W_V = W_V[:n_kv_heads]

        state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = W_Q
        state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K
        state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V
        state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = W_O

        # Initialize biases as zero tensors
        state_dict[f"model.layers.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"model.layers.{l}.attn.b_K"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"model.layers.{l}.attn.b_V"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"model.layers.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        # Feed-forward network (MLP) weights
        state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = mllama.model.layers[l].mlp.gate_proj.weight
        state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = mllama.model.layers[l].mlp.up_proj.weight
        state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = mllama.model.layers[l].mlp.down_proj.weight

        # Initialize MLP biases as zero tensors
        state_dict[f"model.layers.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"model.layers.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        # Layer normalization weights
        state_dict[f"model.layers.{l}.input_layernorm.weight"] = mllama.model.layers[l].input_layernorm.weight
        state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = mllama.model.layers[l].post_attention_layernorm.weight

        # Cross-attention layers if applicable
        if hasattr(mllama.model.layers[l], 'cross_attn'):
            state_dict[f"model.layers.{l}.cross_attn_attn_gate"] = mllama.model.layers[l].cross_attn_attn_gate
            state_dict[f"model.layers.{l}.cross_attn_mlp_gate"] = mllama.model.layers[l].cross_attn_mlp_gate

            state_dict[f"model.layers.{l}.cross_attn.q_proj.weight"] = mllama.model.layers[l].cross_attn.q_proj.weight
            state_dict[f"model.layers.{l}.cross_attn.k_proj.weight"] = mllama.model.layers[l].cross_attn.k_proj.weight
            state_dict[f"model.layers.{l}.cross_attn.v_proj.weight"] = mllama.model.layers[l].cross_attn.v_proj.weight
            state_dict[f"model.layers.{l}.cross_attn.o_proj.weight"] = mllama.model.layers[l].cross_attn.o_proj.weight

            state_dict[f"model.layers.{l}.cross_attn.q_norm.weight"] = mllama.model.layers[l].cross_attn.q_norm.weight
            state_dict[f"model.layers.{l}.cross_attn.k_norm.weight"] = mllama.model.layers[l].cross_attn.k_norm.weight

    # Final layer normalization
    state_dict["model.norm.weight"] = mllama.model.norm.weight

    # LM head weights
    state_dict["lm_head.weight"] = mllama.lm_head.weight

    return state_dict
