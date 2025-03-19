import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleLLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.load_weights(model_name)

    def load_weights(self, model_name):
        """Load weights and config from HuggingFace model"""
        print(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        self.weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = model.config

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads

        if hasattr(self.config, "num_key_value_heads"):
            self.num_kv_heads = self.config.num_key_value_heads
        else:
            self.num_kv_heads = self.num_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.num_layers = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size

        self.weight_map = self._map_weight_names()

        print(
            f"Model loaded: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_size} hidden size"
        )

    def _map_weight_names(self):
        """Map weight names based on LLaMA architecture patterns"""
        weight_map = {
            "token_emb": "model.embed_tokens.weight",
            "norm_final": "model.norm.weight",
            "lm_head": "lm_head.weight",
        }

        # Map layer-specific weights
        for i in range(self.num_layers):
            layer_prefix = f"model.layers.{i}."
            weight_map[f"layer_{i}_input_norm"] = (
                f"{layer_prefix}input_layernorm.weight"
            )
            weight_map[f"layer_{i}_post_attn_norm"] = (
                f"{layer_prefix}post_attention_layernorm.weight"
            )

            # Attention weights
            weight_map[f"layer_{i}_q_proj"] = f"{layer_prefix}self_attn.q_proj.weight"
            weight_map[f"layer_{i}_k_proj"] = f"{layer_prefix}self_attn.k_proj.weight"
            weight_map[f"layer_{i}_v_proj"] = f"{layer_prefix}self_attn.v_proj.weight"
            weight_map[f"layer_{i}_o_proj"] = f"{layer_prefix}self_attn.o_proj.weight"

            # FFN weights (SwiGLU)
            weight_map[f"layer_{i}_gate_proj"] = f"{layer_prefix}mlp.gate_proj.weight"
            weight_map[f"layer_{i}_up_proj"] = f"{layer_prefix}mlp.up_proj.weight"
            weight_map[f"layer_{i}_down_proj"] = f"{layer_prefix}mlp.down_proj.weight"

        return weight_map

    def _rmsnorm(self, x, weight):
        """RMSNorm used in LLaMA"""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        return weight * x

    def _apply_rotary_emb(self, q, k, seq_positions):
        """Apply rotary positional embeddings to queries and keys"""
        device = q.device
        head_dim = q.shape[-1]

        if not hasattr(self, "freqs_cis"):
            theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device) / head_dim)
            seq_idx = torch.arange(
                4096, device=device
            )  # Pre-compute for max sequence length
            emb = seq_idx.unsqueeze(1) * theta.unsqueeze(0)
            self.freqs_cis = torch.stack([torch.cos(emb), torch.sin(emb)], dim=-1)

        max_seq_len = seq_positions.max().item() + 1
        freqs_cis = self.freqs_cis[:max_seq_len]
        freqs_cis = freqs_cis[seq_positions]  # [bs, seq_len, dim//2, 2]

        cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]

        cos = cos.view(*cos.shape, 1).expand(-1, -1, -1, 2)
        sin = sin.view(*sin.shape, 1).expand(-1, -1, -1, 2)
        cos = cos.reshape(*cos.shape[:-2], head_dim)
        sin = sin.reshape(*sin.shape[:-2], head_dim)

        q_rotate = torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1)
        k_rotate = torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1)

        q = q * cos + q_rotate * sin
        k = k * cos + k_rotate * sin

        return q, k

    def forward(self, input_ids):
        """Transformer forward pass using only PyTorch operations"""
        batch_size, seq_length = input_ids.shape

        hidden_states = F.embedding(
            input_ids, self.weights[self.weight_map["token_emb"]]
        )

        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        for layer_idx in range(self.num_layers):
            residual = hidden_states
            hidden_states = self._rmsnorm(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_input_norm"]],
            )

            # Self-attention
            # queries, keys, values
            q = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_q_proj"]],
            )
            k = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_k_proj"]],
            )
            v = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_v_proj"]],
            )

            q_dim = q.size(-1)
            k_dim = k.size(-1)
            v_dim = v.size(-1)

            q_head_dim = q_dim // self.num_heads
            k_head_dim = k_dim // self.num_kv_heads
            v_head_dim = v_dim // self.num_kv_heads

            q = q.view(batch_size, seq_length, self.num_heads, q_head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, seq_length, self.num_kv_heads, k_head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_length, self.num_kv_heads, v_head_dim).transpose(
                1, 2
            )

            if self.num_heads > self.num_kv_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=0)
                v = v.repeat_interleave(repeats, dim=0)
                # Reshape the batch dimension back
                k = k.reshape(batch_size, self.num_heads, seq_length, k_head_dim)
                v = v.reshape(batch_size, self.num_heads, seq_length, v_head_dim)

            q, k = self._apply_rotary_emb(q, k, position_ids)

            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                q_head_dim
            )

            causal_mask = (
                torch.triu(
                    torch.ones(seq_length, seq_length, device=input_ids.device),
                    diagonal=1,
                )
                .bool()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attention_scores.masked_fill_(causal_mask, float("-inf"))

            attention_weights = F.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_weights, v)

            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)

            attn_output = F.linear(
                context_layer,
                self.weights[self.weight_map[f"layer_{layer_idx}_o_proj"]],
            )

            hidden_states = attn_output + residual

            residual = hidden_states
            hidden_states = self._rmsnorm(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_post_attn_norm"]],
            )

            gate_proj = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_gate_proj"]],
            )
            up_proj = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_up_proj"]],
            )

            gate_act = F.silu(gate_proj)
            ffn_output = gate_act * up_proj

            down_proj = F.linear(
                ffn_output,
                self.weights[self.weight_map[f"layer_{layer_idx}_down_proj"]],
            )

            hidden_states = down_proj + residual

        hidden_states = self._rmsnorm(
            hidden_states, self.weights[self.weight_map["norm_final"]]
        )

        logits = F.linear(hidden_states, self.weights[self.weight_map["lm_head"]])

        return logits

    def generate(self, prompt, max_length=512, temperature=1.0, top_p=0.9):
        """Generate text using greedy decoding or sampling"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        device = next(iter(self.weights.values())).device
        input_ids = input_ids.to(device)

        seq_len = input_ids.shape[1]

        kv_cache = {}
        for layer_idx in range(self.num_layers):
            kv_cache[layer_idx] = {"k": None, "v": None}

        for _ in range(max_length - seq_len):
            with torch.no_grad():
                if _ == 0:
                    logits = self.forward(input_ids)
                else:
                    logits = self._forward_with_cache(
                        input_ids[:, -1:], kv_cache, seq_len - 1
                    )

                next_token_logits = logits[:, -1, :]

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                if temperature == 0:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = torch.zeros_like(
                        next_token_logits, dtype=torch.bool
                    ).scatter_(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                seq_len += 1

                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

    def _forward_with_cache(self, input_ids, kv_cache, position_offset):
        """Forward pass with KV caching for efficient generation"""
        batch_size, seq_length = input_ids.shape

        hidden_states = F.embedding(
            input_ids, self.weights[self.weight_map["token_emb"]]
        )

        position_ids = torch.arange(
            position_offset, position_offset + seq_length, device=input_ids.device
        ).unsqueeze(0)

        for layer_idx in range(self.num_layers):
            residual = hidden_states
            hidden_states = self._rmsnorm(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_input_norm"]],
            )

            q = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_q_proj"]],
            )
            k = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_k_proj"]],
            )
            v = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_v_proj"]],
            )

            q_dim = q.size(-1)
            k_dim = k.size(-1)
            v_dim = v.size(-1)

            q_head_dim = q_dim // self.num_heads
            k_head_dim = k_dim // self.num_kv_heads
            v_head_dim = v_dim // self.num_kv_heads

            q = q.view(batch_size, seq_length, self.num_heads, q_head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, seq_length, self.num_kv_heads, k_head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_length, self.num_kv_heads, v_head_dim).transpose(
                1, 2
            )

            if self.num_heads > self.num_kv_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=0)
                v = v.repeat_interleave(repeats, dim=0)
                k = k.reshape(batch_size, self.num_heads, seq_length, k_head_dim)
                v = v.reshape(batch_size, self.num_heads, seq_length, v_head_dim)

            q, k = self._apply_rotary_emb(q, k, position_ids)

            if kv_cache[layer_idx]["k"] is not None:
                past_key = kv_cache[layer_idx]["k"]
                past_value = kv_cache[layer_idx]["v"]
                k = torch.cat((past_key, k), dim=2)
                v = torch.cat((past_value, v), dim=2)

            kv_cache[layer_idx]["k"] = k
            kv_cache[layer_idx]["v"] = v

            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                q_head_dim
            )

            # Softmax and attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_weights, v)

            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.view(batch_size, seq_length, self.hidden_size)

            attn_output = F.linear(
                context_layer,
                self.weights[self.weight_map[f"layer_{layer_idx}_o_proj"]],
            )

            hidden_states = attn_output + residual

            residual = hidden_states
            hidden_states = self._rmsnorm(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_post_attn_norm"]],
            )

            # SwiGLU activation as used in LLaMA
            gate_proj = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_gate_proj"]],
            )
            up_proj = F.linear(
                hidden_states,
                self.weights[self.weight_map[f"layer_{layer_idx}_up_proj"]],
            )

            gate_act = F.silu(gate_proj)
            ffn_output = gate_act * up_proj

            down_proj = F.linear(
                ffn_output,
                self.weights[self.weight_map[f"layer_{layer_idx}_down_proj"]],
            )

            hidden_states = down_proj + residual

        hidden_states = self._rmsnorm(
            hidden_states, self.weights[self.weight_map["norm_final"]]
        )

        logits = F.linear(hidden_states, self.weights[self.weight_map["lm_head"]])

        return logits
