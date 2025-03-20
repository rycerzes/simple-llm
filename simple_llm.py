import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleLLM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = None
        self.tokenizer = None
        self.weights = {}
        self.debug = False

    def load_weights(self, model_name):
        """
        load weights and config of Llama 3 models
        """
        print(f"Loading model {model_name}...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.config = {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_hidden_layers": model.config.num_hidden_layers,
            "rms_norm_eps": model.config.rms_norm_eps,
            "max_position_embeddings": model.config.max_position_embeddings,
            "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        }

        # GQA specific config
        if hasattr(model.config, "num_key_value_heads"):
            self.config["num_key_value_heads"] = model.config.num_key_value_heads
        else:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]

        print("\n===== Model Configuration =====")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("=============================\n")

        for layer_idx in range(self.config["num_hidden_layers"]):
            layer_prefix = f"model.layers.{layer_idx}."

            # attn weights
            q_weight = model.state_dict()[f"{layer_prefix}self_attn.q_proj.weight"].to(
                self.device
            )
            k_weight = model.state_dict()[f"{layer_prefix}self_attn.k_proj.weight"].to(
                self.device
            )
            v_weight = model.state_dict()[f"{layer_prefix}self_attn.v_proj.weight"].to(
                self.device
            )
            o_weight = model.state_dict()[f"{layer_prefix}self_attn.o_proj.weight"].to(
                self.device
            )

            # ff weights
            gate_weight = model.state_dict()[f"{layer_prefix}mlp.gate_proj.weight"].to(
                self.device
            )
            up_weight = model.state_dict()[f"{layer_prefix}mlp.up_proj.weight"].to(
                self.device
            )
            down_weight = model.state_dict()[f"{layer_prefix}mlp.down_proj.weight"].to(
                self.device
            )

            # layer norms
            input_norm_weight = model.state_dict()[
                f"{layer_prefix}input_layernorm.weight"
            ].to(self.device)
            post_norm_weight = model.state_dict()[
                f"{layer_prefix}post_attention_layernorm.weight"
            ].to(self.device)

            self.weights[f"layer_{layer_idx}"] = {
                "q_proj_weight": q_weight,
                "k_proj_weight": k_weight,
                "v_proj_weight": v_weight,
                "o_proj_weight": o_weight,
                "gate_proj_weight": gate_weight,
                "up_proj_weight": up_weight,
                "down_proj_weight": down_weight,
                "input_layernorm_weight": input_norm_weight,
                "post_attention_layernorm_weight": post_norm_weight,
            }

        # embedding and output weights
        self.weights["token_embd"] = model.state_dict()["model.embed_tokens.weight"].to(
            self.device
        )
        self.weights["norm_weight"] = model.state_dict()["model.norm.weight"].to(
            self.device
        )

        del model
        torch.cuda.empty_cache()

        print(
            f"Successfully loaded model with {self.config['num_hidden_layers']} layers"
        )

    def _rope_embedding(self, positions, dim, base=10000):
        """
        rotary positional embedding (RoPE) implementation
        """
        # half the dimensions because applying rotary embeddings to pairs
        half_dim = dim // 2
        if self.debug:
            print(f"RoPE: positions shape = {positions.shape}, half_dim = {half_dim}")

        freqs = 1.0 / (
            base ** (torch.arange(0, half_dim).float().to(self.device) / half_dim)
        )
        angles = positions.unsqueeze(-1) * freqs
        cos_emb = torch.cos(angles)
        sin_emb = torch.sin(angles)

        if self.debug:
            print(
                f"RoPE: cos_emb shape = {cos_emb.shape}, sin_emb shape = {sin_emb.shape}"
            )

        return cos_emb, sin_emb

    def _rotary_embedding(self, x, seq_len):
        """RoPE to input tensors"""
        if self.debug:
            print(f"Rotary input shape: {x.shape}, data type: {x.dtype}")
            print(f"Memory size: {x.nelement() * x.element_size()} bytes")

        head_dim = x.shape[-1]
        positions = torch.arange(seq_len, device=self.device)

        cos_emb, sin_emb = self._rope_embedding(positions, head_dim)

        # expand shape [1, seq_len, 1, head_dim//2] for broadcasting
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]

        if self.debug:
            print(
                f"Broadcast cos_emb shape: {cos_emb.shape}, sin_emb shape: {sin_emb.shape}"
            )

        # even and odd indices for head dimension
        x_even = x[..., 0::2]  # [batch, seq_len, num_heads, head_dim//2]
        x_odd = x[..., 1::2]  # [batch, seq_len, num_heads, head_dim//2]

        if self.debug:
            print(f"x_even shape: {x_even.shape}, x_odd shape: {x_odd.shape}")
            print(f"x_even size: {x_even.nelement()}, x_odd size: {x_odd.nelement()}")

        x_embed_even = x_even * cos_emb - x_odd * sin_emb
        x_embed_odd = x_odd * cos_emb + x_even * sin_emb

        x_embed = torch.zeros_like(x)
        x_embed[..., 0::2] = x_embed_even
        x_embed[..., 1::2] = x_embed_odd

        if self.debug:
            print(f"Rotary output shape: {x_embed.shape}")

        return x_embed

    def _rms_norm(self, x, weight, eps=1e-6):
        """
        RMSNorm
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x

    def _attention(self, q, k, v, mask=None):
        """
        Multi-head attention
        """
        # scale attention scores
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=q.dtype, device=q.device)
        )

        # causal mask (if provided)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # attention weights to values
        output = torch.matmul(attn_weights, v)

        return output

    def _feed_forward(self, x, gate_weight, up_weight, down_weight):
        """
        SwiGLU FFN
        """
        # gate and up projections
        gate = F.silu(F.linear(x, gate_weight))
        up = F.linear(x, up_weight)

        # element-wise multiplication
        intermediate = gate * up

        # down projection
        output = F.linear(intermediate, down_weight)

        return output

    def forward(self, input_ids):
        """
        forward pass

        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]

        Returns:
            logits: Tensor of logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        if self.debug:
            print("\n===== Forward Pass Info =====")
            print(f"Input shape: {input_ids.shape}")
            print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
            print(f"Device: {self.device}")
            print(f"Input dtype: {input_ids.dtype}")

        x = F.embedding(input_ids, self.weights["token_embd"])

        if self.debug:
            print(f"Embedding shape: {x.shape}, dtype: {x.dtype}")

        causal_mask = (
            torch.tril(torch.ones(seq_len, seq_len, device=self.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        if self.debug:
            print(f"Causal mask shape: {causal_mask.shape}")

        for layer_idx in range(self.config["num_hidden_layers"]):
            if self.debug and layer_idx == 0:
                print(f"\n--- Layer {layer_idx} ---")

            layer_weights = self.weights[f"layer_{layer_idx}"]

            # RMSNorm before attention
            residual = x
            x = self._rms_norm(
                x, layer_weights["input_layernorm_weight"], self.config["rms_norm_eps"]
            )

            if self.debug and layer_idx == 0:
                print(f"After RMS norm 1: {x.shape}")

            # self-attention
            hidden_size = self.config["hidden_size"]
            num_heads = self.config["num_attention_heads"]
            num_kv_heads = self.config["num_key_value_heads"]
            head_dim = self.config["head_dim"]

            if self.debug and layer_idx == 0:
                print(
                    f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}"
                )

            # qkv proj
            q = F.linear(x, layer_weights["q_proj_weight"])
            k = F.linear(x, layer_weights["k_proj_weight"])
            v = F.linear(x, layer_weights["v_proj_weight"])

            if self.debug and layer_idx == 0:
                print(
                    f"q raw shape: {q.shape}, k raw shape: {k.shape}, v raw shape: {v.shape}"
                )

            # multi-head attention reshape
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

            if self.debug and layer_idx == 0:
                print(
                    f"q reshaped: {q.shape}, k reshaped: {k.shape}, v reshaped: {v.shape}"
                )

            # RoPE to q k
            if self.debug and layer_idx == 0:
                print("\nApplying rotary embeddings to q:")
            q = self._rotary_embedding(q, seq_len)

            if self.debug and layer_idx == 0:
                print("\nApplying rotary embeddings to k:")
            k = self._rotary_embedding(k, seq_len)

            if self.debug and layer_idx == 0:
                print(f"q rotary: {q.shape}, k rotary: {k.shape}")

            # attention calculation transpose
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.debug and layer_idx == 0:
                print(
                    f"q transposed: {q.shape}, k transposed: {k.shape}, v transposed: {v.shape}"
                )

            # repeat kv for every head in GQA
            if num_heads > num_kv_heads:
                heads_per_kv = num_heads // num_kv_heads

                k = k.repeat_interleave(heads_per_kv, dim=1)
                v = v.repeat_interleave(heads_per_kv, dim=1)

                if self.debug and layer_idx == 0:
                    print(f"k after repeat: {k.shape}, v after repeat: {v.shape}")

            attn_output = self._attention(q, k, v, causal_mask)

            if self.debug and layer_idx == 0:
                print(f"Attention output: {attn_output.shape}")

            # reshape
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, hidden_size)
            )

            if self.debug and layer_idx == 0:
                print(f"Attention output reshaped: {attn_output.shape}")

            # out proj
            attn_output = F.linear(attn_output, layer_weights["o_proj_weight"])

            if self.debug and layer_idx == 0:
                print(f"After output projection: {attn_output.shape}")

            x = residual + attn_output

            if self.debug and layer_idx == 0:
                print(f"After residual 1: {x.shape}")

            # 2nd RMSNorm
            residual = x
            x = self._rms_norm(
                x,
                layer_weights["post_attention_layernorm_weight"],
                self.config["rms_norm_eps"],
            )

            if self.debug and layer_idx == 0:
                print(f"After RMS norm 2: {x.shape}")

            ff_output = self._feed_forward(
                x,
                layer_weights["gate_proj_weight"],
                layer_weights["up_proj_weight"],
                layer_weights["down_proj_weight"],
            )

            if self.debug and layer_idx == 0:
                print(f"After feed-forward: {ff_output.shape}")

            x = residual + ff_output

            if self.debug and layer_idx == 0:
                print(f"After residual 2: {x.shape}")

            if layer_idx == 0:
                self.debug = False

        # final RMSNorm
        x = self._rms_norm(x, self.weights["norm_weight"], self.config["rms_norm_eps"])

        # logits proj
        logits = F.linear(x, self.weights["token_embd"])

        if self.debug:
            print(f"Final logits shape: {logits.shape}")
            print("===== End Forward Pass =====\n")

        self.debug = True

        return logits

    def generate(self, prompt, max_length=512, temperature=1.0, top_k=0, top_p=0.9):
        """
        gen text using token-by-token generation

        Args:
            prompt: String prompt to start generation from
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling (1.0 = no change, <1.0 = more deterministic)
            top_k: Number of highest probability tokens to keep for sampling (0 = all)
            top_p: Cumulative probability threshold for nucleus sampling (1.0 = all)

        Returns:
            generated_text: String of generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        if self.debug:
            print("\n===== Generation Info =====")
            print(f"Prompt: '{prompt}'")
            print(f"Encoded input shape: {input_ids.shape}")
            print(f"Tokenized IDs: {input_ids}")

        if max_length > self.config["max_position_embeddings"]:
            max_length = self.config["max_position_embeddings"]
            print(f"Warning: max_length exceeds model capacity. Set to {max_length}")

        # greedy generation or sampling
        for i in range(max_length - input_ids.shape[1]):
            # last max_context tokens to avoid exceeding context length and save mem
            max_context = min(input_ids.shape[1], 2048)
            input_context = input_ids[:, -max_context:]

            if self.debug and i == 0:
                print(f"\n--- Generation step {i} ---")
                print(f"Context shape: {input_context.shape}")

            # logits from the model
            with torch.no_grad():
                logits = self.forward(input_context)

            if self.debug and i == 0:
                print(f"Logits shape: {logits.shape}")

            # logits for the next token (last position)
            next_token_logits = logits[:, -1, :].squeeze(0)

            if self.debug and i == 0:
                print(f"Next token logits shape: {next_token_logits.shape}")

            # temp scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(0, top_k_indices, top_k_values)

            # top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits.scatter_(0, indices_to_remove, float("-inf"))

            # sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)

            if self.debug and i == 0:
                print(
                    f"Selected token: {next_token.item()}, '{self.tokenizer.decode([next_token.item()])}'"
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if self.debug and i == 0:
                print(f"Updated input shape: {input_ids.shape}")
                self.debug = False

            if next_token.item() == self.tokenizer.eos_token_id:
                if self.debug:
                    print("EOS token generated, stopping.")
                break

        self.debug = True

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
