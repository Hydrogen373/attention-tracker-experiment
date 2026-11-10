"""Hugging Face Qwen3 wrapper. Reads attention *without* changing model parameters."""
import numpy as np
import torch

from .records import PromptCase


class QwenAttentionModel:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        max_input_tokens: int = 1024,
        dtype: str = "bfloat16",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-8B requires a CUDA GPU for the intended experiment")
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer with character offsets is required")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # Required to return raw attention weights
        ).eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        # Hooking captures only one attention row per layer, then drops the full
        # matrix from the layer's returned result. This mitigates O(L*N^2) VRAM.
        # Each eager attention layer still temporarily materializes its matrix.
        self._rows = {}
        self._inst_indices = None
        self._data_indices = None
        self._handles = []
        for index, layer in enumerate(self.model.model.layers):
            handle = layer.self_attn.register_forward_hook(
                lambda module, args, output, layer_id=index: self._attention_hook(layer_id, output)
            )
            self._handles.append(handle)

    def _attention_hook(self, layer_id, output):
        if self._inst_indices is None:
            return output
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError(
                "The selected Transformers/Qwen3 version does not expose eager "
                "attention as (attn_output, attention_weights)."
            )
        weights = output[1]  # [batch=1, heads, query_length, key_length]
        row = weights[0, :, -1, :].float()
        inst = row.index_select(-1, self._inst_indices.to(row.device)).sum(-1)
        data = row.index_select(-1, self._data_indices.to(row.device)).sum(-1)
        self._rows[layer_id] = (inst / (inst + data + 1e-8)).cpu().numpy()

        # Decoder layer keeps the attention result only for optional reporting.
        # Return None in that slot so all layer matrices aren't retained.
        return (output[0], None, *output[2:])

    def _encode(self, case: PromptCase):
        messages = [
            {"role": "system", "content": case.system},
            {"role": "user", "content": case.user},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True,
            return_tensors="pt", truncation=False
        )
        offsets = tokens.pop("offset_mapping")[0].tolist()
        if tokens["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(
                f"Sample {case.sample_id}: {tokens['input_ids'].shape[1]} tokens > "
                f"--max-input-tokens={self.max_input_tokens}; skipped, not truncated."
            )

        # Locate text ranges on the actual Qwen chat-template string, then use
        # tokenizer offset mapping. Avoid guessing offsets from raw token lengths.
        inst_start = rendered.find(case.instruction)
        if inst_start < 0:
            raise ValueError(f"Instruction span missing in {case.sample_id}")
        inst_end = inst_start + len(case.instruction)
        data_ranges = []
        cursor = inst_end
        for part in case.data_parts:
            if not part:
                continue
            start = rendered.find(part, cursor)
            if start < 0:
                raise ValueError(f"Data span missing or out of order in {case.sample_id}")
            data_ranges.append((start, start + len(part)))
            cursor = start + len(part)

        inst_ids, data_ids = [], []
        for idx, (a, b) in enumerate(offsets):
            if b <= a:  # special tokens with empty offset
                continue
            if a < inst_end and b > inst_start:
                inst_ids.append(idx)
            elif any(a < end and b > start for start, end in data_ranges):
                data_ids.append(idx)
        if not inst_ids or not data_ids:
            raise ValueError(f"Empty instruction/data token selection in {case.sample_id}")
        if set(inst_ids) & set(data_ids):
            raise ValueError(f"Overlapping token regions in {case.sample_id}")
        return tokens["input_ids"], inst_ids, data_ids

    @torch.inference_mode()
    def head_map(self, case: PromptCase) -> np.ndarray:
        input_ids, instruction_ids, data_ids = self._encode(case)
        # This feature is deterministic under eval mode and no generation.
        self._inst_indices = torch.tensor(instruction_ids, dtype=torch.long)
        self._data_indices = torch.tensor(data_ids, dtype=torch.long)
        self._rows.clear()
        try:
            output = self.model(
                input_ids=input_ids.to(self.model.get_input_embeddings().weight.device),
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            del output
            if len(self._rows) != self.num_layers:
                raise RuntimeError(
                    f"Only {len(self._rows)} of {self.num_layers} Qwen3 attention hooks fired"
                )
            result = np.stack([self._rows[i] for i in range(self.num_layers)])
            if result.shape != (self.num_layers, self.num_heads):
                raise RuntimeError(f"Unexpected attention map shape: {result.shape}")
            return result
        finally:
            self._inst_indices = None
            self._data_indices = None
            self._rows.clear()

    @torch.inference_mode()
    def generate(self, case: PromptCase, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": case.system},
            {"role": "user", "content": case.user},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        ).to(self.model.get_input_embeddings().weight.device)
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError(f"Generation prompt too long: {case.sample_id}")
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
