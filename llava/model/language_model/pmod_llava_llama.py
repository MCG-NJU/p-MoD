#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig
from llava.model.language_model.modeling_llama_pmod import LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import re
import math

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class PmodLlavaConfig(LlamaConfig):
    model_type = "pmod_llava_llama"

class PmodLlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = PmodLlavaConfig

    def __init__(self, config: LlamaConfig):
        super(PmodLlavaLlamaModel, self).__init__(config) # calls LLavaMetaModel.__init__() then LlamaModel.__init__

        if getattr(config, "mod_target_layers", None) is not None:
            self.build_mod_layers(config)

    def initialize_mod_modules(self, model_args):
        self.config.mod_target_layers = model_args.mod_target_layers
        self.config.mod_target_token_type = model_args.mod_target_token_type

        self.config.mod_router_factor = model_args.mod_router_factor
        self.config.mod_weight_norm = model_args.mod_weight_norm

        self.config.mod_bias_enabled = model_args.mod_bias_enabled
        self.config.mod_special_init = model_args.mod_special_init

        self.build_mod_layers(self.config)

    def build_mod_layers(self, config):

        mod_target_layers = config.mod_target_layers

        if (match := re.match(r"(\w+)_decay_(\d+\.\d+)_(\d+\.\d+)", mod_target_layers)): #"{shiftedcos}_decay_{0.75}_{0.25}"
            decay_type = match.group(1)
            max_ratio = float(match.group(2))
            min_ratio = float(match.group(3))

            decay_func = decay_func_dict[decay_type]

            mod_target_layer_indices = [
                i for i in range(config.num_hidden_layers)
                if decay_func(config.mod_router_factor, i) <= max_ratio
            ]
            config.mod_target_layer_indices = mod_target_layer_indices
            print(f"--------------mod_target_layer_indices:{mod_target_layer_indices}-----------------")

            self.layers = nn.ModuleList([
                ModLayer(
                    layer, config, layer_idx = idx,
                    ratio = max(decay_func(config.mod_router_factor, idx), min_ratio)
                )
                if (idx in mod_target_layer_indices)
                else layer
                for idx,layer in enumerate(self.layers)
            ])

            return

        if mod_target_layers == "deep_all":
            # exclude the last layer
            mod_target_layer_indices = list(range(2, config.num_hidden_layers-1))
        elif mod_target_layers == "deep_all_include_last":
            # include the last layer
            mod_target_layer_indices = list(range(2, config.num_hidden_layers))
        elif mod_target_layers == "interleave":
            mod_target_layer_indices = list(range(1, config.num_hidden_layers, 2))
        else:
            raise NotImplementedError(f"Unsupported mod_target_layers: {mod_target_layers}")

        config.mod_target_layer_indices = mod_target_layer_indices
        print(f"--------------mod_target_layer_indices:{mod_target_layer_indices}-----------------")

        self.layers = nn.ModuleList([
            ModLayer(layer, config, idx)
            if (idx in mod_target_layer_indices)
            else layer
            for idx,layer in enumerate(self.layers)
        ])

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mod_target_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Here we use 2d attention_mask(train with flash_attn), or set attention_mask to 'None' (inference with batchsize=1, for all attention implementation types)
        attention_mask_2d = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        if self._use_flash_attention_2:
            attention_mask = attention_mask_2d
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                if isinstance(decoder_layer, ModLayer):
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        attention_mask_2d, #2d mask
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        mod_target_mask
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                    )
            else:
                if isinstance(decoder_layer, ModLayer):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        attention_mask_2d=attention_mask_2d, #2d mask
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        mod_target_mask=mod_target_mask
                    )
                else:
                    layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PmodLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = PmodLlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PmodLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                mod_target_mask
            ) = self.prepare_inputs_labels_for_multimodal_mod(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )

            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                mod_target_mask=mod_target_mask,
                **kwargs
            )
        else: # no images
            return super().generate(
                inputs=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **kwargs
            )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        mod_target_mask = kwargs.pop("mod_target_mask", None) if past_key_values is None else None

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # pass these values as inputs for forward function, for the first generation step
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        if mod_target_mask is not None:
            inputs['mod_target_mask'] = mod_target_mask

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        mod_target_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                mod_target_mask
            ) = self.prepare_inputs_labels_for_multimodal_mod(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mod_target_mask=mod_target_mask,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, 1, bias=config.mod_bias_enabled)
        if config.mod_special_init:
            nn.init.zeros_(self.router.weight)
            if config.mod_bias_enabled:
                nn.init.zeros_(self.router.bias)

    def forward(self, x):
        return self.router(x).squeeze(-1)  # [batch_size, seq_len]

class ModLayer(nn.Module):
    def __init__(self, layer, config, layer_idx = None, ratio = None):
        super().__init__()

        self.decoder_layer = layer
        self.layer_idx = layer_idx

        self.router = TokenRouter(config)

        if ratio is not None: # with progressive ratio decay, ratio is passed in
            self.mod_router_factor = ratio
        else: # No decay, ratio is specified by mod_router_factor
            self.mod_router_factor = config.mod_router_factor

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        mod_weight_norm = config.mod_weight_norm
        match = re.match(r"(\w+)_(\d+\.\d+|\d+)_(-?\d+\.\d+|-?\d+)", mod_weight_norm) #{tanh}_0.2_1
        if match:
            norm_type, scale, bias = match.groups()
            scale = float(scale)
            bias = float(bias)
        else:
            norm_type = mod_weight_norm
            scale = 1.0
            bias = 1.0
        self.norm_type = norm_type
        self.norm_scale = scale
        self.norm_bias = bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        mod_target_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        batch_size, seq_length = hidden_states.shape[:2]

        if (mod_target_mask is None) or torch.all(~mod_target_mask): # mod_target_mask is None or all False, no mod operation is needed
            if attention_mask is not None and len(attention_mask.shape) == 4 and past_key_value is not None:
                # When inference with vainilla LlamaAttention, decoding(query_length=1), we create a new 4d mask for each MoD layer with corresponding past_key_value length
                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask_2d, (batch_size, seq_length), hidden_states, past_key_value.get_seq_length(self.layer_idx)
                    ) #Here attention_mask_2d should be None

            return self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        attention_mask = attention_mask_2d # 2d mask is used for MoD

        mod_target_lengths = mod_target_mask.sum(dim=1).int() #[batchsize,] , the number of target tokens in each example in the batch

        router_logits = self.router(hidden_states)# batch_size, seq_length

        # mask tokens that are not target for MoD, so that they won't be selected by topk
        router_logits[~mod_target_mask] = float('-inf')

        router_logits = self.weight_normalization(router_logits)

        top_k_lengths = torch.trunc(mod_target_lengths * self.mod_router_factor).int() # [batch_size,]
        max_top_k_length = top_k_lengths.max().item()
        _ , router_indices = torch.topk(router_logits, max_top_k_length, dim=1, sorted=True) # [batchsize,  max_top_k_length]

        top_k_mask = torch.arange(max_top_k_length, device=hidden_states.device).unsqueeze(0) < top_k_lengths.unsqueeze(1) # [batch_size, max_top_k_length], mask for top_k values for each row

        # mask for kept MoD target tokens
        kept_mod_target_tokens_mask = torch.full_like(mod_target_mask, False, dtype=torch.bool)  # [batch_size, seq_length]
        kept_mod_target_tokens_mask.scatter_(
            dim = 1,
            index = router_indices,
            src = top_k_mask # mask values since we take different top_k values for each row
        )

        #logger.info(f"-------layer_index:{self.decoder_layer.self_attn.layer_idx}, total tokens:{mod_target_lengths}, kept tokens:{kept_mod_target_tokens_mask.sum(dim=1).int()}, kept_ratio: {kept_mod_target_tokens_mask.sum(dim=1).int()/mod_target_lengths}-----------------")

        # mask for all kept tokens, including target tokens for MoD and non-target tokens
        kept_tokens_mask = kept_mod_target_tokens_mask.clone() # [batch_size, seq_length]

        # all tokens that are not target for MoD and not masked(padding tokens) are kept
        if attention_mask is not None: # 2d mask is passed through the layers.
            kept_tokens_mask[(~mod_target_mask) & attention_mask] = True #equals to  kept_toknes_mask = kept_tokens_mask | ((~mod_target_mask) & attention_mask)
        else: #Inference
            kept_tokens_mask[~mod_target_mask] = True

        hidden_states = hidden_states.clone() #clone the hidden_states to avoid in-place operation

        # prepare the kept_tokens(hidden_states) and position_ids to be passed to the transformer decoder layer
        kept_tokens_lengths = kept_tokens_mask.sum(dim=1).int() #[batchsize,] , the number of kept tokens in each example in the batch
        kept_tokens = torch.split(
            hidden_states[kept_tokens_mask],
            kept_tokens_lengths.tolist()
        )
        kept_position_ids = torch.split(
            position_ids.repeat(batch_size,1)[kept_tokens_mask],
            kept_tokens_lengths.tolist()
        )
        kept_tokens = nn.utils.rnn.pad_sequence(kept_tokens, batch_first=True) # [batch_size, max_kept_length, hidden_dim]
        kept_position_ids = nn.utils.rnn.pad_sequence(kept_position_ids, batch_first=True) # [batch_size, max_kept_length]

        #prepare attention_mask
        kept_attention_mask = torch.arange(kept_tokens_lengths.max(), device=hidden_states.device).unsqueeze(0) < kept_tokens_lengths.unsqueeze(1) #[batch_size, kept_tokens_lengths.max()]

        kept_attention_mask_2d = kept_attention_mask if (kept_attention_mask is not None and 0 in kept_attention_mask) else None

        kept_tokens_seq_length = kept_tokens.shape[1]
        if self._use_flash_attention_2:
            kept_attention_mask = kept_attention_mask_2d
        elif self._use_sdpa and not output_attentions:
            kept_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                kept_attention_mask,
                (batch_size, kept_tokens_seq_length),
                kept_tokens,
                past_key_values_length=0,
            )
        else:
            # 4d mask is passed through the layers
            kept_attention_mask = _prepare_4d_causal_attention_mask(
                kept_attention_mask, (batch_size, kept_tokens_seq_length), kept_tokens, past_key_values_length=0
            )

        # For a TINY inference speedup:
        # kept_tokens = hidden_states[kept_tokens_mask].unsqueeze(0)
        # kept_attention_mask_2d = kept_attention_mask = None
        # kept_position_ids = position_ids[kept_tokens_mask].unsqueeze(0)

        # call the forward of transformer decoder layer
        outputs, residual = self.decoder_layer.forward_return_residual(
            kept_tokens,
            attention_mask=kept_attention_mask,
            position_ids=kept_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        check_inf_or_nan(outputs[0],"outputs[0]")

        if kept_attention_mask_2d is not None:
            hidden_states[kept_tokens_mask] = outputs[0][kept_attention_mask_2d]
        else:
            # inference with batch_size=1
            # outputs[0].shape is (batchsize=1, seq_len, hidden_dim)
            hidden_states[kept_tokens_mask] = outputs[0].squeeze(0)

        hidden_states[mod_target_mask] *= router_logits[mod_target_mask].unsqueeze(-1)

        # add the residual to the kept tokens
        if kept_attention_mask_2d is not None:
            hidden_states[kept_tokens_mask] += residual[kept_attention_mask_2d]
        else:
            hidden_states[kept_tokens_mask] += residual.squeeze(0)

        check_inf_or_nan(hidden_states,"hidden_states")

        outputs = (hidden_states,) + outputs[1:]
        return outputs

    def weight_normalization(self, router_weights):

        norm_type, scale, bias = self.norm_type, self.norm_scale, self.norm_bias
        #should only be called when hasattr(self, "router") is True
        if norm_type == "softmax":
            router_weights = F.softmax(router_weights, dim=1)
        elif norm_type == "sigmoid":
            router_weights = F.sigmoid(router_weights)
        elif norm_type == "tanh":
            router_weights = F.tanh(router_weights)
        elif norm_type != "none":
            raise NotImplementedError(f"Unsupported norm_type for router: {norm_type}")

        return router_weights * scale + bias

def check_inf_or_nan(x, x_name):
    if torch.isnan(x).any():
        raise ValueError(
            f"NaN detected in input tokens, this is not intended to happen, please check your model. Before retraining, you could try the model with flash-attn-2 enabled.\n{x_name}:{x}")
    elif torch.isinf(x).any():
        raise ValueError(
            f"Inf detected in input tokens, this is not intended to happen, please check your model. Before retraining, you could try the model with flash-attn-2 enabled.\n{x_name}:{x}")

def shifted_cos_with_ratio(average_ratio, layer_idx):
    return math.cos(layer_idx * math.pi / 31) / 2  + average_ratio

def shifted_linear_with_ratio(average_ratio, layer_idx):
    return (- layer_idx / 31 + 0.5) + average_ratio

decay_func_dict = {
    "shiftedcos": shifted_cos_with_ratio,
    "shiftedlinear": shifted_linear_with_ratio,
}

AutoConfig.register("pmod_llava_llama", PmodLlavaConfig)
AutoModelForCausalLM.register(PmodLlavaConfig, PmodLlavaLlamaForCausalLM)
