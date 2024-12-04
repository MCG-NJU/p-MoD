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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

import math

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from transformers.utils import logging

logger = logging.get_logger("transformers")


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):

            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None: #create mm_projector parameters
            self.mm_projector = build_vision_projector(self.config, vision_cfg=self.vision_tower.config)

            if 'unpad' in mm_patch_merge_type: # if unpad, initialize image_newline token
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None: # load pretrained mlp adapter for LLaVA-1.5/LLaVA-NeXT Finetuning
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0: #initialize image start/end tokens' embeddings as the average of all embeddings
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter: #if only tune the mlp adapter(pretraining), freeze the output embeddings and tune the input embeddings since the image start/end tokens needs to be tuned.
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter: # the input embeddings of the image start/end tokens are trained and saved along with the mlp adapter weights, so we need to load them
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def prepare_inputs_labels_for_multimodal_mod(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        """Prepare multimodal inputs and labels for the model, encode images, embed text tokens, and insert image features into the text embeddings. This function is for LLaVA-NeXT model with AnyRes"""

        # 0. if no image-related processing is needed
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1: #No Image or a single input_ids is passed during inference after the first generation step
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # 1. load image-related configurations
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        if image_aspect_ratio != 'anyres': # mod for LLaVA-1.5 without anyres
            return self.prepare_inputs_labels_for_multimodal_mod_without_anyres(
                input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes
            )

        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat') #merge type 'spatial_unpad'
        mod_target_token_type = getattr(self.config, 'mod_target_token_type', 'vision_hr')
        max_num_patches = getattr(self.config, 'max_num_patches', None) # max number of sub-images to perform bilinear interpolation

        if mm_patch_merge_type == 'flat':
            #image_features = [x.flatten(0, 1) for x in image_features] #list of Tensor [num_sub_images * num_tokens_for_every_sub_image,dim] for every image
            raise NotImplementedError
        elif not mm_patch_merge_type.startswith('spatial'):
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        if mod_target_token_type is None:
            raise ValueError("prepare_inputs_labels_for_multimodal_mod should not be called when mod_target_token_type is None.")

        # 2. process and encode images

        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images] #x.ndim == 3 when x is an all-zero image
        elif images.ndim != 5:
            #under 'anyres' setting, this can only happen when all images in a batch are all-zero images of shape [3,336,336]
            images = images.unsqueeze(1) # batchsize, 1, 3, 336, 336

        concat_images = torch.cat([image for image in images], dim=0) # [total_num_sub_images_of_all_images,3,336,336]
        image_features = self.encode_images(concat_images) #[total_num_sub_images_of_all_images,num_tokens,dim]
        split_sizes = [image.shape[0] for image in images] #list of num_sub_images for every image
        image_features = torch.split(image_features, split_sizes, dim=0) #split into tuples of [num_sub_images,num_tokens,dim] for every image

        new_image_features = []
        high_resolution_image_feature_maps = []
        for image_idx, image_feature in enumerate(image_features): # for every image
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0] #base image, [num_tokens,dim]
                image_feature = image_feature[1:] # sub images, [num_sub_images-1,num_tokens,dim]
                height = width = self.get_vision_tower().num_patches_per_side
                #assert height * width == base_image_feature.shape[0] # num_tokens per sub-image

                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size) #num_sub_image_width, num_sub_image_height
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)

                if 'unpad' in mm_patch_merge_type: # 'spatial_unpad'
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3) # merge back to the original feature map, unpad the image, [dim, h, w]
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])

                    if max_num_patches is not None: #biliear interpolation
                        _, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * height**2))
                        if times > 1.01:
                            image_feature = image_feature[None] #1, dim, h, w
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0] # dim, h, w

                    image_feature = torch.cat((
                        image_feature,
                        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                    ), dim=-1) #add learnable image_new_line token
                    image_feature = image_feature.permute(1, 2, 0).flatten(0, 1) # H*W, C

                else: #'spatial' without unpad, flatten the re-organized image # TODO should we also add image_newline token here?
                    raise NotImplementedError
                    image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1).flatten(2, 3)# merge back to the original feature map, [h, w, dim]

                hr_image_feature = image_feature
                image_feature = base_image_feature

            else: #an all-zero image is passed as there is no image for this example.
                image_feature = image_feature[0]
                if 'unpad' in mm_patch_merge_type:
                    image_feature = torch.cat((
                        image_feature,
                        self.model.image_newline[None].to(image_feature.device)
                    ), dim=0)

                hr_image_feature = None

            new_image_features.append(image_feature)
            high_resolution_image_feature_maps.append(hr_image_feature)

        image_features = new_image_features #list of num_images Tensors (shape=[num_tokens,dim]) for every image

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # we can use boolean values to indicate whether the inputs are None, and save the dtype of some inputs
        labels_is_None = labels is None
        position_ids_is_None = position_ids is None
        attention_mask_is_None = attention_mask is None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask_original_dtype = attention_mask.dtype
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids_original_dtype = torch.long
            position_ids_original_device = input_ids.device
        else:
            position_ids_original_dtype = position_ids.dtype
            position_ids_original_device = position_ids.device

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # turn the original batched input_ids&labels back into a list of input_ids&labels of different lenghts, for each example
        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # insert image embeddings into embeds, and prepare the labels and mod_target masks
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        mod_target_mask = []

        for batch_idx, cur_input_ids in enumerate(input_ids): #process each example in the batch

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()# number of times that IMAGE_TOKEN_INDEX appears in the cur_input_ids

            if num_images == 0: # no image token appear in input, no need to insert image features
                #assert high_resolution_image_feature_maps[cur_image_idx] is None
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                mod_target_mask.append(torch.full(
                    (cur_input_embeds.shape[0],),
                    (mod_target_token_type == "all"),
                    dtype=torch.bool, device=cur_input_embeds.device
                ))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [] #cur_input_ids without image tokens(will be split according to image tokens)
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # split the cur_input_ids & cur_input_labels into pieces according to the image tokens
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # embed the text tokens, and split them according to the image tokens
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_mod_target_mask = []

            for i in range(num_images + 1): #insert image features
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_mod_target_mask.append(torch.full(
                    (len(cur_input_embeds_no_im[i]),),
                    (mod_target_token_type == "all"),
                    dtype=torch.bool, device=cur_input_embeds.device
                ))

                if i < num_images:
                    # insert image feature
                    cur_image_features = image_features[cur_image_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_mod_target_mask.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            (mod_target_token_type != 'vision_hr'),
                            dtype=torch.bool,
                            device=cur_input_embeds.device
                        )
                    )

                    # insert high_resolution image feature
                    hr_image_feature = high_resolution_image_feature_maps[cur_image_idx]
                    cur_new_input_embeds.append(hr_image_feature)
                    cur_new_labels.append(torch.full((hr_image_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_mod_target_mask.append(
                        torch.full(
                            (hr_image_feature.shape[0],),
                            True,
                            dtype=torch.bool,
                            device=cur_input_embeds.device
                        )
                    )
                    # update cur_image_idx
                    cur_image_idx += 1

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_mod_target_mask = torch.cat(cur_mod_target_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            mod_target_mask.append(cur_mod_target_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # Here we truncate them first, then pad them to the same length(different from train.DataCollatorForSupervisedDataset), which trades time for memory
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            mod_target_mask = [x[:tokenizer_model_max_length] for x in mod_target_mask]

        # Combine them back into a batch
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids_original_dtype, device=position_ids_original_device)
        # create mod target mask
        mod_target_mask_padded = torch.full((batch_size, max_len), False, dtype=torch.bool, device=new_input_embeds[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            # sampled position_id
            cur_position_ids = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left": #left padding
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = cur_position_ids
                    mod_target_mask_padded[i, -cur_len:] = mod_target_mask[i]
            else: #right padding(for Llama)
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = cur_position_ids
                    mod_target_mask_padded[i, :cur_len] = mod_target_mask[i]


        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # if original value of labels, attention_mask or position_ids is None, then return None.
        if labels_is_None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if attention_mask_is_None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=attention_mask_original_dtype)

        if position_ids_is_None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, \
            mod_target_mask_padded

    def prepare_inputs_labels_for_multimodal_mod_without_anyres(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        """Prepare multimodal inputs and labels for the model, encode images, embed text tokens, and insert image features into the text embeddings. This function is for LLaVA-1.5 and below without anyres."""

        # 0. if no image-related processing is needed
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1: #No Image or a single input_ids is passed during inference after the first generation step
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # 1. load image-related configurations
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

        if image_aspect_ratio != 'square' and image_aspect_ratio != 'pad':
            raise NotImplementedError

        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat') #merge type 'spatial_unpad'
        mod_target_token_type = getattr(self.config, 'mod_target_token_type', 'vision_hr')

        if mm_patch_merge_type != 'flat':
            #image_features = [x.flatten(0, 1) for x in image_features] #list of Tensor [num_sub_images * num_tokens_for_every_sub_image,dim] for every image
            raise NotImplementedError

        if mod_target_token_type is None:
            raise ValueError("prepare_inputs_labels_for_multimodal_mod should not be called when mod_target_token_type is None.")

        # 2. process and encode images
        image_features = self.encode_images(images) #[batchsize, num_tokens, dim] e.g. [128,576,4096]

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # we can use boolean values to indicate whether the inputs are None, and save the dtype of some inputs, which saves some memory
        labels_is_None = labels is None
        position_ids_is_None = position_ids is None
        attention_mask_is_None = attention_mask is None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask_original_dtype = attention_mask.dtype
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids_original_dtype = torch.long
            position_ids_original_device = input_ids.device
        else:
            position_ids_original_dtype = position_ids.dtype
            position_ids_original_device = position_ids.device

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # turn the original batched input_ids&labels back into a list of input_ids&labels of different lenghts, for each example
        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # insert image embeddings into embeds, and prepare the labels and mod_target masks
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        mod_target_mask = []

        for batch_idx, cur_input_ids in enumerate(input_ids): #process each example in the batch

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()# number of times that IMAGE_TOKEN_INDEX appears in the cur_input_ids

            if num_images == 0: # no image token appear in input, no need to insert image features
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                mod_target_mask.append(torch.full(
                    (cur_input_embeds.shape[0],),
                    (mod_target_token_type == "all"),
                    dtype=torch.bool, device=cur_input_embeds.device
                ))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [] #cur_input_ids without image tokens(will be split according to image tokens)
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # split the cur_input_ids & cur_input_labels into pieces according to the image tokens
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # embed the text tokens, and split them according to the image tokens
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_mod_target_mask = []

            for i in range(num_images + 1): #insert image features
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_mod_target_mask.append(torch.full(
                    (len(cur_input_embeds_no_im[i]),),
                    (mod_target_token_type == "all"),
                    dtype=torch.bool, device=cur_input_embeds.device
                ))

                if i < num_images:
                    # insert image feature
                    cur_image_features = image_features[cur_image_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_mod_target_mask.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            True,
                            dtype=torch.bool,
                            device=cur_input_embeds.device
                        )
                    )

                    # update cur_image_idx
                    cur_image_idx += 1

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_mod_target_mask = torch.cat(cur_mod_target_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            mod_target_mask.append(cur_mod_target_mask)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # Here we truncate them first, then pad them to the same length(different from train.DataCollatorForSupervisedDataset), which trades time for memory
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            mod_target_mask = [x[:tokenizer_model_max_length] for x in mod_target_mask]

        # Combine them back into a batch
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids_original_dtype, device=position_ids_original_device)
        # create mod target mask
        mod_target_mask_padded = torch.full((batch_size, max_len), False, dtype=torch.bool, device=new_input_embeds[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            # sampled position_id
            cur_position_ids = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left": #left padding
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = cur_position_ids
                    mod_target_mask_padded[i, -cur_len:] = mod_target_mask[i]
            else: #right padding(for Llama)
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = cur_position_ids
                    mod_target_mask_padded[i, :cur_len] = mod_target_mask[i]


        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # if original value of labels, attention_mask or position_ids is None, then return None.
        if labels_is_None: #if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if attention_mask_is_None: #if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=attention_mask_original_dtype) #attention_mask.to(dtype=_attention_mask.dtype)

        if position_ids_is_None: #if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, \
            mod_target_mask_padded

