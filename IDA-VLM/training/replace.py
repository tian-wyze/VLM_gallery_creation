import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast, Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VLVisionBlock, Qwen2_5_VLPatchMerger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.utils import can_return_tuple

import torchvision.transforms as transforms

def My_Qwen2_5_VLProcessor_call(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    videos: VideoInput = None,
    **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
) -> BatchFeature:
    output_kwargs = self._merge_kwargs(
        Qwen2_5_VLProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )
    image_inputs = videos_inputs = {}
    if images is not None:
        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        image_grid_thw = image_inputs["image_grid_thw"]

    if videos is not None:
        videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
        video_grid_thw = videos_inputs["video_grid_thw"]

        fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
        if isinstance(fps, (int, float)):
            second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
        elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
            second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
        else:
            raise ValueError(
                f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
            )
        videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

    if not isinstance(text, list):
        text = [text]

    text = text.copy()  # below lines change text in-place

    '''
    self.image_start_end = "<|vision_start|><|image_pad|><|vision_end|>"
    if 'input_mode' in kwargs.keys() and kwargs['input_mode'] == 'expert_only':
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                text[i] = text[i].replace(self.image_start_end, "<|vision_pad|>", 1)
                index += 1
    '''

    if images is not None:
        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_image_tokens = image_grid_thw[index].prod() // merge_length
                if 'input_mode' in kwargs.keys() and kwargs['input_mode'] == 'expert_and_image_concat':
                    text[i] = text[i].replace(self.image_start_end, self.image_start_end + "<|vision_pad|>", 1)
                    # # original implementation
                    # text[i] = text[i].replace(self.image_start_end, "<|vision_start|><|image_pad|><|vision_pad|><|vision_end|>", 1)
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                # text[i] = text[i].replace(self.image_token, "<|placeholder|>" * (num_image_tokens - 1) + 'sks', 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

    if videos is not None:
        merge_length = self.video_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.video_token in text[i]:
                num_video_tokens = video_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.video_token)

    return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
    text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
    self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)



def Qwen2_5_VisionTransformerPretrainedModel_init(self, config, *inputs, **kwargs) -> None:
    super(Qwen2_5_VisionTransformerPretrainedModel, self).__init__(config, *inputs, **kwargs)

    self.spatial_merge_size = config.spatial_merge_size
    self.patch_size = config.patch_size
    self.fullatt_block_indexes = config.fullatt_block_indexes
    self.window_size = config.window_size
    self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    self.patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=config.patch_size,
        temporal_patch_size=config.temporal_patch_size,
        in_channels=config.in_channels,
        embed_dim=config.hidden_size,
    )

    head_dim = config.hidden_size // config.num_heads
    self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    self.blocks = nn.ModuleList(
        [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
    )
    self.merger = Qwen2_5_VLPatchMerger(
        dim=config.out_hidden_size,
        context_dim=config.hidden_size,
        spatial_merge_size=config.spatial_merge_size,
    )
    self.gradient_checkpointing = False



'''
def integration_1(self, expert_feature, hidden_states, cu_seqlen_post_merger, window_index):

    if expert_feature is None:
        return hidden_states

    # Break up the original hidden states into 12 chunks
    chunks = [
        hidden_states[cu_seqlen_post_merger[i]:cu_seqlen_post_merger[i+1]] for i in range(len(cu_seqlen_post_merger) - 1)
    ]
    expert_feature = self.expert_projector(expert_feature)

    # Insert [32, hidden_dim] after each image chunk
    expanded_chunks = []
    for i in range(len(chunks)):
        expanded_chunks.append(chunks[i])           # Original image features
        normed_expert_feature = (expert_feature[i] - torch.mean(expert_feature[i], dim=[1])) / torch.std(expert_feature[i], dim=[1])
        scaled_expert_feature = normed_expert_feature * torch.std(chunks[i], dim=[0, 1]) + torch.mean(chunks[i], dim=[0, 1])
        expanded_chunks.append(scaled_expert_feature)

    # Concatenate them back into a single tensor
    new_hidden_states = torch.cat(expanded_chunks, dim=0)

    return new_hidden_states

def integration_2(self, expert_feature, hidden_states, cu_seqlen_post_merger, window_index):

    # Break up the original hidden states into 12 chunks
    chunks = [
        hidden_states[cu_seqlen_post_merger[i]:cu_seqlen_post_merger[i+1]] for i in range(len(cu_seqlen_post_merger) - 1)
    ]

    expert_feature = F.adaptive_avg_pool1d(expert_feature, 1280)
    expert_feature = expert_feature.reshape(-1, expert_feature.shape[-1])  # Assuming expert_feature is [batch_size, 1280, hidden_dim]
    expert_feature = self.merger(expert_feature)
    expert_feature = expert_feature.view(len(chunks), -1, expert_feature.shape[-1])

    expanded_chunks = []
    for i in range(len(chunks)):
        expanded_chunks.append(chunks[i])           # Original image features
        expanded_chunks.append(expert_feature[i])   # Expert features

    # Concatenate them back into a single tensor
    new_hidden_states = torch.cat(expanded_chunks, dim=0)

    return new_hidden_states
'''

def My_Qwen2_5_VisionTransformerPretrainedModel_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:

    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # cu_seqlen_post_merger = cu_seqlens // 4
    # hidden_states = integration_1(self, expert_feature, hidden_states, cu_seqlen_post_merger, window_index)

    return hidden_states


def My_Qwen2_5_VLModel_get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, expert_inputs: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:

    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    return image_embeds

def My_Qwen2_5_VLModel_get_expert_feature(self, expert_inputs):
    feature_mode = expert_inputs['feature_mode']
    gallery_size = expert_inputs['gallery_size']
    batch_size = expert_inputs['batch_size']
    object_type = expert_inputs['object_type']
    expert_inputs = expert_inputs['inputs']

    # if self.visual.expert.training == True:
    #     self.visual.expert.eval()

    if feature_mode == 'expert':
        self.visual.expert.to(torch.float32)
        if object_type == 'person':
            expert_feature, *_ = self.visual.expert(expert_inputs)
        elif object_type == 'sop':
            expert_feature = self.visual.expert(expert_inputs)
            expert_feature = self.visual.expert.head(expert_feature)
        elif object_type == 'pet':
            expert_feature, _ = self.visual.expert(expert_inputs)
        else:
            expert_feature = self.visual.expert(expert_inputs)
    elif feature_mode == 'random':
        expert_feature = torch.randn(1, 1, 768)
        expert_feature = expert_feature.repeat(batch_size * (gallery_size + 1), 1, 1).cuda()
    elif feature_mode == 'fully_random':
        expert_feature = torch.randn(batch_size * (gallery_size + 1), 1, 768).cuda()
    else:
        return None

    expert_feature = self.visual.expert_projector(expert_feature).squeeze(1)
    return expert_feature

def get_expert_attention_mask(
    image_grid_thw,
    expert_feature: torch.Tensor,     # [N, D]
    image_embeds: torch.Tensor,       # [sum_T, D]
) -> torch.Tensor:
    """
    Compute a per-token attention mask between each expert feature and its corresponding image tokens.

    Args:
        image_grid_thw: list of (T, H, W) for each image.
        expert_feature: [N, D] tensor, one expert vector per image.
        image_embeds: [sum_T, D] tensor of all concatenated image tokens.
        div_factor: divide each T*H*W by this value to get the number of tokens per image (default=4).
        normalize: if True, L2-normalize before attention.
        temperature: optional temperature scaling. If None, uses sqrt(D).

    Returns:
        mask: [sum_T, 1] tensor, soft attention weights concatenated across images.
    """
    N, D = expert_feature.shape
    sum_T, D2 = image_embeds.shape
    assert D == D2, f"Dim mismatch: expert {D} vs image_embeds {D2}"

    # Compute per-image token counts
    token_counts = [int((T * H * W) // 4) for (T, H, W) in image_grid_thw]
    assert len(token_counts) == N, "Mismatch between grid length and expert_feature count"
    assert sum(token_counts) == sum_T, f"Sum of token counts {sum(token_counts)} != {sum_T}"

    expert_feature = torch.nn.functional.normalize(expert_feature, dim=-1)
    image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

    masks = []
    start = 0
    for i, Ti in enumerate(token_counts):
        end = start + Ti
        tokens_i = image_embeds[start:end]     # [Ti, D]
        query_i = expert_feature[i]            # [D]
        scores = tokens_i @ query_i
        attn = torch.softmax(scores, dim=0).unsqueeze(-1)  # [Ti, 1]
        masks.append(attn)
        start = end

    return torch.cat(masks, dim=0)  # [sum_T, 1]



# @auto_docstring
def My_Qwen2_5_VLModel_forward(
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
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    expert_inputs: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        if pixel_values is not None and n_image_tokens > 0:
            # Expert_feature is set to None if we provide it separately with images
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, expert_inputs=None)

            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            #---------July 17th temporal expert-image broadcast addition test ---------
            if expert_inputs is not None:
                expert_feature = self.get_expert_feature(expert_inputs)
                if expert_feature is not None:
                    broadcasted_expert_feature = []
                    for i, img_shape in enumerate(image_grid_thw):
                        broadcasted_expert_feature.append(expert_feature[i].repeat(img_shape.prod() // 4, 1))
                    broadcasted_expert_feature = torch.cat(broadcasted_expert_feature, dim=0)
                    # Try this?
                    # broadcasted_expert_feature = broadcasted_expert_feature / broadcasted_expert_feature.std() * image_embeds.std() + image_embeds.mean()
                    if expert_inputs['input_mode'] == 'expert_and_image_add':
                        # check distribution, has normalization or not!!

                        image_embeds = 0.5 * image_embeds + 0.5 * broadcasted_expert_feature
                    elif expert_inputs['input_mode'] == 'expert_only':
                        image_embeds = broadcasted_expert_feature
                    elif expert_inputs['input_mode'] == 'expert_and_image_attn':
                        expert_attention_mask = get_expert_attention_mask(image_grid_thw, expert_feature, image_embeds)
                        image_embeds = (1 - expert_attention_mask) * image_embeds + expert_attention_mask * broadcasted_expert_feature
                        # image_embeds = image_embeds + expert_attention_mask * broadcasted_expert_feature
                    elif expert_inputs['input_mode'] in ('expert_cross_attn', 'expert_qformer'):
                        # Per-sample learnable fuser. Both expert_cross_attn and
                        # expert_qformer consume the same inputs (image_embeds +
                        # per-sample expert descriptors) and produce the same output
                        # shape (residual-added image_embeds). The fuser class
                        # installed on model.visual.expert_fuser decides the math
                        # (see utils/model_utils.py: ExpertCrossAttention vs ExpertQFormer).
                        images_per_sample = expert_inputs['images_per_sample']
                        device = image_embeds.device
                        expert_sample_id = torch.tensor(
                            [s for s, n in enumerate(images_per_sample) for _ in range(n)],
                            device=device, dtype=torch.long,
                        )
                        token_counts = (image_grid_thw.prod(dim=1) // 4).to(
                            device=device, dtype=torch.long,
                        )
                        token_sample_id = torch.repeat_interleave(expert_sample_id, token_counts)
                        image_embeds = self.visual.expert_fuser(
                            image_embeds, expert_feature, token_sample_id, expert_sample_id,
                        )
                    # -----------------------------------------------------------------------

            image_embeds = image_embeds.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        '''
        # This is for the original expert-image concat mode
        n_expert_tokens = (input_ids == 151654).sum().item()
        if expert_feature is not None and n_expert_tokens > 0:
            expert_feature = self.visual.expert_projector(expert_feature).squeeze(1)
            expert_feature = expert_feature.to(inputs_embeds.device, inputs_embeds.dtype)

            expert_mask = input_ids == 151654  # Using the <vision_pad> option.
            expert_mask_unsqueezed = expert_mask.unsqueeze(-1)
            expert_mask_expanded = expert_mask_unsqueezed.expand_as(inputs_embeds)
            expert_mask = expert_mask_expanded.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(expert_mask, expert_feature)
        '''


        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()



@can_return_tuple
# @auto_docstring
def My_Qwen2_5_VLForConditionalGeneration_forward(
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
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    expert_inputs: Optional[torch.FloatTensor] = None,
    input_mode: Optional[str] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        expert_inputs=expert_inputs
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        )


transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor.__call__ = My_Qwen2_5_VLProcessor_call
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = My_Qwen2_5_VisionTransformerPretrainedModel_forward
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.get_image_features = My_Qwen2_5_VLModel_get_image_features
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = My_Qwen2_5_VLModel_forward
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.get_expert_feature = My_Qwen2_5_VLModel_get_expert_feature
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = My_Qwen2_5_VLForConditionalGeneration_forward
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.__init__ = Qwen2_5_VisionTransformerPretrainedModel_init