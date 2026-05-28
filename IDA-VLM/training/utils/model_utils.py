# Model utilities for training

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import fetch_image, process_vision_info
# from PLIP.test import img_to_expert_feature
from torchvision import transforms
from torch.nn import Linear
import torch.nn as nn
import numpy as np
# from insightface.app import FaceAnalysis
# from insightface.utils import face_align
from experts.wyze_embedding import load_person_model


class DINOv2ExpertWrapper(nn.Module):
    """Wraps a DINOv2 model so its forward() returns (feat, None),
    matching the ``expert_feature, *_ = self.visual.expert(x)`` convention."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    def forward(self, x):
        return self.model(x), None


class WyzePersonExpertWrapper(nn.Module):
    """Wraps a Wyze embedding model to normalize its output to a tuple format,
    compatible with the ``expert_feature, *_ = self.visual.expert(x)`` convention
    used in replace.py for the ``person`` object type.

    Handles three Wyze model architectures:
      - ``embedding_model`` (10k / 50k): forward() already returns (embeddings, logits)
      - ``ft_net`` (v02_02_reid): forward() returns a plain tensor; we call extract_features()
      - ``clip_reid_encoder`` (v03_23_token): TorchScript, forward() returns a plain tensor
    """

    def __init__(self, wyze_extractor):
        super().__init__()
        self.model = wyze_extractor.model
        self.architecture = wyze_extractor.model_architecture

    def forward(self, x):
        if self.architecture == 'ft_net':
            feat = self.model.extract_features(x)
        elif self.architecture == 'clip_reid_encoder':
            feat = self.model(x)
        else:  # embedding_model
            feat, _ = self.model(x)
        return feat, None


class TwoLayerHead(nn.Module):
    # Head for SOP expert
    def __init__(self, feature_dim: int, mapped_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, mapped_dim)
        self.classifier = nn.Linear(mapped_dim, num_classes)

    def forward(self, features: torch.Tensor, return_mapped: bool = False):
        mapped = self.proj(features)
        logits = self.classifier(mapped)
        if return_mapped:
            return logits, mapped
        return logits


class ExpertCrossAttention(nn.Module):
    """Per-sample multi-head cross-attention fuser.

    Each image patch token (query) attends to *all* expert feature vectors in
    the same sample (keys/values), plus a learnable "null slot" representing
    "no match". A sample = 1 query image + N gallery images; the K/V set for
    each patch is therefore {null_slot, query_expert, gallery_1_expert, ...,
    gallery_N_expert} — (N+2) slots total. Attention is scoped per-sample
    (tokens never attend to expert vectors from a different sample).

    The null slot gives the softmax an "abstain" path: when no real expert
    matches (stranger queries), attention mass can flow to the null slot,
    whose value projection starts at zero — so the fuser output degenerates
    to the residual identity (image_embeds passes through unchanged).

    Output projection Wo is zero-initialized, so at step 0 the layer is a
    no-op regardless of the attention distribution. The null_slot parameter
    is also zero-initialized; training will shape it into whatever direction
    maximizes similarity to "stranger-type" image tokens.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.H = num_heads
        self.d = hidden_dim // num_heads
        self.scale = self.d ** -0.5

        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Zero-init output projection → at init, fuser output = 0,
        # so the residual add in the model forward returns image_embeds untouched.
        nn.init.zeros_(self.Wo.weight)

        # Learnable "no-match" slot, prepended to every sample's K/V set so
        # the softmax has an explicit abstain path for stranger queries.
        # Zero-init: neutral at step 0; training will move it into the
        # subspace where stranger-query tokens live.
        self.null_slot = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        image_embeds: torch.Tensor,        # [sumT, D] stacked across the batch
        expert_feature: torch.Tensor,      # [N_img, D] one vector per image
        token_sample_id: torch.Tensor,     # [sumT] int64, sample index per token
        expert_sample_id: torch.Tensor,    # [N_img] int64, sample index per image
    ) -> torch.Tensor:
        device = image_embeds.device
        D = image_embeds.shape[-1]
        B = int(expert_sample_id.max().item()) + 1

        # Build padded [B, M, D] K/V tensor + [B, M] valid-slot mask.
        # Slot 0 is reserved for the null slot in every sample; real expert
        # vectors go to slots 1..n_b, so M = max #experts per sample + 1.
        counts = torch.bincount(expert_sample_id, minlength=B)
        M = int(counts.max().item()) + 1

        kv = image_embeds.new_zeros(B, M, D)
        mask = torch.zeros(B, M, dtype=torch.bool, device=device)

        # Null slot: always present and always valid in every sample.
        # Cast to kv's dtype so bf16/fp16 training stays clean.
        kv[:, 0] = self.null_slot.to(kv.dtype)
        mask[:, 0] = True

        for b in range(B):
            idx = (expert_sample_id == b).nonzero(as_tuple=True)[0]
            n = idx.numel()
            kv[b, 1:1 + n] = expert_feature[idx]
            mask[b, 1:1 + n] = True

        q = self.Wq(self.ln_q(image_embeds))                # [sumT, D]
        k = self.Wk(self.ln_kv(kv))                         # [B, M, D]
        v = self.Wv(self.ln_kv(kv))                         # [B, M, D]

        q = q.view(-1, self.H, self.d)                      # [sumT, H, d]
        k = k.view(B, M, self.H, self.d)
        v = v.view(B, M, self.H, self.d)

        # Gather per-token K/V rows by sample id → [sumT, M, H, d]
        k_t = k[token_sample_id]
        v_t = v[token_sample_id]
        mask_t = mask[token_sample_id]                      # [sumT, M]

        scores = torch.einsum('thd,tmhd->thm', q, k_t) * self.scale  # [sumT, H, M]
        scores = scores.masked_fill(~mask_t.unsqueeze(1), float('-inf'))
        # Force fp32 for the softmax to avoid bf16 overflow in exp().
        # Cast back to scores' dtype so downstream ops stay in the autocast dtype.
        attn = scores.float().softmax(dim=-1).to(scores.dtype)
        ctx = torch.einsum('thm,tmhd->thd', attn, v_t).reshape(-1, self.H * self.d)  # [sumT, D]

        # Compute the residual delta in fp32 for safety, then cast back.
        delta = self.Wo(ctx.float()).to(image_embeds.dtype)
        return image_embeds + delta


class ExpertQFormer(nn.Module):
    """BLIP-2 / Perceiver-Resampler-inspired two-stage fusion module.

    Stage 1 (sample-global digest extraction):
        K learnable query tokens  ↔  per-sample (null_slot + expert descriptors)
        → K digest tokens per sample.
    Stage 2 (per-patch injection):
        image patch tokens  ↔  the K digests of their sample
        → residual add into image_embeds.

    Compared with ``ExpertCrossAttention``:
      - The K learnable queries are *trained parameters* (not derived from inputs),
        so they can specialise into different "aspects" of the expert set: one
        query may act as a stranger detector, another as a best-match extractor,
        etc. Under the training loss these specialisations emerge by gradient.
      - Matching happens once per sample (stage 1) instead of once per patch,
        and patches only have to consume the K compact digests (stage 2). This
        decouples matching from per-patch content injection.

    Output projection ``Wo`` is zero-initialised, so at step 0 the module is a
    no-op (image_embeds passes through unchanged) — same as ExpertCrossAttention.
    """

    def __init__(self, hidden_dim: int, num_queries: int = 8, num_heads: int = 8):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.K = num_queries
        self.H = num_heads
        self.d = hidden_dim // num_heads
        self.scale = self.d ** -0.5
        self.D = hidden_dim

        # K learnable query tokens — the "aspects" Q-Former asks about the expert set.
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

        # Learnable null slot for the expert K/V set (same semantics as ExpertCrossAttention):
        # gives the softmax an explicit abstain path so stranger samples don't force
        # mass onto real experts.
        self.null_slot = nn.Parameter(torch.zeros(hidden_dim))

        # Stage 1: queries ↔ (null + experts)
        self.ln_q1 = nn.LayerNorm(hidden_dim)
        self.ln_kv1 = nn.LayerNorm(hidden_dim)
        self.Wq1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Stage 2: image patches ↔ K digest tokens
        self.ln_q2 = nn.LayerNorm(hidden_dim)
        self.ln_kv2 = nn.LayerNorm(hidden_dim)
        self.Wq2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Zero-init output → at step 0, fuser adds zero to image_embeds.
        nn.init.zeros_(self.Wo.weight)

    def forward(
        self,
        image_embeds: torch.Tensor,        # [sumT, D]
        expert_feature: torch.Tensor,      # [N_img, D]
        token_sample_id: torch.Tensor,     # [sumT] int64
        expert_sample_id: torch.Tensor,    # [N_img] int64
    ) -> torch.Tensor:
        device = image_embeds.device
        D = self.D
        B = int(expert_sample_id.max().item()) + 1

        # --- Build per-sample K/V for stage 1: slot 0 = null, slots 1..n = experts.
        counts = torch.bincount(expert_sample_id, minlength=B)
        M = int(counts.max().item()) + 1   # +1 for null

        kv = image_embeds.new_zeros(B, M, D)
        mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        kv[:, 0] = self.null_slot.to(kv.dtype)
        mask[:, 0] = True
        for b in range(B):
            idx = (expert_sample_id == b).nonzero(as_tuple=True)[0]
            n = idx.numel()
            kv[b, 1:1 + n] = expert_feature[idx]
            mask[b, 1:1 + n] = True

        # === Stage 1: K learnable queries attend to the expert K/V set, per sample ===
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)   # [B, K, D]
        q1 = self.Wq1(self.ln_q1(queries))                       # [B, K, D]
        k1 = self.Wk1(self.ln_kv1(kv))                           # [B, M, D]
        v1 = self.Wv1(self.ln_kv1(kv))                           # [B, M, D]

        q1 = q1.view(B, self.K, self.H, self.d)
        k1 = k1.view(B, M, self.H, self.d)
        v1 = v1.view(B, M, self.H, self.d)

        # scores1: [B, K, H, M] = <q1, k1>
        scores1 = torch.einsum('bkhd,bmhd->bkhm', q1, k1) * self.scale
        scores1 = scores1.masked_fill(~mask.view(B, 1, 1, M), float('-inf'))
        # fp32 softmax to avoid bf16 overflow in exp().
        attn1 = scores1.float().softmax(dim=-1).to(scores1.dtype)
        # digest: [B, K, D] = sum_m attn1 * v1
        digest = torch.einsum('bkhm,bmhd->bkhd', attn1, v1).reshape(B, self.K, D)
        # Residual with the raw learnable queries so stage-1 can start as identity.
        digest = digest + queries

        # === Stage 2: image patches attend to per-sample digest ===
        q2 = self.Wq2(self.ln_q2(image_embeds))                  # [sumT, D]
        k2 = self.Wk2(self.ln_kv2(digest))                       # [B, K, D]
        v2 = self.Wv2(self.ln_kv2(digest))                       # [B, K, D]

        q2 = q2.view(-1, self.H, self.d)                         # [sumT, H, d]
        k2 = k2.view(B, self.K, self.H, self.d)
        v2 = v2.view(B, self.K, self.H, self.d)

        # Gather per-token K/V rows by sample id → [sumT, K, H, d]
        k2_t = k2[token_sample_id]
        v2_t = v2[token_sample_id]

        # scores2: [sumT, H, K]
        scores2 = torch.einsum('thd,tkhd->thk', q2, k2_t) * self.scale
        # fp32 softmax (same rationale as stage 1).
        attn2 = scores2.float().softmax(dim=-1).to(scores2.dtype)
        ctx = torch.einsum('thk,tkhd->thd', attn2, v2_t).reshape(-1, self.H * self.d)  # [sumT, D]

        # Compute the residual delta in fp32 for safety, then cast back.
        delta = self.Wo(ctx.float()).to(image_embeds.dtype)
        return image_embeds + delta


class PostFusionAdapter(nn.Module):
    """Zero-init residual bottleneck MLP applied to fused image_embeds.

    Output = x + fc2(GELU(fc1(LayerNorm(x)))), with fc2 weight/bias init
    to zero. At step 0 the layer is the identity, so inserting it does
    not perturb the baseline — gradients decide whether to grow it into
    a useful refinement of the fused vision tokens before they reach the
    LLM.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim // 4
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.act(self.fc1(self.ln(x.float())))).to(x.dtype)
        return x + delta


def detect_face(app, img):
    img = np.array(img)

    face = app.get(img)  # take the first face
    if len(face) == 0:
        return torch.randn(3, 112, 112)
    face = face[0]
    aligned = face_align.norm_crop(img, face.kps, image_size=112)
    rgb = aligned.astype(np.float32)
    tensor = torch.from_numpy((rgb - 127.5) / 128).permute(2,0,1)# .unsqueeze(0)
    return tensor

def get_expert_inputs(model, all_vision_infos, feature_mode):
    """Get expert inputs for the model."""
    if feature_mode == 'expert' and model.visual.expert_transform is not None:
        expert_image_inputs = []
        for vision_infos in all_vision_infos:
            for vision_info in vision_infos:
                if "image" in vision_info or "image_url" in vision_info:
                    expert_image_inputs.append(fetch_image(vision_info))

        expert_image_inputs = [model.visual.expert_transform(expert_input) for expert_input in expert_image_inputs]
        expert_image_inputs = torch.stack(expert_image_inputs)
    else:
        # print("feature_mode != 'expert': no expert inputs will be used.")
        expert_image_inputs = None

    return expert_image_inputs

def load_linear_head(checkpoint_path='../train_dino/dinov2_sop_ft_head.pth', device: torch.device='cuda') -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    feature_dim = int(ckpt['feature_dim'])
    mapped_dim = int(ckpt.get('mapped_dim', feature_dim))
    num_classes = int(ckpt['num_classes'])
    head = TwoLayerHead(feature_dim=feature_dim, mapped_dim=mapped_dim, num_classes=num_classes)
    head.load_state_dict(ckpt['state_dict'])
    head.to(device)
    head = head.proj
    return head

def load_finetuned_model(checkpoint_path='../train_dino/dinov2_sop_ft_triplet_finetune_new_model.pth', device: torch.device='cuda') -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load backbone
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    backbone.to(device)
    backbone.eval()
    # Load head
    feature_dim = int(ckpt['feature_dim'])
    mapped_dim = int(ckpt.get('mapped_dim', feature_dim))
    num_classes = int(ckpt['num_classes'])
    head = TwoLayerHead(feature_dim=feature_dim, mapped_dim=mapped_dim, num_classes=num_classes)
    head.load_state_dict(ckpt['head_state_dict'])
    head.to(device)
    head = head.proj
    head.eval()

    backbone.head = head
    return backbone



def load_model_and_processor(config):
    """Load the model and processor with appropriate configuration."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    print(f"Loaded model {config.model_id}.")

    # Load expert weights if available
    if config.feature_mode != 'expert':

        print("feature_mode != 'expert': no expert weights will be loaded.")
        model.visual.expert = None
        model.visual.expert_transform = None
        model.visual.expert_projector = None

    else:
        print("feature_mode == 'expert': expert weights will be loaded.")
        if config.object_type == 'person':

            expert_feature = config.expert_feature

            if expert_feature == "PLIP":
                print("+++++ Using PLIP expert feature.")
                # PLIP expert feature
                from experts.PLIP.visual_model import Image_encoder_ModifiedResNet
                model.visual.expert = Image_encoder_ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, input_resolution=[256,128], width=64)
                model.visual.expert.eval()
                model.visual.expert.to(torch.float32)
                model.visual.expert_transform = transforms.Compose([
                    transforms.Resize((256, 128), interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.357, 0.323, 0.328),
                                        (0.252, 0.242, 0.239))
                ])
                model.visual.expert.load_state_dict(
                    torch.load('./experts/PLIP/checkpoints/PLIP_RN50.pth.tar', map_location='cuda')['ImgEncoder_state_dict'],
                    strict=True
                )
                print("+++++ Loaded person expert weights PLIP.")
                hidden_size = model.config.hidden_size
                model.visual.expert_projector = nn.Sequential(
                    nn.Linear(768, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )

            elif expert_feature == "wyzev0323token":
                print("+++++ Using Wyze embedding expert feature.")

                # extractor = load_person_model("50k")
                extractor = load_person_model("v03_23_token")
                model.visual.expert = WyzePersonExpertWrapper(extractor)
                model.visual.expert.eval()  # disable dropout in ViT attention/MLP layers

                model.visual.expert_transform = transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
                    ])
                hidden_size = model.config.hidden_size
                model.visual.expert_projector = nn.Sequential(
                    nn.Linear(1280, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                print("+++++ Loaded person expert weights wyze_embedding_v03_23_token.")

            elif expert_feature == "wyzev0415token":
                print("+++++ Using Wyze embedding expert feature v04_15_token.")
                extractor = load_person_model("v04_15_token")
                model.visual.expert = WyzePersonExpertWrapper(extractor)
                model.visual.expert.eval()
                model.visual.expert_transform = transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ])
                hidden_size = model.config.hidden_size
                model.visual.expert_projector = nn.Sequential(
                    nn.Linear(3072, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                print("+++++ Loaded person expert weights wyze_embedding_v04_15_token (output dim=3072).")

            elif expert_feature == "wyzev0202reid":

                extractor = load_person_model("v02_02_reid")
                model.visual.expert = WyzePersonExpertWrapper(extractor)
                model.visual.expert.eval()

                model.visual.expert_transform = transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                ])
                hidden_size = model.config.hidden_size
                model.visual.expert_projector = nn.Sequential(
                    nn.Linear(512, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                print("+++++ Loaded person expert weights wyze_embedding_v02_02_reid.")

            elif expert_feature == "DINOv2":
                print("+++++ Using DINOv2 (ViT-L/14) expert feature.")
                model.visual.expert = DINOv2ExpertWrapper()
                model.visual.expert.eval()
                model.visual.expert_transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ])
                hidden_size = model.config.hidden_size
                model.visual.expert_projector = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                print("+++++ Loaded DINOv2 ViT-L/14 expert (output dim=1024).")

            elif expert_feature == 'None':
                print("+++++ No expert feature for person.")
                model.visual.expert = None
                model.visual.expert_transform = None
                model.visual.expert_projector = None


        elif config.object_type == 'face':
            # Initialize face detector + aligner
            model.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
            model.app.prepare(ctx_id=0, det_size=(256, 256))

            model.visual.expert = torch.load('./qwen/models/arcface_checkpoint.tar')
            model.visual.expert.eval()
            hidden_size = model.config.hidden_size
            model.visual.expert_projector = nn.Sequential(
                nn.Linear(512, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            model.visual.expert_transform = lambda x: detect_face(model.app, x)

        elif config.object_type == 'sop':
            # Will change this later
            model.visual.expert_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            hidden_size = model.config.hidden_size
            model.visual.expert_projector = nn.Sequential(
                nn.Linear(768, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

            model.visual.expert = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            model.visual.expert.eval()
            model.visual.expert = load_finetuned_model()

        elif config.object_type == 'pet':

            from experts.wyze_embedding import load_pet_model

            model.visual.expert = load_pet_model("50k").model
            model.visual.expert_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                ])
            hidden_size = model.config.hidden_size
            model.visual.expert_projector = nn.Sequential(
                nn.Linear(512, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            model.visual.expert = None
            model.visual.expert_transform = None
            model.visual.expert_projector = None

    if model.visual.expert:
        model.visual.expert.to('cuda')
        model.visual.expert_projector.to('cuda')

    # Per-sample learnable fuser. The attribute `model.visual.expert_fuser` is
    # reused across fusion variants — its class depends on `config.input_mode`.
    # Other modes leave it as None so the legacy pipeline (expert_and_image_attn /
    # add / only / image_only) is completely unaffected.
    hidden_size = model.config.hidden_size
    if getattr(config, 'input_mode', None) == 'expert_cross_attn':
        num_heads = getattr(config, 'cross_attn_heads', 8)
        model.visual.expert_fuser = ExpertCrossAttention(hidden_size, num_heads=num_heads)
        model.visual.expert_fuser.to('cuda')
        print(f"+++++ Added ExpertCrossAttention fuser "
              f"(hidden_size={hidden_size}, num_heads={num_heads}).")
    elif getattr(config, 'input_mode', None) == 'expert_qformer':
        num_queries = getattr(config, 'qformer_num_queries', 8)
        num_heads = getattr(config, 'qformer_num_heads', 8)
        model.visual.expert_fuser = ExpertQFormer(
            hidden_size, num_queries=num_queries, num_heads=num_heads,
        )
        model.visual.expert_fuser.to('cuda')
        print(f"+++++ Added ExpertQFormer fuser "
              f"(hidden_size={hidden_size}, num_queries={num_queries}, "
              f"num_heads={num_heads}).")
    else:
        model.visual.expert_fuser = None

    # Optional post-fusion adapter: a zero-init residual MLP applied to the
    # fused image_embeds (after `expert_and_image_attn` mixing) before they
    # are scattered into the LLM input. Identity at init.
    if getattr(config, 'input_mode', None) == 'expert_and_image_attn_adapter':
        bottleneck = getattr(config, 'post_fusion_adapter_bottleneck', None)
        model.visual.post_fusion_adapter = PostFusionAdapter(
            hidden_size, bottleneck_dim=bottleneck,
        )
        model.visual.post_fusion_adapter.to('cuda')
        print(f"+++++ Added PostFusionAdapter "
              f"(hidden_size={hidden_size}, "
              f"bottleneck_dim={model.visual.post_fusion_adapter.fc1.out_features}).")
    else:
        model.visual.post_fusion_adapter = None

    processor = Qwen2_5_VLProcessor.from_pretrained(
        config.model_id,
        max_pixels=256*256# 1280*28*28
    )


    return model, processor


def setup_trainable_parameters(model, training_parameters):
    """Configure which model parameters should be trainable."""
    for name, param in model.named_parameters():
        if "merger" in training_parameters and "model.visual.merger" in name:
            param.requires_grad = True
        elif "expert_projector" in training_parameters and "model.visual.expert_projector" in name:
            param.requires_grad = True
        elif "expert_fuser" in training_parameters and "model.visual.expert_fuser" in name:
            param.requires_grad = True
        elif "post_fusion_adapter" in training_parameters and "model.visual.post_fusion_adapter" in name:
            param.requires_grad = True
        elif "llm" in training_parameters and "model.language_model" in name:
            param.requires_grad = True
        elif "expert" in training_parameters and "model.visual.expert" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params}")

    return {
        k: v.data for k, v in model.named_parameters() if v.requires_grad
    }


def generate_text_from_sample(model, processor, sample, max_new_tokens=4096, device="cuda"):
    """Generate text from a sample using the model."""
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


def load_clip_model_and_processor():
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return clip_model, clip_processor
