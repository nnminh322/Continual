"""
SiNet_SRT: SiNet with SRT (Statistical Routing Theory) routing.

Key differences from SiNet (sinet_inflora.py):
  - SRT Router (ridge shrinkage Mahalanobis) replaces task-ID routing
  - Trainable A AND B matrices (InfLoRA freezes A)
  - No GPM/DualGPM — SRT handles interference mitigation
  - Frozen backbone for SRT signature extraction

Pipeline:
  Training:  forward_train() uses current task LoRA (known from ground truth)
  Inference: forward_inference() uses SRT router → argmin Mahalanobis → selected LoRA
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

from models.vit_inflora import (
    VisionTransformer,
    PatchEmbed,
    resolve_pretrained_cfg,
    build_model_with_cfg,
    checkpoint_filter_fn,
    adapt_input_conv,
    resize_pos_embed,
)
from models.srt_router import SRT_Router


# ─────────────────────────────────────────────────────────────────────────────
# ViT without LoRA (frozen backbone for SRT signature extraction)
# ─────────────────────────────────────────────────────────────────────────────

class ViT_Frozen(nn.Module):
    """
    Frozen ViT using ViT_lora_co with task_id=-1 to skip LoRA entirely.
    Extracts CLS token embedding (index 0) WITHOUT fc_norm.

    User's algorithm: "h(x) = Frozen_ViT(x)[CLS_token]"
    T5:  h(x) = mean_pool(Frozen_T5(x))
    ViT: h(x) = CLS_token(Frozen_ViT(x))
    """

    def __init__(self, args):
        super().__init__()
        self.out_dim = args["embd_dim"]
        model_kwargs = dict(
            patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=12,
            n_tasks=args["total_sessions"], rank=args["rank"],
        )
        # Use ViT_lora_co: task_id=-1 skips LoRA → true frozen backbone
        self.vit = _create_vit_lora('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token from frozen backbone (no LoRA, no fc_norm).

        User's algorithm: "h(x) = Frozen_ViT(x)[CLS_token]"
        Matches NLP's mean_pool(Frozen_T5(x)) role — the SRT input h(x).

        We call forward_features() directly to avoid fc_norm (LayerNorm)
        which destroys the anisotropic hyper-sphere geometry that ZCA exploits.

        Returns: [B, D] CLS token embedding (no normalization applied)
        """
        with torch.no_grad():
            # forward_features returns (B, seq+1, D) after blocks + final norm
            # task=-1 skips LoRA in Attention_LoRA_SRT
            hidden = self.vit.forward_features(x, task=-1)  # (B, seq+1, D)
            # CLS token (index 0) — matches user's spec exactly
            # User's algorithm: "h(x) = Frozen_ViT(x)[CLS_token]"
            return hidden[:, 0]  # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# ViT with LoRA (trainable A and B matrices, per-task)
# ─────────────────────────────────────────────────────────────────────────────

class Attention_LoRA_SRT(nn.Module):
    """
    Attention with per-task LoRA A and B matrices.
    Differs from Attention_LoRA (vit_inflora.py):
      - Trainable A AND B (InfLoRA freezes A)
      - forward_inference(x, task_id) uses single-task LoRA
      - forward_train(x, task_id) uses single-task LoRA
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.rank = r

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Per-task LoRA: A (dim→rank), B (rank→dim)
        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self._delta_Ws: List[torch.Tensor] = []  # Store ΔW_k = B_k @ A_k per task

    def init_param(self):
        """Kaiming init for A, zero init for B (called for task 0)."""
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def _get_delta_W(self, task: int) -> torch.Tensor:
        """Return ΔW = B @ A for given task."""
        W_k = self.lora_B_k[task].weight @ self.lora_A_k[task].weight
        W_v = self.lora_B_v[task].weight @ self.lora_A_v[task].weight
        return W_k + W_v

    def store_delta_W(self, task: int):
        """Store ΔW after training task to use in SGWI for next task."""
        delta = self._get_delta_W(task)
        if len(self._delta_Ws) <= task:
            self._delta_Ws.append(delta.detach().cpu())
        else:
            self._delta_Ws[task] = delta.detach().cpu()

    def get_all_delta_Ws(self) -> List[torch.Tensor]:
        """Return all stored ΔW tensors."""
        return self._delta_Ws

    def forward_train(self, x: torch.Tensor, task: int):
        """
        Training forward: apply LoRA of the current task only.
        task is known from ground truth label.
        task=-1: frozen backbone (no LoRA).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Single-task LoRA (skip when task=-1 for frozen backbone)
        if task > -0.5:
            weight_k = self.lora_B_k[task].weight @ self.lora_A_k[task].weight
            weight_v = self.lora_B_v[task].weight @ self.lora_A_v[task].weight
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_inference(self, x: torch.Tensor, task: int):
        """
        Inference forward: same as training (single task LoRA).
        SRT router selects the task before calling this.
        """
        return self.forward_train(x, task)

    def forward(self, x: torch.Tensor, task: int):
        """Default forward: single-task LoRA (used for training)."""
        return self.forward_train(x, task)


class Block_SRT(nn.Module):
    """ViT Block with Attention_LoRA_SRT."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        n_tasks=10, r=64,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA_SRT(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop, r=r, n_tasks=n_tasks,
        )
        self.ls1 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.drop_path1 = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.ls2 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.drop_path2 = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)

    def forward(self, x, task):
        x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x), task) if self.ls1 is not None else self.attn(self.norm1(x), task))
        x = x + self.drop_path2(self.ls2 * self.mlp(self.norm2(x)) if self.ls2 is not None else self.mlp(self.norm2(x)))
        return x


class VisionTransformerLoRA(nn.Module):
    """VisionTransformer with trainable LoRA A/B per task."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
        embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=None, n_tasks=10, rank=64,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        block_fn = block_fn or Block_SRT

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Use ModuleList (not Sequential) so we can pass `task` to each block
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer, n_tasks=n_tasks, r=rank,
            )
            for i in range(depth)
        ])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = embed_dim

        if weight_init != 'skip':
            nn.init.normal_(self.cls_token, std=1e-6)
            nn.init.normal_(self.pos_embed, std=.02)

    def forward_features(self, x, task=-1):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.size(1), :])
        for blk in self.blocks:
            x = blk(x, task=task)
        x = self.norm(x)
        return x

    def forward(self, x, task):
        """Forward with LoRA (single task)."""
        x = self.forward_features(x, task=task)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return {
            'fmaps': [x],
            'features': x,
        }

    def forward_train(self, x, task):
        """Forward for training (same as forward)."""
        return self.forward(x, task)

    def forward_inference(self, x, task):
        """Forward for inference (same as forward)."""
        return self.forward(x, task)


@torch.no_grad()
def _load_weights_lora(model: VisionTransformerLoRA, checkpoint_path: str, prefix: str = ''):
    """Load official JAX ViT `.npz` weights into the shared LoRA backbone."""
    import numpy as np

    def _n2p(w, transpose=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if transpose:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    weights = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in weights:
        prefix = 'opt/target/'

    embed_conv_w = adapt_input_conv(
        model.patch_embed.proj.weight.shape[1],
        _n2p(weights[f'{prefix}embedding/kernel']),
    )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(weights[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(weights[f'{prefix}cls'], transpose=False))

    pos_embed_w = _n2p(weights[f'{prefix}Transformer/posembed_input/pos_embedding'], transpose=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_tokens', 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)

    if isinstance(model.norm, nn.LayerNorm):
        model.norm.weight.copy_(_n2p(weights[f'{prefix}Transformer/encoder_norm/scale']))
        model.norm.bias.copy_(_n2p(weights[f'{prefix}Transformer/encoder_norm/bias']))

    if (
        isinstance(model.head, nn.Linear)
        and f'{prefix}head/bias' in weights
        and model.head.bias.shape[0] == weights[f'{prefix}head/bias'].shape[-1]
    ):
        model.head.weight.copy_(_n2p(weights[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(weights[f'{prefix}head/bias']))

    for index, block in enumerate(model.blocks):
        block_prefix = f'{prefix}Transformer/encoderblock_{index}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'

        block.norm1.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(weights[f'{mha_prefix}{name}/kernel'], transpose=False).flatten(1).T
            for name in ('query', 'key', 'value')
        ]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(weights[f'{mha_prefix}{name}/bias'], transpose=False).reshape(-1)
            for name in ('query', 'key', 'value')
        ]))
        block.attn.proj.weight.copy_(_n2p(weights[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(weights[f'{mha_prefix}out/bias']))

        block.mlp[0].weight.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_0/kernel']))
        block.mlp[0].bias.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_0/bias']))
        block.mlp[2].weight.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_1/kernel']))
        block.mlp[2].bias.copy_(_n2p(weights[f'{block_prefix}MlpBlock_3/Dense_1/bias']))
        block.norm2.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/bias']))


def _create_vit_lora(variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    # Build model without auto-loading pretrained weights
    model = build_model_with_cfg(
        VisionTransformerLoRA, variant, pretrained=False,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    # Handle .npz pretrained weights (timm 1.x changed from custom_load='npz' to custom_load=True)
    if pretrained and pretrained_cfg.url and pretrained_cfg.url.endswith('.npz'):
        def custom_load_pretrained(self, pretrained_loc):
            _load_weights_lora(self, pretrained_loc)
            print('[sinet_srt_inflora] Loaded .npz pretrained weights into shared ViT backbone; LoRA weights remain task-initialized.')

        model.load_pretrained = custom_load_pretrained.__get__(model, type(model))
        from timm.models._builder import load_pretrained
        cfg_dict = pretrained_cfg.to_dict()
        cfg_dict['custom_load'] = True
        load_pretrained(model, pretrained_cfg=cfg_dict, num_classes=default_num_classes)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SiNet_SRT: Full model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SiNet_SRT(nn.Module):
    """
    SiNet with SRT routing.

    Components:
      - image_encoder_frozen: VisionTransformerFrozen (pretrained, no LoRA)
      - image_encoder: VisionTransformerLoRA (pretrained backbone, trainable LoRA A/B)
      - classifier_pool: per-task linear classifiers
      - srt_router: SRT_Router for task prediction
    """

    def __init__(self, args):
        super().__init__()

        # Frozen backbone for SRT signature extraction (same ViT, LoRA skipped via task_id=-1)
        self.image_encoder_frozen = ViT_Frozen(args)
        self.image_encoder_frozen.eval()

        # Trainable ViT with LoRA per task
        model_kwargs = dict(
            patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=12,
            n_tasks=args["total_sessions"], rank=args["rank"],
        )
        self.image_encoder = _create_vit_lora('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)

        # Initialize LoRA params (kaiming for A, zero for B)
        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA_SRT):
                module.init_param()

        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for _ in range(args["total_sessions"])
        ])
        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for _ in range(args["total_sessions"])
        ])

        # SRT Router V1: ZCA Whitening routing.
        # Matches "Whitening Sentence Representations" (Huang et al., ACL 2021).
        # W_zca = V @ Λ^{-1/2} @ V^T from pooled Σ → spherizes embeddings.
        self.srt_router = SRT_Router(
            embed_dim=args["embd_dim"],
        )

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_frozen_vector(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS embedding from FROZEN backbone (no LoRA).
        Used for SRT router signature computation.
        """
        with torch.no_grad():
            feats = self.image_encoder_frozen(image)  # [B, D]
        return feats

    def extract_vector(self, image: torch.Tensor, task: Optional[int] = None) -> torch.Tensor:
        """
        Extract CLS embedding with LoRA-adapted backbone.
        Used for classifier training / evaluation features.
        """
        if task is None:
            task = self.numtask - 1
        out = self.image_encoder(image, task=task)
        return out['features']

    def update_fc(self, nb_classes: int):
        """Increment task counter and expand classifier pool."""
        self.numtask += 1

    def forward_train(self, image: torch.Tensor) -> dict:
        """
        Training forward: use current task's LoRA (known from ground truth).
        Returns logits for current task only, plus features.
        """
        task = self.numtask - 1
        out = self.image_encoder(image, task=task)
        image_features = out['features']
        logits = self.classifier_pool[task](image_features)
        return {
            'logits': logits,
            'features': image_features,
        }

    def forward_inference(self, image: torch.Tensor) -> dict:
        """
        Inference forward: use SRT router to predict task, then apply that task's LoRA.
        Returns logits from the predicted task's classifier.
        """
        # Extract frozen CLS for SRT routing
        frozen_feats = self.extract_frozen_vector(image)  # [B, D]
        pred_tasks = self.srt_router.route(frozen_feats)  # [B]

        # Apply LoRA for each predicted task
        batch_size = image.shape[0]
        all_logits = []
        for b in range(batch_size):
            t = pred_tasks[b].item()
            if t >= self.numtask:
                t = self.numtask - 1  # fallback to most recent
            out = self.image_encoder(image[b:b+1], task=t)
            feat = out['features']
            logits = self.classifier_pool[min(t, self.numtask - 1)](feat)
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)

        return {
            'logits': logits,
            'features': None,
        }

    def forward(self, image: torch.Tensor) -> dict:
        """Default forward: training mode (current task LoRA)."""
        return self.forward_train(image)

    def interface(self, image: torch.Tensor) -> torch.Tensor:
        """
        Interface for evaluation: SRT routing inference.
        Returns [B, class_num] logits for predicted task's classes.
        The learner adds task offset for global class indexing.
        """
        with torch.no_grad():
            frozen_feats = self.extract_frozen_vector(image)
            pred_tasks = self.srt_router.route(frozen_feats)

            batch_size = image.shape[0]
            all_logits = []
            for b in range(batch_size):
                t = pred_tasks[b].item()
                t = min(t, self.numtask - 1)
                out = self.image_encoder(image[b:b+1], task=t)
                feat = out['features']
                all_logits.append(self.classifier_pool[t](feat))
            return torch.cat(all_logits, dim=0)

    def interface_gt(self, image: torch.Tensor) -> torch.Tensor:
        """
        Interface with ground-truth task labels.
        Used when task boundaries are known (e.g., during evaluation with task labels).
        """
        logits_list = []
        for t in range(self.numtask):
            out = self.image_encoder(image, task=t)
            logits_list.append(self.classifier_pool[t](out['features']))
        return torch.cat(logits_list, dim=1)

    def classifier_backup(self, task_id: int):
        self.classifier_pool_backup[task_id].load_state_dict(
            self.classifier_pool[task_id].state_dict()
        )

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return torch.nn.ModuleDict({k: v.clone() for k, v in self.state_dict().items()})

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
