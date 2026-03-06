"""
Compatibility shims for transformers 4.x APIs removed in transformers 5.0.

This module provides fallback definitions for all symbols and attributes
that were removed from the transformers library in version 5.0, allowing
the trainer code (originally written for transformers ~4.28) to run
without modification on transformers 5.0+.

Usage:
    from compat_transformers import (
        ShardedDDPOption, is_sagemaker_mp_enabled, smp_forward_backward,
        deepspeed_init, deepspeed_load_checkpoint, HPSearchBackend,
        patch_trainer_compat,
    )
    
    # In your Trainer subclass __init__:
    class MyTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            patch_trainer_compat(self)
"""

from enum import Enum

# ============================================================================
# ShardedDDPOption (removed in transformers 5.0, replaced by FSDP)
# ============================================================================
try:
    from transformers.trainer import ShardedDDPOption
except ImportError:
    class ShardedDDPOption(str, Enum):
        SIMPLE = "simple"
        ZERO_DP_2 = "zero_dp_2"
        ZERO_DP_3 = "zero_dp_3"
        OFFLOAD = "offload"

# ============================================================================
# is_sagemaker_mp_enabled (removed in transformers 5.0)
# ============================================================================
try:
    from transformers.trainer import is_sagemaker_mp_enabled
except ImportError:
    def is_sagemaker_mp_enabled():
        return False

# ============================================================================
# smp_forward_backward / smp_forward_only / smp_nested_tensor_size
# (SageMaker Model Parallel — removed in transformers 5.0)
# ============================================================================
try:
    from transformers.trainer import smp_forward_backward
except ImportError:
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
        raise RuntimeError("SageMaker Model Parallel is not available")

try:
    from transformers.trainer import smp_forward_only
except ImportError:
    def smp_forward_only(model, inputs):
        raise RuntimeError("SageMaker Model Parallel is not available")

try:
    from transformers.trainer import smp_nested_tensor_size
except ImportError:
    def smp_nested_tensor_size(tensor):
        return tensor.size()

# ============================================================================
# deepspeed_init (moved/changed in transformers 5.0)
# ============================================================================
try:
    from transformers.integrations import deepspeed_init
except ImportError:
    try:
        from transformers.deepspeed import deepspeed_init
    except ImportError:
        def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=None, inference=False):
            raise RuntimeError("DeepSpeed is not available in this transformers version")

# ============================================================================
# deepspeed_load_checkpoint (moved/changed in transformers 5.0)
# ============================================================================
try:
    from transformers.integrations import deepspeed_load_checkpoint
except ImportError:
    try:
        from transformers.deepspeed import deepspeed_load_checkpoint
    except ImportError:
        def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path):
            raise RuntimeError("DeepSpeed is not available in this transformers version")

# ============================================================================
# HPSearchBackend (removed in transformers 5.0)
# ============================================================================
try:
    from transformers.trainer import HPSearchBackend
except ImportError:
    try:
        from transformers.trainer_utils import HPSearchBackend
    except ImportError:
        class HPSearchBackend(str, Enum):
            OPTUNA = "optuna"
            RAY = "ray"
            SIGOPT = "sigopt"
            WANDB = "wandb"

# ============================================================================
# is_apex_available (removed in transformers 5.0)
# ============================================================================
try:
    from transformers.utils import is_apex_available
except ImportError:
    def is_apex_available():
        return False

# ============================================================================
# ALL_LAYERNORM_LAYERS (removed from transformers.trainer in 5.0)
# ============================================================================
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    except ImportError:
        import torch.nn as _nn
        ALL_LAYERNORM_LAYERS = [_nn.LayerNorm]

# ============================================================================
# is_torch_tpu_available (removed in transformers 5.0, replaced by is_torch_xla_available)
# ============================================================================
try:
    from transformers.utils import is_torch_tpu_available
except ImportError:
    try:
        from transformers.utils import is_torch_xla_available as is_torch_tpu_available
    except ImportError:
        def is_torch_tpu_available(check_device=True):
            return False


def patch_trainer_compat(trainer):
    """
    Patch a Trainer instance with deprecated attributes that were removed
    in transformers 5.0. Call this in __init__ after super().__init__().
    
    Sets safe defaults for all deprecated attributes:
    - sharded_ddp = None  (FSDP replaces this)
    - use_apex = False
    - do_grad_scaling = False
    - scaler = None
    - deepspeed = None (the engine reference)
    - current_flos = 0
    - hp_name = None
    - hp_search_backend = None
    - is_in_train = False
    """
    if not hasattr(trainer, 'sharded_ddp'):
        trainer.sharded_ddp = None
    if not hasattr(trainer, 'fsdp'):
        trainer.fsdp = None
    if not hasattr(trainer, 'use_apex'):
        trainer.use_apex = False
    if not hasattr(trainer, 'do_grad_scaling'):
        trainer.do_grad_scaling = False
    if not hasattr(trainer, 'scaler'):
        trainer.scaler = None
    if not hasattr(trainer, 'deepspeed'):
        trainer.deepspeed = None
    if not hasattr(trainer, 'current_flos'):
        trainer.current_flos = 0
    if not hasattr(trainer, 'hp_name'):
        trainer.hp_name = None
    if not hasattr(trainer, 'hp_search_backend'):
        trainer.hp_search_backend = None
    if not hasattr(trainer, 'is_in_train'):
        trainer.is_in_train = False
    if not hasattr(trainer, 'is_deepspeed_enabled'):
        trainer.is_deepspeed_enabled = False
    if not hasattr(trainer, '_trial'):
        trainer._trial = None
    if not hasattr(trainer, 'model_wrapped'):
        trainer.model_wrapped = getattr(trainer, 'model', None)
