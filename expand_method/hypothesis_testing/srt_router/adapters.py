"""
Model-specific SRT adapters.

These adapters inject SRT Mahalanobis routing into SMoLoRA and HiDe-LLaVA
at inference time, replacing the original routing mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .pooled_mahalanobis import PooledMahalanobisRouter
from .metrics import baseline_cosine_routing, baseline_l2_routing

if TYPE_CHECKING:
    from PIL import Image
    from ...embedding_extractors.clip_extractor import CLIPVisionExtractor


@dataclass
class TaskSignature:
    """Stores sufficient statistics for a single task."""
    task_name: str
    mu: np.ndarray       # (d,) centroid
    Sigma: np.ndarray    # (d, d) covariance
    n: int               # number of samples
    ins_emb: Optional[np.ndarray] = None  # (384,) Sentence-BERT vector for SMoLoRA IF


class SMoLoRASRTAdapter:
    """
    SRT Mahalanobis adapter for SMoLoRA's IF (Instruction-Type) Router.

    SMoLoRA's IF router uses:
        if_router = lora_ins_gate(ins_emb[ins_type])  # learnable linear → top-1

    SRT replaces this with:
        d_t = (h - μ_t)^T Σ_pool^{-1} (h - μ_t)
        t* = argmin_t d_t

    where h = instruction embedding (from ins_emb).

    This adapter builds task signatures from instruction embeddings and
    provides task prediction via SRT Mahalanobis routing.

    Usage (Option A — routing accuracy):
        adapter = SMoLoRASRTAdapter.from_ins_emb_pickle(pickle_path)
        adapter.add_task("scienceqa", ins_emb[0])
        adapter.add_task("textvqa", ins_emb[1])
        ...
        task_pred = adapter.predict(ins_emb_query)

    Usage (Option B — end-to-end):
        # Monkey-patch SMoLoraLinear.forward to use SRT
        adapter = SMoLoRASRTAdapter.from_ins_emb_list(...)
        adapter.patch_smolora_model(model)
        # ... run inference ...
        adapter.unpatch_smolora_model()
    """

    def __init__(
        self,
        shrinkage: str = "ridge",
        sigma_scale: float = 1.0,
    ):
        self.router = PooledMahalanobisRouter(
            shrinkage=shrinkage,
            sigma_scale=sigma_scale,
        )
        self.task_order: List[str] = []

    @classmethod
    def from_ins_emb_pickle(
        cls,
        pickle_path: str,
        task_names: Optional[List[str]] = None,
        shrinkage: str = "ridge",
    ) -> "SMoLoRASRTAdapter":
        """
        Build adapter from SMoLoRA's ins_emb_single.pkl file.

        Args:
            pickle_path: Path to ins_emb_single.pkl.
                Shape: (N_tasks, 384) numpy array.
            task_names: Optional list of task names.
                If None, uses f"task_{i}".
            shrinkage: SRT shrinkage method.
        """
        import pickle

        adapter = cls(shrinkage=shrinkage)
        with open(pickle_path, "rb") as f:
            ins_emb_array = pickle.load(f)  # (N, 384)

        ins_emb_array = np.array(ins_emb_array)
        if ins_emb_array.ndim == 1:
            ins_emb_array = ins_emb_array.reshape(1, -1)

        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(ins_emb_array))]

        for task_name, emb in zip(task_names, ins_emb_array):
            adapter.add_task(task_name, emb)

        return adapter

    @classmethod
    def from_ins_emb_list(
        cls,
        ins_emb_list: List[np.ndarray],
        task_names: Optional[List[str]] = None,
        shrinkage: str = "ridge",
    ) -> "SMoLoRASRTAdapter":
        """
        Build adapter from a list of instruction embedding arrays.

        Args:
            ins_emb_list: List of (384,) arrays, one per task.
            task_names: Optional list of task names.
            shrinkage: SRT shrinkage method.
        """
        adapter = cls(shrinkage=shrinkage)
        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(ins_emb_list))]

        for task_name, emb in zip(task_names, ins_emb_list):
            adapter.add_task(task_name, emb.reshape(1, -1))

        return adapter

    def add_task(self, task_name: str, ins_emb: np.ndarray) -> None:
        """
        Add a task's instruction embedding to the router.

        Args:
            task_name: Task identifier.
            ins_emb: (384,) or (1, 384) instruction embedding.
        """
        if ins_emb.ndim == 1:
            ins_emb = ins_emb.reshape(1, -1)
        self.router.add_task(ins_emb, task_name=task_name)
        self.task_order.append(task_name)

    def predict(self, ins_emb: np.ndarray) -> int:
        """
        Predict task index for an instruction embedding.

        Args:
            ins_emb: (d,) or (1, d) instruction embedding.

        Returns:
            Task index (0 = first added task).
        """
        if ins_emb.ndim == 1:
            ins_emb = ins_emb.reshape(1, -1)
        return int(self.router.route(ins_emb)[0])

    def predict_with_confidence(
        self, ins_emb: np.ndarray
    ) -> Tuple[int, float]:
        """Predict task with confidence score."""
        if ins_emb.ndim == 1:
            ins_emb = ins_emb.reshape(1, -1)
        preds, confs = self.router.route_with_confidence(ins_emb)
        return int(preds[0]), float(confs[0])

    def get_router(self) -> PooledMahalanobisRouter:
        return self.router


class CLIPVisionSRTAdapter:
    """
    SRT Mahalanobis adapter using CLIP Vision embeddings.

    Used for:
        1. SMoLoRA's VU (Visual-User) Router: replaces mean-pooled hidden states
        2. HiDe-LLaVA routing: replaces cosine similarity with CLIP features

    Workflow:
        1. Build signatures: extract CLIP embeddings for each task's images
        2. Predict: extract CLIP embedding for query image → SRT route
    """

    def __init__(
        self,
        clip_extractor: "CLIPVisionExtractor",
        shrinkage: str = "ridge",
        sigma_scale: float = 1.0,
    ):
        from ...embedding_extractors.clip_extractor import CLIPVisionExtractor
        self.clip_extractor: CLIPVisionExtractor = clip_extractor
        self.router = PooledMahalanobisRouter(
            shrinkage=shrinkage,
            sigma_scale=sigma_scale,
        )
        self.task_order: List[str] = []
        self._task_image_paths: Dict[str, List[str]] = {}

    def build_signatures_from_paths(
        self,
        task_to_image_paths: Dict[str, List[str]],
        batch_size: int = 8,
    ) -> None:
        """
        Build SRT task signatures from image paths.

        Args:
            task_to_image_paths: Dict mapping task_name -> list of image paths.
            batch_size: Batch size for CLIP extraction.
        """
        for task_name, paths in task_to_image_paths.items():
            embs = self.clip_extractor.extract_from_paths(paths, batch_size=batch_size)
            self.router.add_task(embs, task_name=task_name)
            self.task_order.append(task_name)
            self._task_image_paths[task_name] = paths

    def build_signatures_from_embeddings(
        self,
        task_to_embeddings: Dict[str, np.ndarray],
    ) -> None:
        """
        Build SRT task signatures from pre-extracted embeddings.

        Args:
            task_to_embeddings: Dict mapping task_name -> (N, d) embeddings.
        """
        for task_name in sorted(task_to_embeddings.keys()):
            embs = task_to_embeddings[task_name]
            self.router.add_task(embs, task_name=task_name)
            self.task_order.append(task_name)

    def add_task(
        self,
        task_name: str,
        images: Optional[List["Image"]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a task's images or embeddings.

        Args:
            task_name: Task identifier.
            images: List of PIL Images. Extracts CLIP embeddings.
            embeddings: (N, d) pre-extracted embeddings. Use one of images or embeddings.
        """
        if embeddings is not None:
            self.router.add_task(embeddings, task_name=task_name)
        elif images is not None:
            embs = self.clip_extractor.extract(images)
            self.router.add_task(embs, task_name=task_name)
        else:
            raise ValueError("Must provide either images or embeddings")
        self.task_order.append(task_name)

    def predict(self, image: "Image") -> int:
        """Predict task for a single image."""
        emb = self.clip_extractor.extract_single(image).reshape(1, -1)
        return int(self.router.route(emb)[0])

    def predict_batch(self, images: List["Image"]) -> np.ndarray:
        """Predict tasks for a batch of images."""
        embs = self.clip_extractor.extract(images)
        return self.router.route(embs)

    def predict_from_embedding(self, emb: np.ndarray) -> int:
        """Predict task from a pre-extracted embedding."""
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return int(self.router.route(emb)[0])

    def predict_from_embedding_with_confidence(
        self, emb: np.ndarray
    ) -> Tuple[int, float]:
        """Predict task with confidence from embedding."""
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        preds, confs = self.router.route_with_confidence(emb)
        return int(preds[0]), float(confs[0])

    def get_router(self) -> PooledMahalanobisRouter:
        return self.router

    def get_task_index(self, task_name: str) -> int:
        """Get index for a task name."""
        return self.task_order.index(task_name)


class HiDeLLaVASRTAdapter:
    """
    SRT Mahalanobis adapter for HiDe-LLaVA's cosine similarity routing.

    HiDe-LLaVA's routing in prepare_inputs_labels_for_multimodal:
        sim_i = cos(CLIP_image, image_anchor[i])
        sim_t = cos(CLIP_text, text_anchor[i])
        sim = (sim_i + sim_t) / 2
        expert_weight = softmax(sim / 0.1)

    SRT replaces this with:
        h = concat(CLIP_image, CLIP_text)  # or separate
        d_t = Mahalanobis distance
        task_pred = argmin(d_t)
        expert_weight = one_hot(task_pred)  # hard routing

    The adapter extracts CLIP features and performs SRT routing.
    For end-to-end use (Option B), monkey-patches the model's
    prepare_inputs_labels_for_multimodal method.
    """

    def __init__(
        self,
        clip_extractor: Optional["CLIPVisionExtractor"] = None,
        text_clip_model: Optional[str] = None,
        shrinkage: str = "ridge",
        feature_fusion: str = "image_only",  # concat | avg | image_only | text_only
        sigma_scale: float = 1.0,
    ):
        """
        Args:
            clip_extractor: CLIP vision extractor.
            text_clip_model: HuggingFace CLIP text model name. If None,
                uses same model as vision extractor.
            shrinkage: SRT shrinkage method.
            feature_fusion: How to combine image and text features.
                - 'image_only': Use CLIP image features only.
                - 'text_only': Use CLIP text features only.
                - 'concat': Concatenate image + text features.
                - 'avg': Average image and text features.
        """
        if clip_extractor is None:
            from ...embedding_extractors.clip_extractor import CLIPVisionExtractor
            clip_extractor = CLIPVisionExtractor()

        self.clip_extractor = clip_extractor
        self.feature_fusion = feature_fusion
        self.router = PooledMahalanobisRouter(
            shrinkage=shrinkage,
            sigma_scale=sigma_scale,
        )
        self.task_order: List[str] = []
        self._original_prepare_inputs = None
        self._patched_model = None

    def build_signatures_from_paths(
        self,
        task_to_image_paths: Dict[str, List[str]],
        batch_size: int = 8,
    ) -> None:
        """Build signatures from image paths using CLIP image features."""
        for task_name, paths in task_to_image_paths.items():
            embs = self.clip_extractor.extract_from_paths(paths, batch_size=batch_size)
            self.router.add_task(embs, task_name=task_name)
            self.task_order.append(task_name)

    def add_task(
        self,
        task_name: str,
        images: Optional[List["Image"]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Add a task's images or embeddings."""
        if embeddings is not None:
            self.router.add_task(embeddings, task_name=task_name)
        elif images is not None:
            embs = self.clip_extractor.extract(images)
            self.router.add_task(embs, task_name=task_name)
        else:
            raise ValueError("Must provide either images or embeddings")
        self.task_order.append(task_name)

    def predict(self, image: "Image") -> int:
        """Predict task for a single image."""
        emb = self.clip_extractor.extract_single(image).reshape(1, -1)
        return int(self.router.route(emb)[0])

    def predict_batch(self, images: List["Image"]) -> np.ndarray:
        """Predict tasks for a batch of images."""
        embs = self.clip_extractor.extract(images)
        return self.router.route(embs)

    def predict_with_confidence(self, image: "Image") -> Tuple[int, float]:
        """Predict task with confidence for a single image."""
        emb = self.clip_extractor.extract_single(image).reshape(1, -1)
        preds, confs = self.router.route_with_confidence(emb)
        return int(preds[0]), float(confs[0])

    def patch_model(self, model) -> None:
        """
        Monkey-patch HiDe-LLaVA model's prepare_inputs_labels_for_multimodal.

        After patching, the model will use SRT Mahalanobis routing instead of
        cosine similarity for expert weight computation.

        WARNING: This modifies the model in-place. Use unpatch_model() to restore.
        """
        if self._patched_model is not None:
            raise RuntimeError("Model is already patched. Call unpatch_model() first.")

        import torch

        def srt_prepare_inputs(self, input_ids, images, *args, **kwargs):
            """SRT-routed version of prepare_inputs_labels_for_multimodal."""
            # Call original method to get base inputs
            result = self._original_prepare_inputs(input_ids, images, *args, **kwargs)

            if not self.training and hasattr(self, "srt_adapter"):
                adapter = self.srt_adapter
                # Extract CLIP image features from images
                if images is not None:
                    clip_emb = adapter.clip_extractor.extract_single(images).reshape(1, -1)
                    task_pred = adapter.router.route(clip_emb.reshape(1, -1))[0]
                    task_idx = int(task_pred)

                    # Set expert_weight as one-hot (hard routing)
                    n_experts = adapter.router.n_tasks
                    expert_weight = torch.zeros(n_experts)
                    expert_weight[task_idx] = 1.0

                    # Assign to last layer's projection modules
                    # (HiDe-LLaVA convention: only last layer uses expert_weight)
                    last_layer = self.get_model().layers[-1]
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                      'gate_proj', 'up_proj', 'down_proj']:
                        proj = getattr(last_layer.self_attn if 'proj' not in proj_name
                                       else getattr(last_layer.self_attn, proj_name, None)
                                       or getattr(last_layer.mlp, proj_name, None),
                                       None)
                        if proj is not None and hasattr(proj, 'expert_weight'):
                            proj.expert_weight = expert_weight.tolist()

            return result

        # Store original and patch
        for cls in type(model).__mro__:
            if hasattr(cls, 'prepare_inputs_labels_for_multimodal'):
                self._original_prepare_inputs = cls.prepare_inputs_labels_for_multimodal
                cls.prepare_inputs_labels_for_multimodal = srt_prepare_inputs
                break

        model.srt_adapter = self
        self._patched_model = model

    def unpatch_model(self) -> None:
        """Restore original prepare_inputs_labels_for_multimodal."""
        if self._patched_model is None:
            return

        if self._original_prepare_inputs is not None:
            for cls in type(self._patched_model).__mro__:
                if hasattr(cls, 'prepare_inputs_labels_for_multimodal'):
                    cls.prepare_inputs_labels_for_multimodal = self._original_prepare_inputs
                    break

        if hasattr(self._patched_model, 'srt_adapter'):
            delattr(self._patched_model, 'srt_adapter')

        self._patched_model = None
        self._original_prepare_inputs = None

    def get_router(self) -> PooledMahalanobisRouter:
        return self.router

    def get_task_index(self, task_name: str) -> int:
        return self.task_order.index(task_name)
