"""Optional EfficientTAM-Ti and SAM 2.1 Hiera Tiny adapters for live selection.

The official video predictors are written for offline videos: ``init_state`` loads every
frame up front and ``propagate_in_video`` keeps every frame's memory. These adapters drive
the same predictor objects one live frame at a time instead, keeping

- the seed frame plus at most ``max_conditioning_frames - 1`` recent correction frames,
- the last ``memory_window_frames`` tracked frames (at least the model's mask-memory and
  object-pointer spans), and
- one frame's cached image features,

so GPU memory does not grow with stream length. The predictor's private per-frame
methods are used only in this module and only against the pinned revisions in
:data:`BACKENDS`; upgrading either repository requires re-running the replay benchmark.

Nothing here is imported by ``vision_pipeline.perception.objects``. Research code is put
on ``sys.path`` only when a segmenter is constructed.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_pipeline.image import PixelFormat
from vision_pipeline.rgbd import AlignedRgbdFrame

from .selection import Click, SegmentationResult, StalePromptError

_OBJECT_ID = 1
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


class BackendUnavailableError(RuntimeError):
    """A research backend's repository, checkpoint, dependency, or GPU is unavailable."""


@dataclass(frozen=True, slots=True)
class ResearchBackendSpec:
    """Pinned upstream identity for one promptable video segmentation model."""

    name: str
    package: str
    builder_module: str
    builder: str
    config: str
    repository_url: str
    revision: str
    checkpoint_url: str
    checkpoint_sha256: str
    license: str


BACKENDS: dict[str, ResearchBackendSpec] = {
    "efficienttam-ti": ResearchBackendSpec(
        name="efficienttam-ti",
        package="efficient_track_anything",
        builder_module="efficient_track_anything.build_efficienttam",
        builder="build_efficienttam_video_predictor",
        config="configs/efficienttam/efficienttam_ti.yaml",
        repository_url="https://github.com/yformer/EfficientTAM",
        revision="abcd061ebd3cc6e7527d152d75b890126aaa53f6",
        checkpoint_url=(
            "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/"
            "efficienttam_ti.pt"
        ),
        checkpoint_sha256="acbb17b28cca1f860acee09c9ecb6efdb732080dc7a85a07292c31813175fa7d",
        license="Apache-2.0",
    ),
    "sam2.1-hiera-tiny": ResearchBackendSpec(
        name="sam2.1-hiera-tiny",
        package="sam2",
        builder_module="sam2.build_sam",
        builder="build_sam2_video_predictor",
        config="configs/sam2.1/sam2.1_hiera_t.yaml",
        repository_url="https://github.com/facebookresearch/sam2",
        revision="2b90b9f5ceec907a1c18123530e92e794ad901a4",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        ),
        checkpoint_sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        license="Apache-2.0 (model code and checkpoints); BSD-3-Clause (cc_torch kernel)",
    ),
}

_SETUP_HINT = "run scripts/setup_selection_models.sh to fetch the pinned repositories/checkpoints"


@dataclass(frozen=True, slots=True)
class ResearchSegmenterConfig:
    backend: str
    repository: Path
    checkpoint: Path
    device: str = "cuda:0"
    memory_window_frames: int = 16
    max_conditioning_frames: int = 4
    recent_memory_frames: int | None = None
    compile_image_encoder: bool = False
    verify_revision: bool = True
    verify_checksum: bool = True

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        if not isinstance(self.repository, Path) or not isinstance(self.checkpoint, Path):
            raise TypeError("repository and checkpoint must be Paths")
        if isinstance(self.memory_window_frames, bool) or self.memory_window_frames < 1:
            raise ValueError("memory_window_frames must be a positive integer")
        if isinstance(self.max_conditioning_frames, bool) or self.max_conditioning_frames < 2:
            raise ValueError("max_conditioning_frames must be at least two")
        if self.recent_memory_frames is not None and (
            isinstance(self.recent_memory_frames, bool) or self.recent_memory_frames < 1
        ):
            raise ValueError("recent_memory_frames must be a positive integer or None")

    @property
    def spec(self) -> ResearchBackendSpec:
        return BACKENDS[self.backend]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _load_predictor(config: ResearchSegmenterConfig) -> Any:
    spec = config.spec
    repository = config.repository.expanduser().resolve()
    checkpoint = config.checkpoint.expanduser().resolve()
    if not (repository / spec.package).is_dir():
        raise BackendUnavailableError(f"{repository} has no {spec.package} package; {_SETUP_HINT}")
    if not checkpoint.is_file():
        raise BackendUnavailableError(f"missing checkpoint {checkpoint}; {_SETUP_HINT}")
    if config.verify_revision:
        revision = _git_revision(repository)
        if revision != spec.revision:
            raise BackendUnavailableError(
                f"{repository} is at revision {revision}, not pinned {spec.revision}; "
                "re-run the replay benchmark before changing the pin"
            )
    if config.verify_checksum:
        digest = file_sha256(checkpoint)
        if digest != spec.checkpoint_sha256:
            raise BackendUnavailableError(
                f"{checkpoint} has SHA-256 {digest}, not pinned {spec.checkpoint_sha256}"
            )
    try:
        import torch
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
    except ModuleNotFoundError as error:
        raise BackendUnavailableError(
            f"{error.name} is required for {spec.name}; install hydra-core and iopath"
        ) from error
    if not config.device.startswith("cuda") or not torch.cuda.is_available():
        raise BackendUnavailableError(f"{spec.name} selection requires a CUDA device")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    try:
        builder_module = importlib.import_module(spec.builder_module)
    except ModuleNotFoundError as error:
        raise BackendUnavailableError(
            f"could not import {spec.builder_module} ({error.name}); {_SETUP_HINT}"
        ) from error
    # Both packages initialize Hydra globally on import, and only the first one wins.
    GlobalHydra.instance().clear()
    initialize_config_module(spec.package, version_base="1.2")
    compile_flag = "true" if config.compile_image_encoder else "false"
    overrides = [
        # The builders' own post-processing defaults, minus hole filling, which needs an
        # optional CUDA extension that is not built here and would otherwise warn per frame.
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        "++model.fill_hole_area=0",
        # Interactive corrections become conditioning memory. Clearing the stale memory
        # around a correction is done by this adapter: at both pinned revisions
        # ``clear_non_cond_mem_around_input=true`` calls a method that does not exist.
        "++model.add_all_frames_to_correct_as_cond=true",
        f"++model.compile_image_encoder={compile_flag}",
    ]
    # TF32 matmul/convolution on Ampere, as the upstream notebooks enable.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    builder = getattr(builder_module, spec.builder)
    return builder(
        spec.config,
        str(checkpoint),
        device=config.device,
        mode="eval",
        hydra_overrides_extra=overrides,
        apply_postprocessing=False,
    )


def _limit_recent_memories(predictor: Any, recent: int) -> None:
    """Attend to only the ``recent`` newest tracked-frame memories (plus conditioning).

    Upstream reads ``maskmem_tpos_enc[num_maskmem - t_pos - 1]``: entry ``k`` encodes a
    memory ``k + 1`` frames old and the last entry encodes conditioning frames. Keeping
    the first ``recent`` entries and the last one preserves every encoding the model was
    trained with for the memories that remain; older memories are simply not attended.
    """

    import torch

    trained = int(predictor.num_maskmem)
    if recent >= trained - 1:
        return
    table = predictor.maskmem_tpos_enc.data
    predictor.maskmem_tpos_enc = torch.nn.Parameter(
        torch.cat((table[:recent], table[trained - 1 :]), dim=0), requires_grad=False
    )
    predictor.num_maskmem = recent + 1


class ResearchVideoSegmenter:
    """``PromptableVideoSegmenter`` over a pinned EfficientTAM or SAM 2 video predictor."""

    def __init__(self, config: ResearchSegmenterConfig) -> None:
        if not isinstance(config, ResearchSegmenterConfig):
            raise TypeError("config must be a ResearchSegmenterConfig")
        import torch

        self._config = config
        self._torch = torch
        self._predictor = _load_predictor(config)
        if config.recent_memory_frames is not None:
            _limit_recent_memories(self._predictor, config.recent_memory_frames)
        self._device = torch.device(config.device)
        self._window = max(
            config.memory_window_frames,
            int(self._predictor.num_maskmem),
            int(self._predictor.max_obj_ptrs_in_encoder),
        )
        image_size = int(self._predictor.image_size)
        self._image_size = image_size
        self._mean = torch.tensor(_IMAGE_MEAN, device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGE_STD, device=self._device).view(1, 3, 1, 1)
        self._state: dict[str, Any] | None = None
        self._frames: OrderedDict[str, tuple[int, AlignedRgbdFrame]] = OrderedDict()
        self._next_index = 0

    @property
    def name(self) -> str:
        return self._config.backend

    @property
    def config(self) -> ResearchSegmenterConfig:
        return self._config

    @property
    def memory_frame_count(self) -> int:
        """Frames currently held in temporal memory (conditioning plus recent)."""

        if self._state is None:
            return 0
        outputs = self._state["output_dict_per_obj"][0]
        return len(outputs["cond_frame_outputs"]) + len(outputs["non_cond_frame_outputs"])

    def reset(self) -> None:
        self._state = None
        self._frames.clear()
        self._next_index = 0

    def seed(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult:
        if not clicks or not any(click.positive for click in clicks):
            raise ValueError("seeding requires at least one positive click")
        self.reset()
        state = self._new_state(frame)
        self._state = state
        with self._inference():
            index = self._admit(frame)
            self._set_image(index, frame)
            self._add_clicks(index, clicks)
            return self._conditioned_result(index, frame)

    def refine(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult:
        if self._state is None:
            raise RuntimeError("seed() is required before refine()")
        if not clicks:
            raise ValueError("refinement requires at least one click")
        remembered = self._frames.get(frame.frameset_id)
        if remembered is None or remembered[1].color.header != frame.color.header:
            raise StalePromptError(
                f"frame {frame.frameset_id} is not among the last {self._window} tracked frames"
            )
        index = remembered[0]
        with self._inference():
            self._set_image(index, frame)
            self._add_clicks(index, clicks)
            return self._conditioned_result(index, frame)

    def track(self, frame: AlignedRgbdFrame) -> SegmentationResult:
        state = self._state
        if state is None:
            raise RuntimeError("seed() is required before track()")
        if (frame.depth.payload.height, frame.depth.payload.width) != (
            state["video_height"],
            state["video_width"],
        ):
            raise ValueError("stream dimensions changed; reselect the object")
        predictor = self._predictor
        with self._inference():
            index = self._admit(frame)
            self._set_image(index, frame)
            outputs = state["output_dict_per_obj"][0]
            current, masks = predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=outputs,
                frame_idx=index,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            outputs["non_cond_frame_outputs"][index] = current
            state["frames_tracked_per_obj"][0][index] = {"reverse": False}
            self._prune(index)
            return self._result(frame, masks, current["object_score_logits"])

    @contextmanager
    def _inference(self) -> Iterator[None]:
        torch = self._torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield

    def _new_state(self, frame: AlignedRgbdFrame) -> dict[str, Any]:
        height, width = frame.depth.payload.height, frame.depth.payload.width
        return {
            "images": {},
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": height,
            "video_width": width,
            "device": self._device,
            "storage_device": self._device,
            "point_inputs_per_obj": {0: {}},
            "mask_inputs_per_obj": {0: {}},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict([(_OBJECT_ID, 0)]),
            "obj_idx_to_id": OrderedDict([(0, _OBJECT_ID)]),
            "obj_ids": [_OBJECT_ID],
            "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
            "temp_output_dict_per_obj": {
                0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            },
            "frames_tracked_per_obj": {0: {}},
        }

    def _admit(self, frame: AlignedRgbdFrame) -> int:
        assert self._state is not None
        if frame.frameset_id in self._frames:
            raise ValueError(f"frame {frame.frameset_id} was already processed")
        index = self._next_index
        self._next_index += 1
        self._frames[frame.frameset_id] = (index, frame)
        self._state["num_frames"] = index + 1
        return index

    def _set_image(self, index: int, frame: AlignedRgbdFrame) -> None:
        """Resize/normalize on the GPU exactly once per frame, as upstream expects."""

        assert self._state is not None
        if index in self._state["cached_features"]:
            return  # upstream reads images only on a feature-cache miss
        torch = self._torch
        pixels = torch.from_numpy(frame.color.payload.data).to(self._device, non_blocking=True)
        image = pixels.permute(2, 0, 1).unsqueeze(0)
        if frame.color.payload.pixel_format is PixelFormat.BGR8:
            image = image.flip(1)
        image = image.float().div_(255.0)
        image = torch.nn.functional.interpolate(
            image,
            size=(self._image_size, self._image_size),
            mode="bicubic",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        image = (image - self._mean) / self._std
        self._state["images"] = {index: image[0]}

    def _add_clicks(self, index: int, clicks: tuple[Click, ...]) -> None:
        assert self._state is not None
        self._predictor.add_new_points_or_box(
            self._state,
            frame_idx=index,
            obj_id=_OBJECT_ID,
            points=np.asarray([[click.x, click.y] for click in clicks], dtype=np.float32),
            labels=np.asarray([int(click.positive) for click in clicks], dtype=np.int32),
            clear_old_points=False,
        )
        # Encodes the prompted frame into memory and makes it a conditioning frame.
        self._predictor.propagate_in_video_preflight(self._state)
        self._state["frames_tracked_per_obj"][0][index] = {"reverse": False}
        # Tracked memory near a correction still shows the uncorrected mask.
        radius = int(self._predictor.num_maskmem)
        stale = self._state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        for key in [key for key in stale if abs(key - index) <= radius]:
            del stale[key]

    def _conditioned_result(self, index: int, frame: AlignedRgbdFrame) -> SegmentationResult:
        assert self._state is not None
        self._prune(max(self._frames[key][0] for key in self._frames))
        output = self._state["output_dict_per_obj"][0]["cond_frame_outputs"][index]
        return self._result(frame, output["pred_masks"], output["object_score_logits"])

    def _prune(self, newest_index: int) -> None:
        state = self._state
        assert state is not None
        oldest_kept = newest_index - self._window + 1
        outputs = state["output_dict_per_obj"][0]
        points = state["point_inputs_per_obj"][0]
        for key in [key for key in outputs["non_cond_frame_outputs"] if key < oldest_kept]:
            del outputs["non_cond_frame_outputs"][key]
        tracked = state["frames_tracked_per_obj"][0]
        for key in [key for key in tracked if key < oldest_kept]:
            del tracked[key]
        while self._frames and next(iter(self._frames.values()))[0] < oldest_kept:
            self._frames.popitem(last=False)
        conditioning = sorted(outputs["cond_frame_outputs"])
        recent = conditioning[1:][-(self._config.max_conditioning_frames - 1) :]
        keep = set(conditioning[:1] + recent)
        for key in conditioning:
            if key not in keep:
                del outputs["cond_frame_outputs"][key]
        for key in [key for key in points if key not in keep and key < oldest_kept]:
            del points[key]

    def _result(
        self, frame: AlignedRgbdFrame, low_res_masks: Any, object_score_logits: Any
    ) -> SegmentationResult:
        assert self._state is not None
        _, video_masks = self._predictor._get_orig_video_res_output(self._state, low_res_masks)
        mask = (video_masks[0, 0] > 0).cpu().numpy()
        score = float(object_score_logits.float().sigmoid().item())
        return SegmentationResult(
            frameset_id=frame.frameset_id,
            source=frame.color.header,
            mask=mask,
            object_score=score if math.isfinite(score) else 0.0,
            backend=self.name,
        )


REPOSITORY_DIRECTORIES = {"efficienttam-ti": "EfficientTAM", "sam2.1-hiera-tiny": "sam2"}
CHECKPOINT_FILES = {
    "efficienttam-ti": "efficienttam_ti.pt",
    "sam2.1-hiera-tiny": "sam2.1_hiera_tiny.pt",
}


def build_research_segmenter(
    backend: str,
    *,
    models_root: Path,
    device: str = "cuda:0",
    memory_window_frames: int = 16,
    max_conditioning_frames: int = 4,
    recent_memory_frames: int | None = None,
    compile_image_encoder: bool = False,
) -> ResearchVideoSegmenter:
    """Construct a backend from the layout ``scripts/setup_selection_models.sh`` creates:
    ``<models_root>/{EfficientTAM,sam2}`` and ``<models_root>/model-checkpoints/*.pt``."""

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    return ResearchVideoSegmenter(
        ResearchSegmenterConfig(
            backend=backend,
            repository=models_root / REPOSITORY_DIRECTORIES[backend],
            checkpoint=models_root / "model-checkpoints" / CHECKPOINT_FILES[backend],
            device=device,
            memory_window_frames=memory_window_frames,
            max_conditioning_frames=max_conditioning_frames,
            recent_memory_frames=recent_memory_frames,
            compile_image_encoder=compile_image_encoder,
        )
    )


__all__ = [
    "BACKENDS",
    "CHECKPOINT_FILES",
    "REPOSITORY_DIRECTORIES",
    "BackendUnavailableError",
    "ResearchBackendSpec",
    "ResearchSegmenterConfig",
    "ResearchVideoSegmenter",
    "build_research_segmenter",
    "file_sha256",
]
