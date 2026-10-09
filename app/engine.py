from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
from typing import Any, AsyncIterator, Optional

import numpy as np

from app.config import Settings


log = logging.getLogger(__name__)

# VoxCPM's repo ships its package under ``engine/src/voxcpm/``. The Dockerfile
# sets PYTHONPATH accordingly, but for local runs we also honour
# ``VOXCPM_ROOT`` (pointing at the submodule checkout) as a fallback.
_voxcpm_root = os.environ.get("VOXCPM_ROOT")
if _voxcpm_root:
    for _p in (f"{_voxcpm_root}/src", _voxcpm_root):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


@contextlib.contextmanager
def _prefer_cuda_weight_loading(device: str):
    """Load VoxCPM checkpoint tensors on CUDA directly when supported."""
    if not device.startswith("cuda"):
        yield
        return

    try:
        import voxcpm.model.voxcpm as voxcpm_v1
        import voxcpm.model.voxcpm2 as voxcpm_v2
        from safetensors.torch import load_file as safetensors_load_file
    except Exception:
        yield
        return

    patched = []

    def _patch_module(module):
        original_load_file = getattr(module, "load_file", None)
        if original_load_file is not None:
            patched.append((module, "load_file", original_load_file))

            def _load_file(path, *args, **kwargs):
                if "device" not in kwargs:
                    kwargs["device"] = device
                return safetensors_load_file(path, *args, **kwargs)

            module.load_file = _load_file

        torch_mod = getattr(module, "torch", None)
        original_torch_load = getattr(torch_mod, "load", None) if torch_mod else None
        if original_torch_load is not None:
            patched.append((torch_mod, "load", original_torch_load))

            def _torch_load(f, *args, **kwargs):
                filename = str(f)
                if (
                    kwargs.get("map_location") == "cpu"
                    and os.path.basename(filename) == "pytorch_model.bin"
                ):
                    kwargs["map_location"] = device
                return original_torch_load(f, *args, **kwargs)

            torch_mod.load = _torch_load

    _patch_module(voxcpm_v1)
    _patch_module(voxcpm_v2)

    try:
        yield
    finally:
        for owner, attr, original in reversed(patched):
            setattr(owner, attr, original)


class TTSEngine:
    """Thin async wrapper around :class:`voxcpm.VoxCPM`.

    Public methods mirror the open-tts engine contract (``synthesize_clone`` /
    ``synthesize_design`` / ``synthesize_realtime``) and translate the
    normalised request fields into VoxCPM's native ``generate`` /
    ``generate_streaming`` API.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = settings.resolved_device
        # CPU path cannot reliably compile nor run fp16 inference.
        self._optimize = settings.voxcpm_compile and self._device.startswith("cuda")
        self._dtype_str = settings.effective_dtype
        if settings.voxcpm_compile and not self._device.startswith("cuda"):
            log.warning(
                "voxcpm_compile=true but device=%s; disabling torch.compile on CPU",
                self._device,
            )

        self._model = self._load_model()
        self._sample_rate = int(self._model.tts_model.sample_rate)
        self._is_v2 = self._detect_v2()
        self._clone_mode = self._resolve_default_clone_mode(
            settings.voxcpm_clone_mode
        )

    # ------------------------------------------------------------------
    # Public attributes

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_str(self) -> str:
        return self._dtype_str

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_id(self) -> str:
        return self._settings.voxcpm_model

    @property
    def is_v2(self) -> bool:
        return self._is_v2

    @property
    def clone_mode(self) -> str:
        return self._clone_mode

    # VoxCPM has no built-in voices; the server layer consults this list to
    # decide whether to advertise `builtin_voices=true`.
    @property
    def builtin_voices_list(self) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Model loading

    def _load_model(self):
        from voxcpm import VoxCPM

        with _prefer_cuda_weight_loading(self._device):
            return VoxCPM.from_pretrained(
                hf_model_id=self._settings.voxcpm_model,
                load_denoiser=self._settings.voxcpm_load_denoiser,
                zipenhancer_model_id=self._settings.voxcpm_zipenhancer_model,
                cache_dir=None,
                local_files_only=self._settings.voxcpm_local_files_only,
                optimize=self._optimize,
                device=None if self._settings.voxcpm_device == "auto" else self._device,
            )

    def _detect_v2(self) -> bool:
        try:
            from voxcpm.model.voxcpm2 import VoxCPM2Model
        except Exception:  # pragma: no cover - import guard
            return False
        return isinstance(self._model.tts_model, VoxCPM2Model)

    def _resolve_default_clone_mode(self, requested: str) -> str:
        if requested in ("reference", "ref_continuation") and not self._is_v2:
            log.warning(
                "VOXCPM_CLONE_MODE=%s requires VoxCPM2; current model is not v2. "
                "Falling back to 'continuation'.",
                requested,
            )
            return "continuation"
        return requested

    # ------------------------------------------------------------------
    # Request-field helpers

    def _apply_instructions(self, text: str, instructions: Optional[str]) -> str:
        """Prepend VoxCPM-style ``(<instructions>)`` control syntax."""
        if instructions is None:
            return text
        trimmed = instructions.strip()
        if not trimmed:
            return text
        return f"({trimmed}){text}"

    def _resolve_clone_mode(self, requested: Optional[str]) -> str:
        mode = requested or self._clone_mode
        if mode in ("reference", "ref_continuation") and not self._is_v2:
            log.warning(
                "clone_mode=%s requested but model is not VoxCPM2; "
                "using 'continuation' instead",
                mode,
            )
            return "continuation"
        return mode

    def _gen_kwargs(self, engine_specific: dict[str, Any]) -> dict[str, Any]:
        """Merge per-request overrides with the configured defaults."""
        s = self._settings

        def pick(key: str, default):
            override = engine_specific.get(key)
            return default if override is None else override

        return {
            "cfg_value": pick("cfg_value", s.voxcpm_default_cfg_value),
            "inference_timesteps": pick(
                "inference_timesteps", s.voxcpm_default_inference_timesteps
            ),
            "min_len": pick("min_len", s.voxcpm_default_min_len),
            "max_len": pick("max_len", s.voxcpm_default_max_len),
            "normalize": pick("normalize", s.voxcpm_default_normalize),
            "denoise": pick("denoise", s.voxcpm_default_denoise),
            "retry_badcase": pick("retry_badcase", s.voxcpm_default_retry_badcase),
            "retry_badcase_max_times": pick(
                "retry_badcase_max_times", s.voxcpm_default_retry_badcase_max_times
            ),
            "retry_badcase_ratio_threshold": pick(
                "retry_badcase_ratio_threshold",
                s.voxcpm_default_retry_badcase_ratio_threshold,
            ),
        }

    def _clone_call_kwargs(
        self,
        *,
        ref_audio: str,
        ref_text: str,
        mode: str,
    ) -> dict[str, Any]:
        """Map (ref_audio, ref_text, mode) to VoxCPM.generate kwargs.

        Raises ValueError when the caller asks for ref_text='' on a
        non-VoxCPM2 model — the server layer converts it to HTTP 422.
        """
        has_text = bool(ref_text and ref_text.strip())
        if not has_text:
            if not self._is_v2:
                raise ValueError(
                    "empty ref_text (reference-only mode) requires VoxCPM2"
                )
            return {"reference_wav_path": ref_audio}

        if mode == "reference":
            return {"reference_wav_path": ref_audio}
        if mode == "ref_continuation":
            return {
                "prompt_wav_path": ref_audio,
                "prompt_text": ref_text,
                "reference_wav_path": ref_audio,
            }
        # continuation (default, works on v1/v1.5/v2)
        return {"prompt_wav_path": ref_audio, "prompt_text": ref_text}

    # ------------------------------------------------------------------
    # Non-streaming synthesis paths

    async def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio: str,
        ref_text: str,
        ref_mtime: Optional[float] = None,  # noqa: ARG002 - reserved for future LRU
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **engine_specific: Any,
    ) -> np.ndarray:
        if speed != 1.0:
            log.info("speed=%s ignored: VoxCPM has no native speed control", speed)

        clone_mode = self._resolve_clone_mode(engine_specific.pop("clone_mode", None))
        call_kwargs = self._clone_call_kwargs(
            ref_audio=ref_audio, ref_text=ref_text, mode=clone_mode
        )
        call_kwargs.update(self._gen_kwargs(engine_specific))
        final_text = self._apply_instructions(text, instructions)

        def _run() -> np.ndarray:
            wav = self._model.generate(text=final_text, **call_kwargs)
            return np.asarray(wav, dtype=np.float32).reshape(-1)

        return await asyncio.to_thread(_run)

    async def synthesize_design(
        self,
        text: str,
        *,
        instruct: Optional[str] = None,
        **engine_specific: Any,
    ) -> np.ndarray:
        call_kwargs = self._gen_kwargs(engine_specific)
        # design path has no reference audio -> denoise has nothing to denoise.
        call_kwargs["denoise"] = False
        final_text = self._apply_instructions(text, instruct)

        def _run() -> np.ndarray:
            wav = self._model.generate(text=final_text, **call_kwargs)
            return np.asarray(wav, dtype=np.float32).reshape(-1)

        return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Streaming synthesis

    async def synthesize_realtime(
        self,
        text: str,
        *,
        kind: str = "clone",
        voice: str = "",  # noqa: ARG002 - routing handled by the server
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_mtime: Optional[float] = None,  # noqa: ARG002 - reserved
        instructions: Optional[str] = None,
        speed: float = 1.0,
        **engine_specific: Any,
    ) -> AsyncIterator[np.ndarray]:
        if speed != 1.0:
            log.info("speed=%s ignored: VoxCPM has no native speed control", speed)

        if kind == "clone":
            if ref_audio is None:
                raise ValueError("clone streaming requires ref_audio")
            clone_mode = self._resolve_clone_mode(
                engine_specific.pop("clone_mode", None)
            )
            call_kwargs = self._clone_call_kwargs(
                ref_audio=ref_audio, ref_text=ref_text or "", mode=clone_mode
            )
        elif kind == "design":
            engine_specific.pop("clone_mode", None)
            call_kwargs = {}
        else:
            raise ValueError(f"unknown realtime kind: {kind!r}")

        call_kwargs.update(self._gen_kwargs(engine_specific))
        if kind == "design":
            call_kwargs["denoise"] = False
        final_text = self._apply_instructions(text, instructions)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        sentinel = object()

        def _producer() -> None:
            try:
                gen = self._model.generate_streaming(
                    text=final_text, **call_kwargs
                )
                for chunk in gen:
                    arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    asyncio.run_coroutine_threadsafe(queue.put(arr), loop).result()
            except Exception as exc:  # pragma: no cover - surfaced via stream
                log.exception("streaming producer failed")
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
                except Exception:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(sentinel), loop
                    ).result()
                except Exception:
                    pass

        thread = threading.Thread(
            target=_producer, name="voxcpm-stream", daemon=True
        )
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
