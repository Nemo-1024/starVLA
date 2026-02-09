# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.util
from collections import OrderedDict
from threading import Lock
from typing import Any

import av
# av.logging.set_level(av.logging.ERROR)
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Probe decord availability without importing it (avoid FFmpeg symbol conflicts unless needed).
DECORD_AVAILABLE = importlib.util.find_spec("decord") is not None

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


class _TorchcodecDecoderCache:
    """Small per-process LRU cache to avoid re-creating decoders on every sample."""

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, tuple[tuple[str, Any], ...]], Any] = OrderedDict()
        self._lock = Lock()

    def get(self, video_path: str, decoder_kwargs: dict[str, Any]) -> Any:
        key = (video_path, tuple(sorted(decoder_kwargs.items())))
        with self._lock:
            decoder = self._cache.get(key)
            if decoder is not None:
                self._cache.move_to_end(key)
                return decoder

        decoder = torchcodec.decoders.VideoDecoder(video_path, **decoder_kwargs)
        with self._lock:
            decoder = self._cache.setdefault(key, decoder)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return decoder


_TORCHCODEC_DECODER_CACHE = _TorchcodecDecoderCache(max_size=32)


def _get_decord_module():
    """Import decord only when a decord backend is explicitly requested."""
    if not DECORD_AVAILABLE:
        raise ImportError("decord is not available.")
    try:
        return importlib.import_module("decord")
    except Exception as e:
        raise ImportError("decord is installed but failed to import.") from e


def _split_decord_kwargs(video_backend_kwargs: dict | None) -> tuple[dict, bool]:
    """Split kwargs into decord kwargs and fallback strategy."""
    backend_kwargs = dict(video_backend_kwargs) if video_backend_kwargs is not None else {}
    fallback_to_torchvision_av = bool(backend_kwargs.pop("fallback_to_torchvision_av", True))
    # Timestamp-only knobs: avoid passing unsupported keys to VideoReader in other code paths.
    backend_kwargs.pop("use_precise_timestamps", None)
    backend_kwargs.pop("fixed_fps", None)
    backend_kwargs.pop("t0", None)
    return backend_kwargs, fallback_to_torchvision_av


def _split_torchcodec_kwargs(video_backend_kwargs: dict | None) -> tuple[dict[str, Any], bool, bool, float | None, float | None]:
    backend_kwargs = dict(video_backend_kwargs) if video_backend_kwargs is not None else {}

    decoder_kwargs: dict[str, Any] = {}
    for k in ("stream_index", "dimension_order", "num_ffmpeg_threads", "device", "seek_mode"):
        if k in backend_kwargs:
            decoder_kwargs[k] = backend_kwargs.pop(k)
    decoder_kwargs.setdefault("device", "cpu")
    decoder_kwargs.setdefault("dimension_order", "NHWC")
    decoder_kwargs.setdefault("num_ffmpeg_threads", 0)

    use_decoder_cache = bool(backend_kwargs.pop("use_decoder_cache", True))
    use_index_for_timestamps = bool(backend_kwargs.pop("use_index_for_timestamps", True))
    fixed_fps = backend_kwargs.pop("fixed_fps", None)
    t0_override = backend_kwargs.pop("t0", None)

    return decoder_kwargs, use_decoder_cache, use_index_for_timestamps, fixed_fps, t0_override


def _get_torchcodec_decoder(video_path: str, decoder_kwargs: dict[str, Any], use_decoder_cache: bool):
    if not TORCHCODEC_AVAILABLE:
        raise ImportError("torchcodec is not available.")
    if use_decoder_cache:
        return _TORCHCODEC_DECODER_CACHE.get(video_path, decoder_kwargs)
    return torchcodec.decoders.VideoDecoder(video_path, **decoder_kwargs)


def _torchcodec_batch_to_numpy(frames_batch: Any) -> np.ndarray:
    frames = frames_batch.data
    if isinstance(frames, torch.Tensor):
        return frames.cpu().numpy()
    return np.asarray(frames)


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        backend_kwargs, fallback_to_torchvision_av = _split_decord_kwargs(video_backend_kwargs)
        try:
            decord_mod = _get_decord_module()
            vr = decord_mod.VideoReader(video_path, **backend_kwargs)
            if len(vr) == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
            if idx.size == 0:
                raise ValueError("indices must be non-empty")
            idx = np.clip(idx, 0, len(vr) - 1)
            frames = vr.get_batch(idx)
            return frames.asnumpy()
        except Exception:
            if fallback_to_torchvision_av:
                return get_frames_by_indices(
                    video_path,
                    indices,
                    video_backend="torchvision_av",
                    video_backend_kwargs={},
                )
            raise
    elif video_backend == "torchcodec":
        decoder_kwargs, use_decoder_cache, _, _, _ = _split_torchcodec_kwargs(video_backend_kwargs)
        decoder = _get_torchcodec_decoder(video_path, decoder_kwargs, use_decoder_cache)
        num_frames = len(decoder)
        if num_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("indices must be non-empty")
        idx = np.clip(idx, 0, num_frames - 1)
        frames = decoder.get_frames_at(indices=idx.tolist())
        return _torchcodec_batch_to_numpy(frames)
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("indices must be non-empty")
        if np.any(idx < 0):
            raise ValueError("indices must be non-negative")
        max_idx = int(idx.max())
        reader = None
        frame_dict: dict[int, np.ndarray] = {}
        needed = set(int(i) for i in idx.tolist())
        try:
            reader = torchvision.io.VideoReader(video_path, "video")
            for i, frame in enumerate(reader):
                if i > max_idx:
                    break
                if i in needed:
                    frame_data = frame["data"]
                    if isinstance(frame_data, torch.Tensor):
                        frame_data = frame_data.cpu().numpy()
                    frame_dict[i] = frame_data
                    if len(frame_dict) == len(needed):
                        break
            missing = [int(i) for i in idx.tolist() if int(i) not in frame_dict]
            if missing:
                raise ValueError(f"Unable to load frame indices {missing} from {video_path}")
        finally:
            if reader is not None and hasattr(reader, "container"):
                reader.container.close()
            reader = None
        frames = np.asarray([frame_dict[int(i)] for i in idx.tolist()])
        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        backend_kwargs = dict(video_backend_kwargs) if video_backend_kwargs is not None else {}
        fallback_to_torchvision_av = bool(backend_kwargs.pop("fallback_to_torchvision_av", True))
        use_precise = bool(backend_kwargs.pop("use_precise_timestamps", False))
        fixed_fps = backend_kwargs.pop("fixed_fps", None)
        t0_override = backend_kwargs.pop("t0", None)

        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == 0:
            raise ValueError("timestamps must be non-empty")

        try:
            decord_mod = _get_decord_module()
            vr = decord_mod.VideoReader(video_path, **backend_kwargs)
            num_frames = len(vr)
            if num_frames <= 0:
                raise ValueError(f"Video has no frames: {video_path}")

            # Fast path: map timestamps -> indices via FPS (O(K)).
            # This avoids calling get_frame_timestamp(range(num_frames)) which can be O(num_frames) per sample.
            # If you need more accurate mapping for variable-fps videos, set:
            #   video_backend_kwargs={"use_precise_timestamps": True}
            indices: np.ndarray
            if not use_precise:
                if fixed_fps is not None:
                    try:
                        fps = float(fixed_fps)
                    except Exception:
                        fps = 0.0
                else:
                    try:
                        fps = float(vr.get_avg_fps())
                    except Exception:
                        fps = 0.0
                if fps > 0:
                    # decord frame timestamps may start at a non-zero offset (e.g., absolute time).
                    # Align by the first frame timestamp when possible.
                    if t0_override is not None:
                        try:
                            t0 = float(t0_override)
                        except Exception:
                            t0 = 0.0
                    else:
                        try:
                            t0 = float(np.asarray(vr.get_frame_timestamp([0]))[0, 0])
                        except Exception:
                            t0 = 0.0
                    indices = np.rint((ts - t0) * fps).astype(np.int64)
                    indices = np.clip(indices, 0, num_frames - 1)
                else:
                    use_precise = True

            if use_precise:
                # Retrieve the timestamps for each frame in the video
                frame_ids = np.arange(num_frames, dtype=np.int64)
                frame_ts: np.ndarray = vr.get_frame_timestamp(frame_ids)
                frame_ts_1d = np.asarray(frame_ts[:, 0], dtype=np.float64)  # start_seconds
                indices = np.abs(frame_ts_1d[:, None] - ts[None, :]).argmin(axis=0).astype(
                    np.int64
                )

            frames = vr.get_batch(indices)
            return frames.asnumpy()
        except Exception:
            if fallback_to_torchvision_av:
                return get_frames_by_timestamps(
                    video_path,
                    ts,
                    video_backend="torchvision_av",
                    video_backend_kwargs={},
                )
            raise
    elif video_backend == "torchcodec":
        decoder_kwargs, use_decoder_cache, use_index_for_timestamps, fixed_fps, t0_override = (
            _split_torchcodec_kwargs(video_backend_kwargs)
        )
        decoder = _get_torchcodec_decoder(video_path, decoder_kwargs, use_decoder_cache)
        num_frames = len(decoder)
        if num_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == 0:
            raise ValueError("timestamps must be non-empty")

        if use_index_for_timestamps:
            try:
                if fixed_fps is not None:
                    fps = float(fixed_fps)
                else:
                    fps = float(decoder.metadata.average_fps)
                if t0_override is not None:
                    t0 = float(t0_override)
                else:
                    t0 = float(decoder.metadata.begin_stream_seconds)
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("invalid fps")
                indices = np.rint((ts - t0) * fps).astype(np.int64)
                indices = np.clip(indices, 0, num_frames - 1)
                frames = decoder.get_frames_at(indices=indices.tolist())
                return _torchcodec_batch_to_numpy(frames)
            except Exception:
                # Fall back to precise timestamp lookup if fps metadata is missing/inaccurate.
                pass

        frames = decoder.get_frames_played_at(seconds=ts.tolist())
        return _torchcodec_batch_to_numpy(frames)
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.size == 0:
            raise ValueError("timestamps must be non-empty")

        loaded_frames: list[np.ndarray] = []
        loaded_ts: list[float] = []
        reader = None
        try:
            reader = torchvision.io.VideoReader(video_path, "video")

            first_ts = float(ts.min())
            last_ts = float(ts.max())
            # Seek once to the closest keyframe before the first requested timestamp.
            reader.seek(first_ts, keyframes_only=True)

            # Sequentially read until covering the last requested timestamp.
            for frame in reader:
                current_ts = float(frame["pts"])
                frame_data = frame["data"]
                if isinstance(frame_data, torch.Tensor):
                    frame_data = frame_data.cpu().numpy()
                loaded_frames.append(frame_data)
                loaded_ts.append(current_ts)
                if current_ts >= last_ts:
                    break

            if len(loaded_frames) == 0:
                raise RuntimeError(
                    f"No frames loaded from {video_path} for timestamps={ts.tolist()}"
                )

            loaded_ts_arr = np.asarray(loaded_ts, dtype=np.float64)
            # Nearest-neighbor timestamp matching: output order follows requested timestamps.
            nearest_idx = np.abs(loaded_ts_arr[:, None] - ts[None, :]).argmin(axis=0)
            selected_frames = [loaded_frames[int(i)] for i in nearest_idx]
        finally:
            # Explicitly close container resources; avoid per-call global GC.
            if reader is not None:
                if hasattr(reader, "container"):
                    reader.container.close()
            reader = None

        frames = np.asarray(selected_frames)
        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        backend_kwargs, fallback_to_torchvision_av = _split_decord_kwargs(video_backend_kwargs)
        try:
            decord_mod = _get_decord_module()
            vr = decord_mod.VideoReader(video_path, **backend_kwargs)
            if len(vr) == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            frame_ids = np.arange(len(vr), dtype=np.int64)
            frames = vr.get_batch(frame_ids).asnumpy()
        except Exception:
            if fallback_to_torchvision_av:
                frames = get_all_frames(
                    video_path,
                    video_backend="torchvision_av",
                    video_backend_kwargs={},
                    resize_size=None,
                )
            else:
                raise
    elif video_backend == "torchcodec":
        decoder_kwargs, use_decoder_cache, _, _, _ = _split_torchcodec_kwargs(video_backend_kwargs)
        decoder = _get_torchcodec_decoder(video_path, decoder_kwargs, use_decoder_cache)
        num_frames = len(decoder)
        if num_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        frame_ids = np.arange(num_frames, dtype=np.int64)
        frames = decoder.get_frames_at(indices=frame_ids.tolist())
        frames = _torchcodec_batch_to_numpy(frames)
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
