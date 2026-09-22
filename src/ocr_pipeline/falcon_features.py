"""Scoped feature capture from an existing native paged prefill forward pass."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time

import torch


_ACTIVE_MODELS = set()
_REGISTRY_LOCK = threading.Lock()


@dataclass
class LiveFeatures:
    """Compact CPU records; timers are inclusive host elapsed, not GPU timings."""

    records: dict = field(default_factory=dict)
    prefill_calls: int = 0
    forward_calls: int = 0
    skipped_continuations: int = 0
    validation_seconds: float = 0.0
    pooling_seconds: float = 0.0
    post_records: dict = field(default_factory=dict)
    decode_calls: int = 0
    post_pooling_seconds: float = 0.0
    post_completion_seconds: float = 0.0


@contextmanager
def capture_prefill(engine, *, post=False):
    """Capture one synchronous request collection; reject overlapping captures.

    Records commit only after native prefill succeeds. Request indices must remain
    unique within this scope. Continuation prefills never replace initial features.
    """
    model = engine.model
    original_prefill = engine.prefill_sequences
    if not callable(original_prefill) or not callable(model.forward):
        raise TypeError("Native prefill and model forward must be callable")
    if post and (
        not callable(getattr(engine, "decode_step", None))
        or not callable(getattr(engine, "_decode_forward", None))
    ):
        raise TypeError("Native decode methods are required for post capture")
    previous_decode = vars(engine).get("decode_step")
    had_decode = "decode_step" in vars(engine)
    original_decode = engine.decode_step if post else None
    previous_prefill = vars(engine).get("prefill_sequences")
    had_prefill = "prefill_sequences" in vars(engine)
    key = id(model)
    owner = threading.get_ident()
    with _REGISTRY_LOCK:
        if key in _ACTIVE_MODELS:
            raise RuntimeError("Nested or concurrent native feature capture")
        _ACTIVE_MODELS.add(key)
    capture = LiveFeatures()
    call_lock = threading.Lock()
    emissions = EmissionCapture(capture) if post else None

    def prefill(sequences):
        if threading.get_ident() != owner or not call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent native prefill capture is unsupported")
        original_forward = model.forward
        previous_forward = vars(model).get("forward")
        had_forward = "forward" in vars(model)
        pending = {}
        calls = 0
        staged_post = None
        try:
            snapshot = (
                emissions.snapshot(sequences, allow_initial=True) if emissions else None
            )
            indices = [sequence.request_idx for sequence in sequences]
            if any(type(index) is not int or index < 0 for index in indices) or len(
                set(indices)
            ) != len(indices):
                raise ValueError("Native prefill request indices are not unique")
            initial = [
                sequence for sequence in sequences if sequence.output_length == 0
            ]
            if any(sequence.request_idx in capture.records for sequence in initial):
                raise ValueError("Initial request index reused within capture scope")

            def forward(*args, **kwargs):
                nonlocal calls, staged_post
                if threading.get_ident() != owner:
                    raise RuntimeError("Concurrent model forward during native capture")
                calls += 1
                if calls != 1 or args:
                    raise ValueError("Expected one keyword-only native prefill forward")
                started = time.perf_counter()
                try:
                    association = associate_sequences(
                        engine, sequences, kwargs, include_continuations=post
                    )
                finally:
                    capture.validation_seconds += time.perf_counter() - started
                capture.forward_calls += 1
                result = original_forward(**kwargs)
                started = time.perf_counter()
                try:
                    pending.update(
                        pool_features(
                            result,
                            [
                                item
                                for item in association
                                if "image_token_count" in item
                            ],
                        )
                    )
                    if emissions:
                        positions = [item["last_prompt_index"] for item in association]
                        staged_post = emissions.stage(
                            result[1][0, positions], len(sequences)
                        )
                finally:
                    capture.pooling_seconds += time.perf_counter() - started
                return result

            model.forward = forward
            capture.prefill_calls += 1
            result = original_prefill(sequences)
            if calls != 1 or set(pending) != {
                sequence.request_idx for sequence in initial
            }:
                raise ValueError("Initial native prefill features are incomplete")
            capture.records.update(pending)
            if emissions:
                emissions.commit(snapshot, staged_post, "prefill")
            capture.skipped_continuations += len(sequences) - len(initial)
            return result
        finally:
            restore_attribute(model, "forward", had_forward, previous_forward)
            call_lock.release()

    def decode(sequences):
        if threading.get_ident() != owner or not call_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent native decode capture is unsupported")
        original_forward = engine._decode_forward
        previous_forward = vars(engine).get("_decode_forward")
        had_forward = "_decode_forward" in vars(engine)
        staged = None
        calls = 0
        try:
            snapshot = emissions.snapshot(sequences)

            def forward(actual):
                nonlocal calls, staged
                if (
                    threading.get_ident() != owner
                    or len(actual) != len(sequences)
                    or any(a is not b for a, b in zip(actual, sequences, strict=True))
                ):
                    raise ValueError("Decode sequence association changed")
                calls += 1
                if calls != 1:
                    raise ValueError("Expected one native decode forward")
                result = original_forward(actual)
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or result[1].ndim != 3
                    or result[1].shape[0] < len(actual)
                    or result[1].shape[1:] != (1, 768)
                ):
                    raise ValueError("Invalid native decode hidden output")
                # CUDA replay reuses its output buffer; copy before returning it.
                staged = emissions.stage(result[1][: len(actual), -1], len(actual))
                return result

            engine._decode_forward = forward
            capture.decode_calls += 1
            result = original_decode(sequences)
            if calls != 1:
                raise ValueError("Native decode forward was not captured")
            emissions.commit(snapshot, staged, "decode")
            return result
        finally:
            restore_attribute(engine, "_decode_forward", had_forward, previous_forward)
            call_lock.release()

    failure = None
    try:
        engine.prefill_sequences = prefill
        if post:
            engine.decode_step = decode
        yield capture
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        restore_attribute(engine, "prefill_sequences", had_prefill, previous_prefill)
        if post:
            restore_attribute(engine, "decode_step", had_decode, previous_decode)
        try:
            if emissions:
                emissions.finish(failure)
        finally:
            with _REGISTRY_LOCK:
                _ACTIVE_MODELS.remove(key)


def associate_sequences(engine, sequences, arguments, *, include_continuations=False):
    tokens = arguments["tokens"]
    slots = arguments["batch_idx"]
    positions = arguments["input_pos"]
    if (
        tokens.ndim != 2
        or tokens.shape[0] != 1
        or slots.shape != tokens.shape
        or positions.shape != tokens.shape
    ):
        raise ValueError("Expected native packed prefill tensors")
    if arguments["kv_cache"] is not engine.paged_kv_cache:
        raise ValueError("Native prefill KV identity differs")
    sequence_slots = [sequence.batch_idx for sequence in sequences]
    if len(set(sequence_slots)) != len(sequence_slots) or any(
        slot <= 0 for slot in sequence_slots
    ):
        raise ValueError("Native sequence slots must be distinct and positive")
    image_sequences = [
        sequence for sequence in sequences if sequence.image_tensor is not None
    ]
    scatter = arguments.get("img_scatter_info") or []
    pixels = arguments.get("pixel_values") or []
    if len(scatter) != len(image_sequences) or len(pixels) != len(image_sequences):
        raise ValueError("Native image/scatter association is incomplete")
    image_entries = {
        id(sequence): (entry, image)
        for sequence, entry, image in zip(image_sequences, scatter, pixels, strict=True)
    }
    associations = []
    for sequence in sequences:
        packed = (slots[0] == sequence.batch_idx).nonzero(as_tuple=True)[0]
        expected = sequence.total_token_ids.to(device=tokens.device)
        if (
            len(packed) != sequence.total_length
            or not len(packed)
            or int(packed[-1] - packed[0] + 1) != len(packed)
            or not torch.equal(tokens[0, packed], expected)
            or not torch.equal(
                positions[0, packed], torch.arange(len(packed), device=positions.device)
            )
        ):
            raise ValueError("Packed tokens/positions differ from native sequence")
        if sequence.output_length != 0:
            if include_continuations:
                associations.append(
                    {
                        "request_idx": sequence.request_idx,
                        "last_prompt_index": int(packed[-1]),
                    }
                )
            continue
        if id(sequence) not in image_entries:
            raise ValueError("Initial native feature request has no image")
        entry, image = image_entries[id(sequence)]
        image_positions = (sequence.input_ids == engine.model.args.img_id).nonzero(
            as_tuple=True
        )[0]
        start = int(packed[0])
        patch = engine.model.args.spatial_patch_size
        native_shape = sequence.image_tensor.shape
        if (
            not len(image_positions)
            or entry.batch_idx != 0
            or entry.token_start != start + int(image_positions[0])
            or entry.n_tokens != len(image_positions)
            or int(image_positions[-1] - image_positions[0] + 1) != len(image_positions)
            or entry.h_valid_patches != native_shape[1] // patch
            or entry.w_valid_patches != native_shape[2] // patch
            or entry.n_tokens != entry.h_valid_patches * entry.w_valid_patches
            or tuple(image.shape)
            != (
                native_shape[0],
                ((native_shape[1] + patch - 1) // patch) * patch,
                ((native_shape[2] + patch - 1) // patch) * patch,
                native_shape[3],
            )
        ):
            raise ValueError("Native visual scatter does not match request image")
        native_image = torch.as_tensor(
            sequence.image_tensor, device=image.device, dtype=image.dtype
        )
        height, width = native_shape[1:3]
        if (
            not torch.equal(image[:, :height, :width], native_image)
            or torch.count_nonzero(image[:, height:])
            or torch.count_nonzero(image[:, :height, width:])
        ):
            raise ValueError("Native prefill pixels differ from request image")
        associations.append(
            {
                "request_idx": sequence.request_idx,
                "native_slot": sequence.batch_idx,
                "packed_start": start,
                "last_prompt_index": int(packed[-1]),
                "image_token_start": entry.token_start,
                "image_token_count": entry.n_tokens,
                "image_shape": list(native_shape),
                "valid_patch_grid": [entry.h_valid_patches, entry.w_valid_patches],
                "input_token_ids": sequence.input_ids.tolist(),
            }
        )
    return associations


def pool_features(result, associations):
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Expected native logits and normalized hidden output")
    hidden = result[1]
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 768:
        raise ValueError("Expected native packed hidden dimension 768")
    records = {}
    with torch.inference_mode():
        for association in associations:
            start = association["image_token_start"]
            count = association["image_token_count"]
            last = association["last_prompt_index"]
            if start + count > hidden.shape[1] or last >= hidden.shape[1]:
                raise ValueError("Native feature indices exceed hidden output")
            vector = torch.cat(
                [
                    hidden[0, last].float(),
                    hidden[0, start : start + count].float().mean(dim=0),
                ]
            )
            vector = vector.detach().cpu().clone()
            if vector.shape != (1536,) or not torch.isfinite(vector).all():
                raise ValueError("Native live feature vector is nonfinite or malformed")
            records[association["request_idx"]] = association | {"features": vector}
    return records


class EmissionCapture:
    """Pool one predictor state per emitted token, keyed by native Sequence identity."""

    def __init__(self, capture):
        self.capture = capture
        self.states = {}

    def snapshot(self, sequences, *, allow_initial=False):
        indices = [sequence.request_idx for sequence in sequences]
        slots = [sequence.batch_idx for sequence in sequences]
        if (
            any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)
            or any(type(slot) is not int or slot <= 0 for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("Ambiguous native emission identities")
        snapshot = []
        for sequence in sequences:
            index, length = sequence.request_idx, sequence.output_length
            state = self.states.get(index)
            if state is None:
                if not allow_initial or length != 0:
                    raise ValueError("Missing initial emission decision")
                state = {
                    "sequence": sequence,
                    "count": 0,
                    "sum": None,
                    "last": None,
                    "prefill": 0,
                    "decode": 0,
                }
                self.states[index] = state
            if state["sequence"] is not sequence or length != state["count"]:
                raise ValueError("Native emission identity or output length changed")
            snapshot.append((sequence, index, sequence.batch_idx, length))
        return snapshot

    def stage(self, hidden, count):
        started = time.perf_counter()
        try:
            if hidden.shape != (count, 768):
                raise ValueError("Invalid emission decision state shape")
            return hidden.detach().to(dtype=torch.float32, copy=True)
        finally:
            self.capture.post_pooling_seconds += time.perf_counter() - started

    def commit(self, snapshot, staged, kind):
        if staged is None or any(
            sequence.request_idx != index
            or sequence.batch_idx != slot
            or sequence.output_length != length + 1
            for sequence, index, slot, length in snapshot
        ):
            raise ValueError("Native step did not append exactly one token per request")
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                for (_, index, _, _), vector in zip(snapshot, staged, strict=True):
                    state = self.states[index]
                    if state["sum"] is None:
                        state["sum"] = vector.clone()
                    else:
                        state["sum"].add_(vector)
                    state["last"] = vector
                    state["count"] += 1
                    state[kind] += 1
        finally:
            self.capture.post_pooling_seconds += time.perf_counter() - started

    def finish(self, failure):
        started = time.perf_counter()
        try:
            for index, state in self.states.items():
                sequence, count = state["sequence"], state["count"]
                complete = (
                    failure is None
                    and count > 0
                    and count == sequence.output_length
                    and sequence.finished
                )
                error = failure or (
                    None if complete else "incomplete_emission_decisions"
                )
                vector = None
                if count:
                    with torch.inference_mode():
                        vector = (
                            torch.cat((state["last"], state["sum"] / count))
                            .detach()
                            .cpu()
                            .clone()
                        )
                    if not torch.isfinite(vector).all():
                        vector = None
                        error = "nonfinite_emission_decisions"
                        complete = False
                self.capture.post_records[index] = {
                    "features": vector,
                    "status": "success"
                    if complete
                    else "partial"
                    if vector is not None
                    else "failed",
                    "error": error,
                    "request_idx": index,
                    "decision_count": count,
                    "output_length": sequence.output_length,
                    "sequence_finished": bool(sequence.finished),
                    "prefill_decisions": state["prefill"],
                    "decode_decisions": state["decode"],
                }
        finally:
            self.capture.post_completion_seconds += time.perf_counter() - started


def restore_attribute(instance, name, existed, value):
    if existed:
        setattr(instance, name, value)
    else:
        delattr(instance, name)
