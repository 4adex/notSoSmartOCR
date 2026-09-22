"""Isolated training path for the official Falcon-OCR inference checkpoint."""

import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import (
    AuxRequest,
    and_masks,
    or_masks,
    create_block_mask,
    flex_attention,
)
from falcon_perception.attention import (
    get_causal_mask_mod,
    get_document_mask_mod,
    get_non_left_pad_mask_mod,
    get_image_prefix_mask_mod,
)
from falcon_perception.model import ImgScatterEntry
from falcon_perception.rope import apply_golden_freqs_cis_to_visual_pos

attention = torch.compile(flex_attention, dynamic=True)


def prepare_batch(model, tokenizer, prompt, transcripts):
    """Official processed prompts plus tokenized transcripts, including the task stop token."""
    if len(transcripts) != prompt["tokens"].shape[0] or any(
        not ids for ids in transcripts
    ):
        raise ValueError("One nonempty target token sequence per prompt is required")
    length = max(map(len, transcripts))
    device = model.device
    targets = torch.full(
        (len(transcripts), length), -100, dtype=torch.long, device=device
    )
    for row, ids in enumerate(transcripts):
        targets[row, : len(ids)] = torch.tensor(ids, device=device)
    continuation = targets[:, :-1].masked_fill(
        targets[:, :-1] == -100, tokenizer.pad_token_id
    )
    tokens = torch.cat((prompt["tokens"], continuation), dim=1)
    if tokens.shape[1] > model.args.max_seq_len:
        raise ValueError("Training sequence exceeds model context")
    positions = torch.arange(1, length, device=device)[None, :]
    pos_t = torch.cat(
        (prompt["pos_t"], prompt["pos_t"][:, -1:] + positions), dim=1
    ).long()
    pos_hw = torch.cat(
        (
            prompt["pos_hw"],
            prompt["pos_hw"].new_zeros((len(transcripts), length - 1, 2)),
        ),
        dim=1,
    )
    # Match native prefill's full-block classification at the partial final block.
    mask_length = (tokens.shape[1] + 127) // 128 * 128
    mask_tokens = F.pad(
        tokens, (0, mask_length - tokens.shape[1]), value=tokenizer.pad_token_id
    )
    causal = and_masks(
        get_causal_mask_mod(),
        get_document_mask_mod(mask_tokens, tokenizer.eos_token_id),
        get_non_left_pad_mask_mod(mask_tokens, tokenizer.pad_token_id),
    )
    mask_mod = or_masks(
        get_image_prefix_mask_mod(
            mask_tokens, tokenizer.image_cls_token_id, tokenizer.end_of_image_token_id
        ),
        causal,
    )
    # Unlike the inference helper, these tensors must support backward saves.
    mask = create_block_mask(
        mask_mod,
        tokens.shape[0],
        None,
        mask_length,
        mask_length,
        device=str(device),
    )
    mask.seq_lengths = (tokens.shape[1], tokens.shape[1])
    scatter = []
    for row, (ids, pixel_mask) in enumerate(
        zip(prompt["tokens"].cpu(), prompt["pixel_mask"].cpu())
    ):
        indices = (ids == model.args.img_id).nonzero(as_tuple=True)[0]
        if indices.numel():
            ps = model.args.spatial_patch_size
            h = int(pixel_mask.sum(dim=-2).max()) // ps
            w = int(pixel_mask.sum(dim=-1).max()) // ps
            scatter.append(ImgScatterEntry(row, int(indices[0]), len(indices), h, w))
    return dict(
        tokens=tokens,
        pos_t=pos_t,
        pos_hw=pos_hw,
        mask=mask,
        scatter=scatter,
        pixels=prompt["pixel_values"],
        target_mask=prompt.get("target_mask"),
        targets=targets,
        start=prompt["tokens"].shape[1] - 1,
    )


def forward(model, batch):
    """No cache writes, exact sink-adjusted attention, transcript-position logits."""
    if model.args.perception_heads:
        raise ValueError("This transcript-only training path requires Falcon-OCR")
    h = model.tok_embeddings(batch["tokens"])
    with target_conditioning(model, batch.get("target_mask")):
        h = model._scatter_img_tokens_with_projector(
            h, batch["pixels"], batch["scatter"]
        )
    freqs = model.freqs_cis[batch["pos_t"]]
    visual_freqs = apply_golden_freqs_cis_to_visual_pos(
        model.freqs_cis_golden, batch["pos_hw"]
    )
    for layer in model.layers.values():
        args = (h, layer, freqs, visual_freqs, batch["mask"])
        h = (
            checkpoint(forward_layer, *args, use_reentrant=False)
            if model.training and torch.is_grad_enabled()
            else forward_layer(*args)
        )
    return model.output(model.norm(h))[:, batch["start"] :]


def forward_layer(h, layer, freqs, visual_freqs, mask):
    q, k, v = layer.attention._pre_attention(h, freqs, visual_freqs)
    output, auxiliary = attention(
        q,
        k,
        v,
        block_mask=mask,
        return_aux=AuxRequest(lse=True),
        kernel_options={"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1},
    )
    h = h + layer.attention._post_attention(output, auxiliary.lse)
    normalized = F.rms_norm(h, (h.shape[-1],))
    packed = layer.feed_forward.w13(normalized)
    # Weight rows interleave gate/up projections, with gate scaling already folded in.
    activated = F.relu(packed[..., 0::2]).square() * packed[..., 1::2]
    return h + layer.feed_forward.w2(activated)


def add_target_embedding(model):
    model.register_parameter(
        "target_embedding",
        nn.Parameter(
            torch.zeros(model.args.dim, device=model.device, dtype=torch.float32)
        ),
    )


def target_patch_fractions(model, prompt, polygons):
    """Native word-polygon union coverage on the actual valid visual patch grid."""
    import numpy as np
    from shapely import area, box, intersection
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if prompt["tokens"].shape[0] != 1 or prompt["pixel_mask"].shape[1] != 1:
        raise ValueError("Target conditioning requires one image per invocation")
    mask = prompt["pixel_mask"][0].cpu()
    height = int(mask.sum(dim=-2).max())
    width = int(mask.sum(dim=-1).max())
    ps = model.args.spatial_patch_size
    count = int((prompt["tokens"] == model.args.img_id).sum())
    positions = (prompt["tokens"][0] == model.args.img_id).nonzero().flatten()
    if (
        height % ps
        or width % ps
        or count != height * width // ps**2
        or int(mask.sum()) != height * width
        or not mask[0, :height, :width].all()
        or (positions[1:] - positions[:-1] != 1).any()
    ):
        raise ValueError("Native valid pixels and visual-token grid disagree")
    if not polygons or any(
        len(points) < 3
        or any(
            len(point) != 2
            or any(
                not math.isfinite(value) or not 0 <= value <= 1000 for value in point
            )
            for point in points
        )
        for points in polygons
    ):
        raise ValueError("Target polygons must use finite normalized image coordinates")
    shapes = [
        Polygon([(x * width / 1000, y * height / 1000) for x, y in points])
        for points in polygons
    ]
    if any(not shape.is_valid or shape.area <= 0 for shape in shapes):
        raise ValueError("Target polygon has invalid geometry")
    target = unary_union(shapes)
    ys = np.repeat(np.arange(height // ps) * ps, width // ps)
    xs = np.tile(np.arange(width // ps) * ps, height // ps)
    left, top, right, bottom = target.bounds
    selected = (xs < right) & (xs + ps > left) & (ys < bottom) & (ys + ps > top)
    fractions = np.zeros(count, dtype=np.float32)
    patches = box(xs[selected], ys[selected], xs[selected] + ps, ys[selected] + ps)
    fractions[selected] = area(intersection(target, patches)) / ps**2
    return torch.from_numpy(fractions)


@contextmanager
def target_conditioning(model, fractions):
    """Apply one request's additive target signal in both native and training paths."""
    if getattr(model, "_target_mask_active", False):
        raise ValueError("Target conditioning is already active")
    if fractions is None:
        yield
        return
    if not hasattr(model, "target_embedding"):
        raise ValueError("Target conditioning parameter is missing")
    if (
        fractions.ndim != 1
        or not torch.isfinite(fractions).all()
        or not ((fractions >= 0) & (fractions <= 1)).all()
    ):
        raise ValueError(
            "Target patch fractions must be finite and between zero and one"
        )
    model._target_mask_active = True

    def add_signal(module, inputs, output):
        if output.shape != (len(fractions), model.args.dim):
            raise ValueError(
                "Target patch fractions and projected visual tokens disagree"
            )
        signal = (
            fractions.to(device=output.device, dtype=torch.float32)[:, None]
            * model.target_embedding[None, :]
        )
        return output + signal.to(output.dtype)

    handle = None
    try:
        handle = model.img_projector.register_forward_hook(add_signal)
        yield
    finally:
        if handle is not None:
            handle.remove()
        model._target_mask_active = False


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(
            torch.empty(
                (rank, base.in_features), device=base.weight.device, dtype=torch.float32
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                (base.out_features, rank),
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0  # alpha equals rank

    def forward(self, x):
        return self.base(x) + (
            F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        ).to(x.dtype)


def add_lora(model, rank=8):
    model.requires_grad_(False)
    for layer in model.layers.values():
        for module, names in (
            (layer.attention, ("wqkv", "wo")),
            (layer.feed_forward, ("w13", "w2")),
        ):
            for name in names:
                setattr(module, name, LoRALinear(getattr(module, name), rank))
    if hasattr(model, "target_embedding"):
        model.target_embedding.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def load_adapter(model, path, rank=8):
    """Restore only complete, finite unmerged adapter tensors onto the original base."""
    add_lora(model, rank=rank)
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not isinstance(state, dict) or set(state) != set(parameters):
        raise ValueError("Adapter keys do not match the Falcon projection adapters")
    for name, parameter in parameters.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])
            if not torch.equal(parameter.cpu(), state[name]):
                raise ValueError(f"Adapter tensor did not reload exactly: {name}")
    model.requires_grad_(False)
    return model.eval()
