"""Dense grouped linear kernels for Arrow cell populations.

The semantic reference uses compact ``einsum`` expressions.  The grouped
backend reshapes the population into batched matrix multiplications so CUDA
libraries and ``torch.compile`` see regular GEMMs instead of many implicit
cell-local contractions. Optional zero padding aligns reduction/output
dimensions without adding trainable parameters.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


DenseKernelBackend = Literal["einsum", "grouped_bmm"]
DenseNodeBackend = Literal["einsum", "folded_bmm", "monarch"]


def _validate_padding(pad_to_multiple: int) -> None:
    if (
        isinstance(pad_to_multiple, bool)
        or not isinstance(pad_to_multiple, int)
        or pad_to_multiple <= 0
    ):
        raise ValueError("pad_to_multiple must be a positive integer")


def _padded_size(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def _pad_linear_operands(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pad_to_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Zero-pad the final input/output dimensions of linear operands."""

    _validate_padding(pad_to_multiple)
    input_features = inputs.shape[-1]
    output_features = weight.shape[-2]
    padded_input = _padded_size(input_features, pad_to_multiple)
    padded_output = _padded_size(output_features, pad_to_multiple)
    input_padding = padded_input - input_features
    output_padding = padded_output - output_features
    if input_padding:
        inputs = F.pad(inputs, (0, input_padding))
    if input_padding or output_padding:
        weight = F.pad(
            weight,
            (0, input_padding, 0, output_padding),
        )
    if bias is not None and output_padding:
        bias = F.pad(bias, (0, output_padding))
    return inputs, weight, bias, output_features


def cell_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backend: DenseKernelBackend = "grouped_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply independent cell weights to ``[B,C,T,I]`` inputs.

    ``weight`` and ``bias`` have shapes ``[C,O,I]`` and ``[C,O]``. The result
    has shape ``[B,C,T,O]``.
    """

    if inputs.ndim != 4:
        raise ValueError("cell inputs must have [batch,cells,tokens,input] shape")
    if weight.ndim != 3 or (bias is not None and bias.ndim != 2):
        raise ValueError("cell weight/bias must have [cells,output,input] shapes")
    batch, cells, tokens, input_features = inputs.shape
    if weight.shape[0] != cells or weight.shape[2] != input_features:
        raise ValueError("cell weight must match input cells/features")
    if bias is not None and bias.shape != weight.shape[:2]:
        raise ValueError("cell bias must match weight cells/output")
    if backend == "einsum":
        output = torch.einsum("bcti,coi->bcto", inputs, weight)
        return output if bias is None else output + bias[None, :, None]
    if backend != "grouped_bmm":
        raise ValueError("dense kernel backend must be 'einsum' or 'grouped_bmm'")

    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    grouped_inputs = inputs.permute(1, 0, 2, 3).reshape(
        cells, batch * tokens, inputs.shape[-1]
    )
    grouped_output = torch.bmm(
        grouped_inputs,
        weight.transpose(1, 2),
    )
    if bias is not None:
        grouped_output = grouped_output + bias[:, None]
    return grouped_output[..., :output_features].reshape(
        cells, batch, tokens, output_features
    ).permute(1, 0, 2, 3)


def cell_node_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    backend: DenseKernelBackend = "grouped_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply independent cell/node weights to ``[B,C,T,N,I]`` inputs."""

    if inputs.ndim != 5:
        raise ValueError(
            "cell-node inputs must have [batch,cells,tokens,nodes,input] shape"
        )
    if weight.ndim != 4 or bias.ndim != 3:
        raise ValueError(
            "cell-node weight/bias must have [cells,nodes,output,input] shapes"
        )
    batch, cells, tokens, nodes, input_features = inputs.shape
    if (
        weight.shape[0] != cells
        or weight.shape[1] != nodes
        or weight.shape[3] != input_features
    ):
        raise ValueError("cell-node weight must match input cells/nodes/features")
    if bias.shape != weight.shape[:3]:
        raise ValueError("cell-node bias must match weight cells/nodes/output")
    if backend == "einsum":
        return (
            torch.einsum("bctni,cnoi->bctno", inputs, weight)
            + bias[None, :, None]
        )
    if backend != "grouped_bmm":
        raise ValueError("dense kernel backend must be 'einsum' or 'grouped_bmm'")

    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    grouped_inputs = inputs.permute(1, 3, 0, 2, 4).reshape(
        cells * nodes,
        batch * tokens,
        inputs.shape[-1],
    )
    grouped_weight = weight.reshape(
        cells * nodes, weight.shape[-2], weight.shape[-1]
    )
    grouped_bias = bias.reshape(cells * nodes, bias.shape[-1])
    grouped_output = torch.bmm(
        grouped_inputs,
        grouped_weight.transpose(1, 2),
    )
    grouped_output = grouped_output + grouped_bias[:, None]
    return grouped_output[..., :output_features].reshape(
        cells, nodes, batch, tokens, output_features
    ).permute(2, 0, 3, 1, 4)


def padded_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Apply one shared linear transform with parameter-free tile padding."""

    if inputs.ndim < 2 or weight.ndim != 2:
        raise ValueError("linear inputs/weight must have [...,input] and [output,input]")
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError("linear input and weight features must match")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("linear bias must match weight output")
    inputs, weight, bias, output_features = _pad_linear_operands(
        inputs,
        weight,
        bias,
        pad_to_multiple=pad_to_multiple,
    )
    return F.linear(inputs, weight, bias)[..., :output_features]


def fold_cell_node_parameters(
    node_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold N affine node transforms into one wide transform per cell.

    For node ``n`` the original affine input is ``base + node_embedding[n]``.
    Linearity permits:

    ``W[n] @ base + (W[n] @ node_embedding[n] + bias[n])``.

    The first term becomes one ``D -> N*O`` grouped GEMM while the second term
    is a small relay-invariant offset.
    """

    if node_embedding.ndim != 3 or weight.ndim != 4 or bias.ndim != 3:
        raise ValueError(
            "node embedding/weight/bias must have [C,N,I], [C,N,O,I], "
            "and [C,N,O] shapes"
        )
    cells, nodes, input_features = node_embedding.shape
    if weight.shape[:2] != (cells, nodes) or weight.shape[-1] != input_features:
        raise ValueError("node weight must match embedding cells/nodes/features")
    if bias.shape != weight.shape[:3]:
        raise ValueError("node bias must match weight cells/nodes/output")
    output_features = weight.shape[-2]
    folded_weight = weight.reshape(
        cells, nodes * output_features, input_features
    )
    offset = (
        torch.einsum("cni,cnoi->cno", node_embedding, weight)
        + bias
    )
    return folded_weight, offset


def folded_cell_node_linear(
    base: torch.Tensor,
    node_embedding: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    backend: DenseNodeBackend = "folded_bmm",
    pad_to_multiple: int = 1,
) -> torch.Tensor:
    """Evaluate node-affine outputs without materializing repeated node input."""

    if base.ndim != 4:
        raise ValueError("base must have [batch,cells,tokens,input] shape")
    if backend == "einsum":
        return cell_node_linear(
            base.unsqueeze(3) + node_embedding[None, :, None],
            weight,
            bias,
            backend="einsum",
        )
    if backend != "folded_bmm":
        raise ValueError("dense node backend must be 'einsum' or 'folded_bmm'")
    folded_weight, offset = fold_cell_node_parameters(
        node_embedding, weight, bias
    )
    projected = cell_linear(
        base,
        folded_weight,
        backend="grouped_bmm",
        pad_to_multiple=pad_to_multiple,
    )
    return projected.view(
        base.shape[0],
        base.shape[1],
        base.shape[2],
        weight.shape[1],
        weight.shape[2],
    ) + offset[None, :, None]


def _validate_monarch_factors(
    monarch_left: torch.Tensor, monarch_right: torch.Tensor
) -> tuple[int, int, int, int]:
    if monarch_left.ndim != 5 or monarch_right.ndim != 5:
        raise ValueError(
            "monarch_left/monarch_right must have [C,N,q,p,p] and "
            "[C,N,p,q,q] shapes"
        )
    cells, nodes, q, p, p2 = monarch_left.shape
    if p != p2:
        raise ValueError("monarch_left must have square [C,N,q,p,p] blocks")
    cells_r, nodes_r, p_r, q_r, q_r2 = monarch_right.shape
    if q_r != q_r2:
        raise ValueError("monarch_right must have square [C,N,p,q,q] blocks")
    if (cells_r, nodes_r, p_r, q_r) != (cells, nodes, p, q):
        raise ValueError(
            "monarch_left [C,N,q,p,p] and monarch_right [C,N,p,q,q] must "
            "share the same C, N, p, q"
        )
    return cells, nodes, p, q


def monarch_node_linear(
    base: torch.Tensor,
    monarch_left: torch.Tensor,
    monarch_right: torch.Tensor,
) -> torch.Tensor:
    """Apply per-(cell,node) Monarch-factorized weights to ``[B,C,T,I]``
    inputs shared across all nodes, producing ``[B,C,T,N,I]`` -- the same
    output contract as ``folded_cell_node_linear`` (Monarch is a square
    factorization, so output features equal input features).

    Replaces a dense ``[C,N,I,I]`` per-(cell,node) matrix with a product
    of two block-diagonal factors: ``monarch_left: [C,N,q,p,p]`` (q
    independent p*p blocks) and ``monarch_right: [C,N,p,q,q]`` (p
    independent q*q blocks), where ``p*q == I``. This is the standard
    2-factor Monarch construction (Dao et al., "Monarch: Expressive
    Structured Matrices for Efficient and Accurate Training") -- composing
    two differently-axed block-diagonal maps with a transpose between them
    spans a rich, non-degenerate subspace of I*I matrices at
    ``p*q*(p+q)`` parameters instead of ``I**2``.

    Forward, verified self-consistent by hand (row-major reshape of the
    last axis into ``[p,q]``, apply ``monarch_left`` batched over the q
    axis contracting p, transpose, apply ``monarch_right`` batched over
    the (now last) p axis contracting q, transpose back, flatten -- the
    same row-major convention must be used everywhere or the two
    factors silently act on different implicit bases without erroring;
    see ``fold_monarch_node_parameters`` for the paired offset path
    using this exact same convention):

        X = base.reshape(B,C,T,p,q)
        Y = einsum("bctij,cnjoi->bctnoj", X, monarch_left)      # -> [B,C,T,N,p,q]
        Z = einsum("bctnkm,cnmlk->bctnlm", Y.transpose(-2,-1), monarch_right)  # -> [B,C,T,N,q,p]
        return Z.transpose(-2,-1).reshape(B,C,T,N,p*q)

    Batches over cells AND nodes via einsum broadcasting, not a Python
    loop -- ``base`` has no node axis (shared across nodes) while
    ``monarch_left``/``monarch_right`` do, so the node axis is introduced
    purely by the first einsum's output, not by looping over it.
    """

    if base.ndim != 4:
        raise ValueError("base must have [batch,cells,tokens,input] shape")
    cells, nodes, p, q = _validate_monarch_factors(monarch_left, monarch_right)
    batch, base_cells, tokens, input_features = base.shape
    if base_cells != cells:
        raise ValueError("base cells must match monarch weight cells")
    if input_features != p * q:
        raise ValueError("base input features must equal monarch p*q")

    x = base.reshape(batch, cells, tokens, p, q)
    y = torch.einsum("bctij,cnjoi->bctnoj", x, monarch_left)
    z = torch.einsum("bctnkm,cnmlk->bctnlm", y.transpose(-2, -1), monarch_right)
    return z.transpose(-2, -1).reshape(batch, cells, tokens, nodes, p * q)


def fold_monarch_node_parameters(
    node_embedding: torch.Tensor,
    monarch_left: torch.Tensor,
    monarch_right: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Monarch analogue of ``fold_cell_node_parameters``'s offset term:
    ``monarch_forward(node_embedding[c,n]) + bias[c,n]``, computed once
    (relay-invariant, same as the dense fold) rather than materializing
    a per-relay broadcasted node-embedding input.

    Unlike ``monarch_node_linear``, ``node_embedding`` already carries
    its own ``[C,N]`` axes paired directly against
    ``monarch_left``/``monarch_right``'s ``[C,N]`` (no separate
    batch/token axes, no node-axis broadcast to introduce) -- a
    genuinely different einsum batching shape from the main forward
    above, sharing only the row-major ``[p,q]`` reshape convention.
    """

    if node_embedding.ndim != 3 or bias.ndim != 3:
        raise ValueError("node_embedding/bias must have [C,N,I] shape")
    cells, nodes, p, q = _validate_monarch_factors(monarch_left, monarch_right)
    if node_embedding.shape != (cells, nodes, p * q):
        raise ValueError("node_embedding must match monarch weight cells/nodes/p*q")
    if bias.shape != node_embedding.shape:
        raise ValueError("bias must match node_embedding shape")

    e = node_embedding.reshape(cells, nodes, p, q)
    y = torch.einsum("cnij,cnjoi->cnoj", e, monarch_left)
    z = torch.einsum("cnkm,cnmlk->cnlm", y.transpose(-2, -1), monarch_right)
    transformed = z.transpose(-2, -1).reshape(cells, nodes, p * q)
    return transformed + bias
