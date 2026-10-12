"""Selected steward-and-edge transforms for Gator-Path.

The reference transform bank stores private parameters for every
``(cell, node)`` but only gathers and evaluates the entries that currently
steward packets. Its activation shapes therefore follow the packet/route prefix and
never contain a total-cell axis. The later packed engine should replace the
selected ``[..., D, D]`` weight gather with grouped dispatch; this module is
the readable correctness oracle.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_index(
    value: Tensor,
    *,
    name: str,
    prefix: tuple[int, ...],
    upper_bound: int,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if tuple(value.shape) != prefix:
        raise ValueError(
            f"{name} must have shape {prefix}; got {tuple(value.shape)}"
        )
    # This is a reference/correctness guard.  The later packed engine can omit
    # it after proving that its route decoder always emits legal coordinates.
    torch._assert(
        torch.all((value >= 0) & (value < upper_bound)),
        f"{name} must be in [0, {upper_bound})",
    )


class SelectedPathTransform(nn.Module):
    """Apply only router-selected private path transforms.

    ``payload`` and ``mailbox`` have shape ``[..., D]``. ``address`` has shape
    ``[..., Da]`` and all integer selections have shape ``[...]``.  A route
    beam is represented naturally by including ``K`` in that leading prefix.
    No operation creates ``[..., C, N, D]``.
    """

    def __init__(
        self,
        *,
        num_cells: int,
        num_nodes: int,
        payload_width: int,
        address_width: int,
        num_cell_actions: int,
    ) -> None:
        super().__init__()
        self.num_cells = _positive_int(num_cells, "num_cells")
        self.num_nodes = _positive_int(num_nodes, "num_nodes")
        self.payload_width = _positive_int(
            payload_width, "payload_width"
        )
        self.address_width = _positive_int(
            address_width, "address_width"
        )
        self.num_cell_actions = _positive_int(
            num_cell_actions, "num_cell_actions"
        )

        width = self.payload_width
        self.mailbox_weight = nn.Parameter(torch.empty(width, width))
        self.address_weight = nn.Parameter(
            torch.empty(width, self.address_width)
        )
        self.cell_action_embedding = nn.Parameter(
            torch.empty(self.num_cell_actions, width)
        )
        self.node_edge_embedding = nn.Parameter(
            torch.empty(self.num_nodes, self.num_nodes, width)
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.num_cells,
                self.num_nodes,
                width,
                width,
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(self.num_cells, self.num_nodes, width)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mailbox_weight)
        nn.init.xavier_uniform_(self.address_weight)
        nn.init.normal_(
            self.cell_action_embedding,
            std=1.0 / math.sqrt(self.payload_width),
        )
        nn.init.normal_(
            self.node_edge_embedding,
            std=1.0 / math.sqrt(self.payload_width),
        )
        for cell in range(self.num_cells):
            for node in range(self.num_nodes):
                nn.init.xavier_uniform_(self.weight[cell, node])
        nn.init.zeros_(self.bias)

    def forward(
        self,
        payload: Tensor,
        mailbox: Tensor,
        address: Tensor,
        destination_cell: Tensor,
        source_node: Tensor,
        destination_node: Tensor,
        cell_action: Tensor,
        *,
        steward_cell: Tensor | None = None,
        steward_node: Tensor | None = None,
        validate_indices: bool = True,
    ) -> Tensor:
        """Return one residual delta per selected packet or route branch."""

        if type(validate_indices) is not bool:
            raise TypeError("validate_indices must be a boolean")
        if (steward_cell is None) != (steward_node is None):
            raise ValueError(
                "steward_cell and steward_node must be provided together"
            )
        if not isinstance(payload, Tensor) or payload.ndim < 2:
            raise ValueError("payload must have shape [..., D]")
        prefix = tuple(payload.shape[:-1])
        if payload.shape[-1] != self.payload_width:
            raise ValueError(
                f"payload width must be {self.payload_width}; "
                f"got {payload.shape[-1]}"
            )
        if not isinstance(mailbox, Tensor) or tuple(mailbox.shape) != tuple(
            payload.shape
        ):
            raise ValueError("mailbox must have the same shape as payload")
        expected_address = prefix + (self.address_width,)
        if not isinstance(address, Tensor) or tuple(address.shape) != (
            expected_address
        ):
            raise ValueError(
                f"address must have shape {expected_address}; "
                f"got {getattr(address, 'shape', None)}"
            )

        if validate_indices:
            _validate_index(
                destination_cell,
                name="destination_cell",
                prefix=prefix,
                upper_bound=self.num_cells,
            )
            _validate_index(
                source_node,
                name="source_node",
                prefix=prefix,
                upper_bound=self.num_nodes,
            )
            _validate_index(
                destination_node,
                name="destination_node",
                prefix=prefix,
                upper_bound=self.num_nodes,
            )
            _validate_index(
                cell_action,
                name="cell_action",
                prefix=prefix,
                upper_bound=self.num_cell_actions,
            )
            if steward_cell is not None:
                _validate_index(
                    steward_cell,
                    name="steward_cell",
                    prefix=prefix,
                    upper_bound=self.num_cells,
                )
                _validate_index(
                    steward_node,
                    name="steward_node",
                    prefix=prefix,
                    upper_bound=self.num_nodes,
                )

        base = (
            payload
            + F.linear(mailbox, self.mailbox_weight)
            + F.linear(address, self.address_weight)
            + self.cell_action_embedding[cell_action]
            + self.node_edge_embedding[source_node, destination_node]
        )
        transform_cell = (
            destination_cell if steward_cell is None else steward_cell
        )
        transform_node = (
            destination_node if steward_node is None else steward_node
        )
        selected_weight = self.weight[transform_cell, transform_node]
        selected_bias = self.bias[transform_cell, transform_node]
        return torch.tanh(
            torch.einsum("...oi,...i->...o", selected_weight, base)
            + selected_bias
        )


__all__ = ["SelectedPathTransform"]
