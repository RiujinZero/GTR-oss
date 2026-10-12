"""Gator-Path experimental packet-routing architecture."""

from .core import CoreResult, GatorPathCore
from .frontier import (
    FrontierRelayResult,
    FrontierResult,
    FrontierState,
    LocalPacketFrontier,
)
from .mailbox import (
    MailboxRead,
    SparseMailbox,
    build_sparse_mailbox,
    encode_location_keys,
)
from .packet_state import PacketState
from .persistent_memory import (
    PersistentRelativeMemoryState,
    build_path_relative_memory,
    build_relative_memory,
    empty_relative_memory,
    update_path_relative_memory,
    update_relative_memory,
)
from .prefix import (
    GatorPathPrefixConfig,
    GatorPathPrefixModel,
    fixed_address_code,
)
from .relay import GatorPathRelay, RelayResult
from .routing import LocalRouteResult, route_local_actions
from .topology import (
    FIXED_RING_CELL_OFFSETS,
    VALIDATED_NODE_ADJACENCY,
    FactorizedAction,
    PathAction,
    PathTopology,
    decode_joint_index,
    encode_joint_indices,
    fixed_ring_topology,
)
from .transforms import SelectedPathTransform

__all__ = [
    "CoreResult",
    "FIXED_RING_CELL_OFFSETS",
    "VALIDATED_NODE_ADJACENCY",
    "FactorizedAction",
    "FrontierRelayResult",
    "FrontierResult",
    "FrontierState",
    "GatorPathCore",
    "GatorPathPrefixConfig",
    "GatorPathPrefixModel",
    "GatorPathRelay",
    "LocalRouteResult",
    "LocalPacketFrontier",
    "MailboxRead",
    "PacketState",
    "PersistentRelativeMemoryState",
    "PathAction",
    "PathTopology",
    "RelayResult",
    "SelectedPathTransform",
    "SparseMailbox",
    "build_sparse_mailbox",
    "build_path_relative_memory",
    "decode_joint_index",
    "encode_location_keys",
    "encode_joint_indices",
    "fixed_ring_topology",
    "fixed_address_code",
    "route_local_actions",
    "build_relative_memory",
    "empty_relative_memory",
    "update_path_relative_memory",
    "update_relative_memory",
]
