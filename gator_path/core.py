"""Shared-recurrent multi-relay core for Gator-Path.

This module intentionally stops at persistent packet refinement.  Token
embeddings, formation queries, and simultaneous vocabulary readout belong to
the later task shell.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .packet_state import PacketState
from .relay import GatorPathRelay, RelayResult


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class CoreResult:
    """Final packet state and diagnostics from every recurrent relay."""

    state: PacketState
    relay_results: tuple[RelayResult, ...]

    @property
    def diagnostics(self) -> tuple[RelayResult, ...]:
        """Alias emphasizing that relay results describe each old snapshot."""

        return self.relay_results

    @property
    def relay_steps(self) -> int:
        return len(self.relay_results)


class GatorPathCore(nn.Module):
    """Apply one shared :class:`GatorPathRelay` recurrently.

    ``relay_steps`` changes recurrent compute depth only.  It never creates a
    per-step module list or duplicates relay parameters.
    """

    def __init__(
        self,
        relay: GatorPathRelay,
        *,
        relay_steps: int,
    ) -> None:
        super().__init__()
        if not isinstance(relay, GatorPathRelay):
            raise TypeError("relay must be a GatorPathRelay")
        self.relay = relay
        self.relay_steps = _positive_int(relay_steps, "relay_steps")

    def _validate_write_gate_schedule(
        self,
        schedule: Tensor | None,
        state: PacketState,
    ) -> Tensor | None:
        if schedule is None:
            return None
        if not isinstance(schedule, Tensor):
            raise TypeError(
                "write_gate_schedule must be a torch.Tensor or None"
            )
        expected = (
            self.relay_steps,
            state.batch_size,
            state.slot_count,
        )
        if tuple(schedule.shape) != expected:
            raise ValueError(
                "write_gate_schedule must have shape "
                f"[R, B, L] = {expected}; got {tuple(schedule.shape)}"
            )
        if not schedule.is_floating_point():
            raise TypeError(
                "write_gate_schedule must have a floating-point dtype"
            )
        if schedule.dtype != state.write_gate.dtype:
            raise TypeError(
                "write_gate_schedule and state.write_gate must have "
                "the same dtype"
            )
        if schedule.device != state.write_gate.device:
            raise ValueError(
                "write_gate_schedule and state.write_gate must be on "
                "the same device"
            )
        return schedule

    def forward(
        self,
        state: PacketState,
        *,
        write_gate_schedule: Tensor | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        straight_through_hard: bool | None = None,
        cell_access_mask: Tensor | None = None,
        cell_destination_table: Tensor | None = None,
        retain_diagnostics: bool = True,
    ) -> CoreResult:
        """Refine a packet snapshot through shared recurrent relay depth.

        When supplied, row ``r`` of ``write_gate_schedule`` is committed to
        the packet state before relay ``r`` reads its mailbox.  Without a
        schedule, the existing gate tensor is preserved across every step.
        """

        if not isinstance(state, PacketState):
            raise TypeError("state must be a PacketState")
        schedule = self._validate_write_gate_schedule(
            write_gate_schedule,
            state,
        )
        if cell_access_mask is not None:
            if not isinstance(cell_access_mask, Tensor):
                raise TypeError(
                    "cell_access_mask must be a torch.Tensor or None"
                )
            if cell_access_mask.ndim not in (2, 3):
                raise ValueError(
                    "cell_access_mask must have shape [C, C] or [R, C, C]"
                )
            if (
                cell_access_mask.ndim == 3
                and cell_access_mask.shape[0] != self.relay_steps
            ):
                raise ValueError(
                    "relay-phased cell_access_mask must have one row "
                    "per relay step"
                )

        current = state
        relay_results: list[RelayResult] = []
        for relay_step in range(self.relay_steps):
            if schedule is not None:
                current = current.replace(
                    write_gate=schedule[relay_step]
                )
            result = self.relay(
                current,
                top_k=top_k,
                temperature=temperature,
                straight_through_hard=straight_through_hard,
                cell_access_mask=(
                    cell_access_mask[relay_step]
                    if cell_access_mask is not None
                    and cell_access_mask.ndim == 3
                    else cell_access_mask
                ),
                cell_destination_table=cell_destination_table,
                # The external snapshot is checked once.  Every later
                # location was emitted by the validated relay decoder.
                validate_coordinates=(relay_step == 0),
            )
            if retain_diagnostics:
                relay_results.append(result)
            current = result.state

        return CoreResult(
            state=current,
            relay_results=tuple(relay_results),
        )


__all__ = ["CoreResult", "GatorPathCore"]
