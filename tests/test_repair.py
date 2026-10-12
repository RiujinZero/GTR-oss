"""Tests for Procoim-S Phase 7: the repair queue."""

from __future__ import annotations

import pytest

from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.repair import RepairQueue, repair_utility
from gator.optimizers.retention_monitor import ForgettingEvent, ForgettingSeverity


def _config(**overrides) -> ProcoimSConfig:
    defaults = dict(repair_queue_capacity=3, repair_budget_per_consolidation=2)
    defaults.update(overrides)
    return ProcoimSConfig(**defaults)


def _event(item_id: str, severity: ForgettingSeverity) -> ForgettingEvent:
    return ForgettingEvent(item_id=item_id, forgetting_surprise=5.0, severity=severity)


class TestEnqueue:
    def test_none_severity_does_not_enqueue(self) -> None:
        queue = RepairQueue(_config())
        ticket = queue.enqueue(_event("a", ForgettingSeverity.NONE), target_groups=("routing",), opened_step=0)
        assert ticket is None
        assert len(queue) == 0

    def test_warning_severity_does_not_enqueue(self) -> None:
        queue = RepairQueue(_config())
        ticket = queue.enqueue(_event("a", ForgettingSeverity.WARNING), target_groups=("routing",), opened_step=0)
        assert ticket is None
        assert len(queue) == 0

    def test_repair_severity_enqueues(self) -> None:
        queue = RepairQueue(_config())
        ticket = queue.enqueue(_event("a", ForgettingSeverity.REPAIR), target_groups=("routing",), opened_step=5)
        assert ticket is not None
        assert ticket.item_id == "a"
        assert ticket.opened_step == 5
        assert len(queue) == 1

    def test_critical_severity_enqueues(self) -> None:
        queue = RepairQueue(_config())
        ticket = queue.enqueue(_event("a", ForgettingSeverity.CRITICAL), target_groups=("routing",), opened_step=0)
        assert ticket is not None
        assert len(queue) == 1

    def test_capacity_bound_drops_oldest(self) -> None:
        queue = RepairQueue(_config(repair_queue_capacity=2))
        queue.enqueue(_event("a", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        queue.enqueue(_event("b", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        queue.enqueue(_event("c", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        assert len(queue) == 2
        remaining_ids = {ticket.item_id for ticket in queue.pop_budget(2)}
        assert remaining_ids == {"b", "c"}


class TestPopBudget:
    def test_default_budget_comes_from_config(self) -> None:
        queue = RepairQueue(_config(repair_queue_capacity=5, repair_budget_per_consolidation=1))
        queue.enqueue(_event("a", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        queue.enqueue(_event("b", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        popped = queue.pop_budget()
        assert len(popped) == 1
        assert len(queue) == 1

    def test_critical_tickets_pop_before_repair_tickets(self) -> None:
        queue = RepairQueue(_config(repair_queue_capacity=5))
        queue.enqueue(_event("a", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        queue.enqueue(_event("b", ForgettingSeverity.CRITICAL), target_groups=(), opened_step=0)
        popped = queue.pop_budget(1)
        assert popped[0].item_id == "b"

    def test_popped_tickets_are_removed_from_the_queue(self) -> None:
        queue = RepairQueue(_config())
        queue.enqueue(_event("a", ForgettingSeverity.REPAIR), target_groups=(), opened_step=0)
        queue.pop_budget(1)
        assert len(queue) == 0

    def test_rejects_negative_budget(self) -> None:
        queue = RepairQueue(_config())
        with pytest.raises(ValueError):
            queue.pop_budget(-1)


class TestRepairUtility:
    def test_matches_manual_formula(self) -> None:
        utility = repair_utility(
            gain_forgotten=2.0, current_damage=0.5, other_replay_regression=0.25,
            damage_weight=1.0, replay_weight=2.0,
        )
        assert utility == pytest.approx(2.0 - 0.5 - 0.5)

    def test_large_current_damage_can_make_utility_negative(self) -> None:
        utility = repair_utility(gain_forgotten=0.1, current_damage=5.0, other_replay_regression=0.0)
        assert utility < 0
