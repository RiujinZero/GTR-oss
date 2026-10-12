"""Tests for Procoim-S Phase 5: surprise-gated fast-weight ticketing."""

from __future__ import annotations

import pytest

from gator.optimizers.procoim_s import ProcoimSConfig
from gator.optimizers.procoim_surprise import FastWeightTicket, TicketController


def _config(**overrides) -> ProcoimSConfig:
    defaults = dict(
        weight_ticket_threshold=0.5,
        minimum_reliability=0.6,
        minimum_confirmations=2,
        maximum_candidate_directions=64,
        fast_weight_window_examples=10,
        fast_weight_max_norm=0.02,
    )
    defaults.update(overrides)
    return ProcoimSConfig(**defaults)


class TestRecordRecurrence:
    def test_independent_signatures_count_separately(self) -> None:
        controller = TicketController(_config())
        assert controller.record_recurrence("a") == 1
        assert controller.record_recurrence("a") == 2
        assert controller.record_recurrence("b") == 1


class TestWeightWriteEligibility:
    def test_v01_formula_is_product_of_three_terms(self) -> None:
        controller = TicketController(_config())
        eligibility = controller.weight_write_eligibility(surprise=0.5, reliability=0.8, recurrence=1.0)
        assert eligibility == pytest.approx(0.4)

    def test_future_utility_term_clamps_negative_to_zero(self) -> None:
        controller = TicketController(_config())
        eligibility = controller.weight_write_eligibility(
            surprise=1.0, reliability=1.0, recurrence=1.0, future_utility=-5.0
        )
        assert eligibility == pytest.approx(0.0)

    def test_future_utility_term_scales_positive_values(self) -> None:
        controller = TicketController(_config())
        eligibility = controller.weight_write_eligibility(
            surprise=1.0, reliability=1.0, recurrence=1.0, future_utility=2.0
        )
        assert eligibility == pytest.approx(2.0)


class TestMaybeOpen:
    def test_low_reliability_never_opens_a_ticket(self) -> None:
        controller = TicketController(_config(minimum_reliability=0.9))
        for _ in range(5):
            ticket = controller.maybe_open(
                source_example_id=1,
                surprise=1.0,
                reliability=0.5,
                signature="pattern-a",
                eligible_groups=("routing",),
                opened_step=0,
            )
        assert ticket is None

    def test_a_single_occurrence_never_opens_a_ticket(self) -> None:
        """Design spec section 7.4: one-off surprise, however reliable,
        stays memory-only -- a ticket needs recurrence."""

        controller = TicketController(_config(minimum_confirmations=2))
        ticket = controller.maybe_open(
            source_example_id=1,
            surprise=1.0,
            reliability=1.0,
            signature="pattern-a",
            eligible_groups=("routing",),
            opened_step=0,
        )
        assert ticket is None

    def test_repeated_high_surprise_high_reliability_opens_a_ticket(self) -> None:
        controller = TicketController(_config(minimum_confirmations=2, weight_ticket_threshold=0.5))
        controller.maybe_open(
            source_example_id=1,
            surprise=1.0,
            reliability=1.0,
            signature="pattern-a",
            eligible_groups=("routing",),
            opened_step=0,
        )
        ticket = controller.maybe_open(
            source_example_id=2,
            surprise=1.0,
            reliability=1.0,
            signature="pattern-a",
            eligible_groups=("routing", "stream_mix"),
            opened_step=1,
        )
        assert isinstance(ticket, FastWeightTicket)
        assert ticket.eligible_groups == ("routing", "stream_mix")
        assert ticket.recurrence == pytest.approx(1.0)
        assert ticket.opened_step == 1
        assert ticket.expires_step == 11

    def test_low_eligibility_below_threshold_does_not_open(self) -> None:
        controller = TicketController(
            _config(minimum_confirmations=1, weight_ticket_threshold=0.99, minimum_reliability=0.0)
        )
        ticket = controller.maybe_open(
            source_example_id=1,
            surprise=0.1,
            reliability=0.1,
            signature="pattern-a",
            eligible_groups=("routing",),
            opened_step=0,
        )
        assert ticket is None

    def test_default_forward_budget_derives_from_maximum_candidate_directions(self) -> None:
        controller = TicketController(
            _config(minimum_confirmations=1, weight_ticket_threshold=0.0, maximum_candidate_directions=128)
        )
        ticket = controller.maybe_open(
            source_example_id=1,
            surprise=1.0,
            reliability=1.0,
            signature="pattern-a",
            eligible_groups=("routing",),
            opened_step=0,
        )
        assert ticket.max_forward_budget == 256

    def test_explicit_forward_budget_override(self) -> None:
        controller = TicketController(_config(minimum_confirmations=1, weight_ticket_threshold=0.0))
        ticket = controller.maybe_open(
            source_example_id=1,
            surprise=1.0,
            reliability=1.0,
            signature="pattern-a",
            eligible_groups=("routing",),
            opened_step=0,
            max_forward_budget=999,
        )
        assert ticket.max_forward_budget == 999

    def test_event_ids_increment_across_tickets(self) -> None:
        controller = TicketController(_config(minimum_confirmations=1, weight_ticket_threshold=0.0))
        first = controller.maybe_open(
            source_example_id=1, surprise=1.0, reliability=1.0, signature="a",
            eligible_groups=("routing",), opened_step=0,
        )
        second = controller.maybe_open(
            source_example_id=2, surprise=1.0, reliability=1.0, signature="b",
            eligible_groups=("routing",), opened_step=0,
        )
        assert second.event_id == first.event_id + 1

    def test_max_delta_norm_comes_from_config(self) -> None:
        controller = TicketController(
            _config(minimum_confirmations=1, weight_ticket_threshold=0.0, fast_weight_max_norm=0.07)
        )
        ticket = controller.maybe_open(
            source_example_id=1, surprise=1.0, reliability=1.0, signature="a",
            eligible_groups=("routing",), opened_step=0,
        )
        assert ticket.max_delta_norm == pytest.approx(0.07)


class TestFastWeightTicketExpiry:
    def test_boundary_is_inclusive(self) -> None:
        ticket = FastWeightTicket(
            event_id=0,
            source_example_id=0,
            eligible_groups=("routing",),
            surprise=1.0,
            reliability=1.0,
            recurrence=1.0,
            priority=1.0,
            opened_step=0,
            expires_step=10,
            max_forward_budget=64,
            max_delta_norm=0.02,
        )
        assert not ticket.is_expired(9)
        assert ticket.is_expired(10)
