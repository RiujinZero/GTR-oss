from __future__ import annotations

import unittest

import torch

from gator_linear import (
    GatorLinearConfig,
    GatorLinearCore,
    GatorLinearPrefixConfig,
    GatorLinearPrefixModel,
)
from gator_linear import coalition as core_coalition
from gator_linear import core as core_module


def small_config(**overrides) -> GatorLinearConfig:
    settings = {
        "vocab_size": 16,
        "streams": 6,
        "nodes": 20,
        "width": 8,
        "relays": 3,
        "degree": 4,
        "buses": 3,
        "mailbox_channels": 3,
        "mailbox_width": 6,
        "basis_rank": 4,
        "address_width": 4,
    }
    settings.update(overrides)
    return GatorLinearConfig(**settings)


def build_model(**overrides) -> GatorLinearPrefixModel:
    torch.manual_seed(0)
    return GatorLinearPrefixModel(
        GatorLinearPrefixConfig(core=small_config(**overrides))
    )


class TransportConservesMassTest(unittest.TestCase):
    def test_occupancy_and_edges_remain_distributions(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (4, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        self.assertEqual(len(diagnostics["relays"]), 3)
        for relay, record in enumerate(diagnostics["relays"]):
            occupancy = record["occupancy"].sum(-1)
            torch.testing.assert_close(
                occupancy,
                torch.ones_like(occupancy),
                msg=f"relay {relay} lost route mass",
            )
            edges = record["edge_probability"].sum(-1)
            torch.testing.assert_close(edges, torch.ones_like(edges))


class BroadButStructuredSupportTest(unittest.TestCase):
    """Occupancy must cover every node without becoming uniform.

    Both directions are failure modes: a starved node receives no gradient,
    but a perfectly flat distribution means every node holds identical
    content and the graph cannot address anything.
    """

    def test_every_node_receives_meaningful_mass(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        hidden, _ = model.initial_hidden(observed, output_slots=5)
        occupancy = model.core.initial_occupancy(hidden)
        per_node = occupancy.mean(dim=(0, 1, 2))
        uniform = 1.0 / model.config.core.nodes
        self.assertGreater(float(per_node.min()), 0.1 * uniform)

    def test_occupancy_is_not_degenerately_uniform(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        hidden, _ = model.initial_hidden(observed, output_slots=5)
        occupancy = model.core.initial_occupancy(hidden)
        entropy = -(
            occupancy.clamp_min(1e-9).log() * occupancy
        ).sum(-1).mean()
        maximum = torch.tensor(
            float(model.config.core.nodes)
        ).log()
        self.assertLess(float(entropy), float(maximum) - 0.2)

    def test_slot_profiles_differ_so_addressing_is_possible(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        hidden, _ = model.initial_hidden(observed, output_slots=5)
        occupancy = model.core.initial_occupancy(hidden)
        per_slot = occupancy.mean(dim=(0, 2))
        self.assertGreater(float(torch.cdist(per_slot, per_slot).mean()), 0.01)


class EvidenceReachesOutputSlotsTest(unittest.TestCase):
    """Regression test for the silent mailbox-attenuation bug.

    Dropping the numerator/denominator normalisation in the mailbox left the
    context at ~S/N of the residual stream, so after a few relays an output
    slot could not see the observed tokens at all. Accuracy sat at chance and
    looked exactly like a tuning problem; only a causal probe distinguished
    them.
    """

    def test_perturbing_an_observed_token_changes_the_output(self) -> None:
        model = build_model().eval()
        observed = torch.randint(0, 12, (64, 6)) + 4
        with torch.no_grad():
            base = model(observed, output_slots=5).log_softmax(-1)
        for index in range(observed.shape[1]):
            perturbed = observed.clone()
            perturbed[:, index] = (perturbed[:, index] - 4 + 6) % 12 + 4
            with torch.no_grad():
                alternative = model(
                    perturbed, output_slots=5
                ).log_softmax(-1)
            delta = float((alternative - base).abs().mean())
            self.assertGreater(
                delta,
                1e-3,
                msg=f"observed slot {index} does not reach the outputs",
            )

    def test_gradient_reaches_every_observed_slot(self) -> None:
        model = build_model()
        observed = torch.randint(0, 12, (16, 6)) + 4
        hidden, addresses = model.initial_hidden(observed, output_slots=5)
        hidden.retain_grad()
        final, _, _ = model.core(hidden, addresses)
        final.sum().backward()
        per_slot = hidden.grad.abs().mean(dim=(0, 2, 3))
        self.assertGreater(float(per_slot.min()), 0.0)


class NoDeadUnitsTest(unittest.TestCase):
    def test_every_parameter_receives_gradient(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5).log_softmax(-1).mean().backward()
        starved = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                parameter.grad is None
                or float(parameter.grad.abs().sum()) == 0.0
            )
        ]
        self.assertEqual(starved, [])

    def test_no_stream_is_born_dead(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        weights = diagnostics["fusion_weights"].mean(dim=(0, 1))
        self.assertGreater(float(weights.min()), 0.01)


class ScanNormalisationTest(unittest.TestCase):
    """Normalising the scans converts running totals into running means.

    With a constant input a true prefix sum at slot l equals l+1. The
    normalised scan returns exactly 1.0 at every slot -- it carries no
    information about how much has accumulated, which is fatal for any task
    built on accumulation.
    """

    def test_normalised_scan_destroys_the_running_count(self) -> None:
        length = 12
        values = torch.ones(1, length, 1)
        decay = torch.tensor([0.98])
        raw = core_module.decayed_scan(values, decay, chunk_size=64)[0, :, 0]
        steps = torch.arange(1, length + 1, dtype=torch.float32)
        normaliser = (1 - decay[0] ** steps) / (1 - decay[0])
        normalised = raw / normaliser
        # Normalised is flat: slot 0 and slot 11 are indistinguishable.
        self.assertAlmostEqual(
            float(normalised[0]), float(normalised[-1]), places=4
        )
        # Raw grows with the accumulated count.
        self.assertGreater(float(raw[-1]), 5.0 * float(raw[0]))

    def test_flag_controls_normalisation(self) -> None:
        values = torch.randn(2, 9, 4, 3, 6)
        outputs = {}
        for normalise in (True, False):
            torch.manual_seed(0)
            mixer = core_module.LinearSequenceMixer(
                small_config(mailbox_channels=3, mailbox_width=6,
                             scan_normalise=normalise)
            )
            outputs[normalise] = mixer(values)
        self.assertFalse(
            torch.allclose(outputs[True], outputs[False], atol=1e-5)
        )


class HardRoutingTest(unittest.TestCase):
    """Straight-through commit: the reference mechanism GTR-L was missing.

    The reference's forward pass is a genuine discrete route at every relay
    ("no route beam survives this relay"); GTR-L blended everywhere, which
    made "average toward the marginal" a real, cheap forward-pass output.
    Hard commit removes that point from the range: forward is a one-hot
    argmax, gradient still flows through the full soft distribution.
    """

    def test_straight_through_forward_value_is_exactly_one_hot(self) -> None:
        distribution = torch.softmax(torch.randn(4, 7, 10), dim=-1)
        committed = core_module.straight_through_commit(distribution)
        expected = torch.zeros_like(distribution).scatter_(
            -1, distribution.argmax(dim=-1, keepdim=True), 1.0
        )
        torch.testing.assert_close(committed, expected)
        row_sums = committed.sum(-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    def test_gradient_reaches_every_node_not_just_the_argmax(self) -> None:
        # A plain .sum() is invariant to a softmax row's own normalisation
        # (it always sums to one), which trivially zeroes this gradient
        # regardless of the mechanism under test. Weight each node
        # differently so the loss is actually sensitive to the distribution.
        logits = torch.randn(4, 10, requires_grad=True)
        distribution = torch.softmax(logits, dim=-1)
        committed = core_module.straight_through_commit(distribution)
        weights = torch.arange(10, dtype=torch.float32)
        (committed * weights).sum().backward()
        self.assertTrue((logits.grad.abs() > 0).all())

    def test_model_runs_and_conserves_mass_with_hard_routing(self) -> None:
        model = build_model(hard_routing=True)
        observed = torch.randint(0, 16, (8, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        for record in diagnostics["relays"]:
            occupancy = record["occupancy"]
            row_sums = occupancy.sum(-1)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums)
            )
            # Hard routing: every row is one-hot (within float32 tolerance;
            # the straight-through (a+b)-c evaluation order can leave a
            # ~1e-7 residual on values already near 1.0).
            is_binary = torch.isclose(
                occupancy, torch.zeros_like(occupancy), atol=1e-4
            ) | torch.isclose(occupancy, torch.ones_like(occupancy), atol=1e-4)
            self.assertTrue(bool(is_binary.all()))

    def test_hard_routing_still_trains_without_dead_parameters(self) -> None:
        model = build_model(hard_routing=True)
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5).log_softmax(-1).mean().backward()
        starved = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                parameter.grad is None
                or float(parameter.grad.abs().sum()) == 0.0
            )
        ]
        self.assertEqual(starved, [])


class RouterOccupancyTest(unittest.TestCase):
    """Router occupancy input: the other reference mechanism GTR-L lacked.

    Both reference cells condition their routing decision on current
    per-node load (gtr_a_experiments.py's GTRFixedSlots feeds the router
    ``occupancy = q.mean(1, keepdim=True).expand_as(q)``; gator_path's
    LocalPacketFrontier feeds a per-location load scalar). GTR-L's router
    saw only hidden/context/address. This is opt-in (``router_occupancy``)
    so existing configs are unaffected.
    """

    def test_default_is_unaffected(self) -> None:
        model = build_model()
        self.assertFalse(model.config.core.router_occupancy)

    def test_model_runs_and_conserves_mass_with_router_occupancy(self) -> None:
        model = build_model(router_occupancy=True)
        observed = torch.randint(0, 16, (8, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        for record in diagnostics["relays"]:
            occupancy = record["occupancy"]
            row_sums = occupancy.sum(-1)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    def test_gradient_reaches_every_parameter_with_router_occupancy(self) -> None:
        model = build_model(router_occupancy=True)
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5).log_softmax(-1).mean().backward()
        starved = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                parameter.grad is None
                or float(parameter.grad.abs().sum()) == 0.0
            )
        ]
        self.assertEqual(starved, [])

    def test_router_query_input_width_grows_by_node_count(self) -> None:
        off = build_model(router_occupancy=False)
        on = build_model(router_occupancy=True)
        relay_off = off.core.relays[0]
        relay_on = on.core.relays[0]
        nodes = off.config.core.nodes
        self.assertEqual(
            relay_on.router_query.in_features,
            relay_off.router_query.in_features + nodes,
        )

    def test_occupancy_feature_changes_the_route(self) -> None:
        # Two different incoming occupancy tensors, same hidden/context/
        # address, must produce different edge routing once the router can
        # see load -- otherwise the new input term is dead weight.
        torch.manual_seed(0)
        relay = core_module.LinearRelay(small_config(router_occupancy=True))
        batch, length, streams = 2, 5, 6
        nodes = relay.config.nodes
        hidden = torch.randn(batch, length, streams, relay.config.width)
        addresses = torch.randn(length, relay.config.address_width)
        edge_destinations = torch.randint(0, nodes, (nodes, relay.config.degree))

        occ_a = torch.softmax(torch.randn(batch, length, streams, nodes), dim=-1)
        occ_b = torch.softmax(torch.randn(batch, length, streams, nodes), dim=-1)

        with torch.no_grad():
            _, next_a, _ = relay(hidden, occ_a, addresses, edge_destinations)
            _, next_b, _ = relay(hidden, occ_b, addresses, edge_destinations)
        self.assertFalse(torch.allclose(next_a, next_b))


class ControlledWritingTest(unittest.TestCase):
    """Controlled-writing schedule: the third reference mechanism gap.

    gator_path/prefix.py's write_gate_schedule silences a slot's
    contribution to the shared mailbox early (output slots start at gate=0,
    since they are still blank queries) and ramps to full by the final
    relay, once the slot has had a chance to read a real answer. GTR-L's
    mailbox write had no such schedule -- every slot, including blank
    output queries, wrote at full strength from relay 0. Opt-in via
    ``controlled_writing`` so existing configs are unaffected.
    """

    def test_default_is_enabled(self) -> None:
        # Validated on the length=3 prefix wall: 4/4 seeds escaped with this
        # on vs 2/4 with it off, rescuing both seeds that otherwise walled.
        model = build_model()
        self.assertTrue(model.config.core.controlled_writing)

    def test_schedule_shape_and_endpoints(self) -> None:
        model = build_model(controlled_writing=True)
        schedule = model.write_gate_schedule(
            batch_size=2, observed_length=3, output_slots=4,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertEqual(tuple(schedule.shape), (3, 2, 7))
        # Observed slots always write at full strength.
        torch.testing.assert_close(
            schedule[:, :, :3], torch.ones_like(schedule[:, :, :3])
        )
        # Output slots ramp from 0 at relay 0 to 1 at the final relay.
        torch.testing.assert_close(
            schedule[0, :, 3:], torch.zeros_like(schedule[0, :, 3:])
        )
        torch.testing.assert_close(
            schedule[-1, :, 3:], torch.ones_like(schedule[-1, :, 3:])
        )

    def test_model_runs_and_conserves_mass_with_controlled_writing(self) -> None:
        model = build_model(controlled_writing=True)
        observed = torch.randint(0, 16, (8, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        for record in diagnostics["relays"]:
            occupancy = record["occupancy"]
            row_sums = occupancy.sum(-1)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    def test_gradient_reaches_every_parameter_with_controlled_writing(self) -> None:
        model = build_model(controlled_writing=True)
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5).log_softmax(-1).mean().backward()
        starved = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                parameter.grad is None
                or float(parameter.grad.abs().sum()) == 0.0
            )
        ]
        self.assertEqual(starved, [])

    def test_relay_zero_write_gate_zeroes_out_that_slots_contribution(self) -> None:
        # Direct relay-level check: gating a slot's write to zero must make
        # its message invisible to the mailbox that relay, without touching
        # its own next hidden state or occupancy computation.
        torch.manual_seed(0)
        config = small_config()
        relay = core_module.LinearRelay(config)
        batch, length, streams = 2, 4, config.streams
        nodes = config.nodes
        hidden = torch.randn(batch, length, streams, config.width)
        occupancy = torch.softmax(
            torch.randn(batch, length, streams, nodes), dim=-1
        )
        addresses = torch.randn(length, config.address_width)
        edge_destinations = torch.randint(0, nodes, (nodes, config.degree))

        full_gate = torch.ones(batch, length, 1, 1)
        zero_gate = full_gate.clone()
        zero_gate[:, -1] = 0.0

        with torch.no_grad():
            _, occ_full, _ = relay(
                hidden, occupancy, addresses, edge_destinations,
                write_gate=full_gate,
            )
            _, occ_zeroed, _ = relay(
                hidden, occupancy, addresses, edge_destinations,
                write_gate=zero_gate,
            )
        self.assertFalse(torch.allclose(occ_full, occ_zeroed))


class LinearAttentionTest(unittest.TestCase):
    """Content-addressed mailbox read via a decayed key(x)value scan.

    The reference's mailbox read is relative-position attention: a slot
    picks out *which* earlier packet matters to it, not just a fixed
    decay-weighted blend of everything. GTR-L's mixer gives every position
    the same content-independent combination of decay channels. This closes
    that gap the linear-attention/RetNet way -- a decayed running key(x)
    value outer product read out with a content query -- so cost stays
    O(L) per node (same order as the existing occupancy-weighted read), no
    [L, L] or [N, N] tensor anywhere. Opt-in via ``linear_attention``.
    """

    def test_default_is_disabled(self) -> None:
        model = build_model()
        self.assertFalse(model.config.core.linear_attention)

    def test_model_runs_and_conserves_mass(self) -> None:
        model = build_model(linear_attention=True)
        observed = torch.randint(0, 16, (8, 6))
        _, diagnostics = model(
            observed, output_slots=5, return_diagnostics=True
        )
        for record in diagnostics["relays"]:
            occupancy = record["occupancy"]
            row_sums = occupancy.sum(-1)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

    def test_gradient_reaches_every_parameter(self) -> None:
        model = build_model(linear_attention=True)
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5).log_softmax(-1).mean().backward()
        starved = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                parameter.grad is None
                or float(parameter.grad.abs().sum()) == 0.0
            )
        ]
        self.assertEqual(starved, [])

    def test_content_changes_the_read_not_just_position(self) -> None:
        # Direct relay-level check: two different value sequences at the
        # same positions/occupancy must produce different content reads --
        # otherwise this is just another position-decay channel and the
        # content-addressing claim is false.
        torch.manual_seed(0)
        config = small_config(linear_attention=True)
        relay = core_module.LinearRelay(config)
        batch, length, streams = 2, 5, config.streams
        nodes = config.nodes

        hidden_a = torch.randn(batch, length, streams, config.width)
        hidden_b = torch.randn(batch, length, streams, config.width)
        write_occupancy = torch.softmax(
            torch.randn(batch, length, streams, nodes), dim=-1
        )
        occupancy = torch.softmax(
            torch.randn(batch, length, streams, nodes), dim=-1
        )

        with torch.no_grad():
            content_a = relay._linear_attention(
                hidden_a, write_occupancy, occupancy
            )
            content_b = relay._linear_attention(
                hidden_b, write_occupancy, occupancy
            )
        self.assertFalse(torch.allclose(content_a, content_b))

    def test_content_read_is_only_as_large_as_the_existing_read(self) -> None:
        # Cost sanity check: the content-addressed read must be a per
        # (position, node) or (position, stream, node) contraction, never
        # an [L, L] tensor. Shape-check rather than a timing test, since
        # timing is unreliable in CI.
        torch.manual_seed(0)
        config = small_config(linear_attention=True)
        relay = core_module.LinearRelay(config)
        batch, length, streams = 2, 9, config.streams
        nodes = config.nodes
        hidden = torch.randn(batch, length, streams, config.width)
        write_occupancy = torch.softmax(
            torch.randn(batch, length, streams, nodes), dim=-1
        )
        occupancy = torch.softmax(
            torch.randn(batch, length, streams, nodes), dim=-1
        )
        content = relay._linear_attention(hidden, write_occupancy, occupancy)
        self.assertEqual(tuple(content.shape), (batch, length, streams, config.width))

    def test_empty_nodes_have_zero_content_gradient(self) -> None:
        torch.manual_seed(0)
        config = small_config(linear_attention=True, hard_routing=True)
        relay = core_module.LinearRelay(config)
        batch, length, streams = 2, 5, config.streams
        hidden = torch.randn(
            batch, length, streams, config.width, requires_grad=True
        )
        write_occupancy = torch.zeros(
            batch,
            length,
            streams,
            config.nodes,
            requires_grad=True,
        )
        occupancy = torch.zeros_like(write_occupancy)
        occupancy[..., 0] = 1.0

        content = relay._linear_attention(
            hidden, write_occupancy, occupancy
        )
        content.sum().backward()

        torch.testing.assert_close(
            content,
            relay.attn_out.bias.view(1, 1, -1).expand_as(content),
        )
        self.assertIsNotNone(write_occupancy.grad)
        torch.testing.assert_close(
            write_occupancy.grad,
            torch.zeros_like(write_occupancy.grad),
        )
        self.assertIsNotNone(hidden.grad)
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden.grad))


class CoalitionTest(unittest.TestCase):
    """The readout must be able to express a complementary coalition.

    The reference does not settle on a winning trace: three material lanes
    scored 33% alone, 83% as the best pair, 99% together. Reproducing that
    is an acceptance criterion, so the measurement has to detect
    complementarity rather than merely report vote concentration.
    """

    def test_per_stream_decomposition_matches_the_fused_output(self) -> None:
        model = build_model().eval()
        observed = torch.randint(0, 16, (16, 6))
        with torch.no_grad():
            fused = model(observed, output_slots=5)
            per_stream, weights = model.stream_logits(
                observed, output_slots=5
            )
        slot_weights = weights[:, -5:]
        slot_weights = slot_weights / slot_weights.sum(-1, keepdim=True)
        recombined = (per_stream * slot_weights.unsqueeze(-1)).sum(2)
        torch.testing.assert_close(fused, recombined, atol=1e-4, rtol=1e-4)

    def test_detects_complementary_streams(self) -> None:
        # Three streams, each correct on a disjoint third of the slots and
        # uninformative elsewhere. No single stream solves the task; together
        # they do. This is the pattern the reference exhibits.
        batch, slots, streams, vocab = 8, 9, 3, 5
        targets = torch.randint(0, vocab, (batch, slots))
        # Off its own slots a stream votes for a wrong class, and a
        # *different* wrong class per stream. Abstaining with zeros would let
        # argmax score accidental hits and inflate the single-stream
        # baseline; agreeing on one wrong class would let the errors reinforce
        # into a tie. Decorrelated errors are also the actual mechanism --
        # the reference's three 33% lanes reach 99% together precisely
        # because they fail differently.
        correct = torch.nn.functional.one_hot(targets, vocab).float() * 20.0
        logits = torch.zeros(batch, slots, streams, vocab)
        for stream in range(streams):
            wrong_class = (targets + 1 + stream) % vocab
            logits[:, :, stream] = torch.nn.functional.one_hot(
                wrong_class, vocab
            ).float() * 10.0
            owned = torch.nonzero(
                torch.arange(slots) % streams == stream
            ).flatten().tolist()
            for index in owned:
                logits[:, index, stream] = correct[:, index]
        weights = torch.full((batch, slots, streams), 1.0 / streams)
        report = core_coalition.coalition_report(logits, weights, targets)
        self.assertGreater(report["full_accuracy"], 0.95)
        self.assertLess(report["best_by_size"][1]["accuracy"], 0.5)
        self.assertGreater(report["complementarity_gain"], 0.5)

    def test_identical_streams_report_zero_spread(self) -> None:
        # Ten copies of the same voter. Whatever level they score at, zero
        # spread is the collapse signature -- this is the property to watch,
        # not whether any stream reaches a particular accuracy.
        batch, slots, streams, vocab = 8, 9, 4, 5
        targets = torch.randint(0, vocab, (batch, slots))
        single = torch.randn(batch, slots, vocab)
        logits = single.unsqueeze(2).repeat(1, 1, streams, 1)
        weights = torch.full((batch, slots, streams), 1.0 / streams)
        report = core_coalition.coalition_report(logits, weights, targets)
        self.assertAlmostEqual(report["solo_spread"], 0.0, places=6)
        self.assertAlmostEqual(report["complementarity_gain"], 0.0, places=6)

    def test_differentiated_streams_report_nonzero_spread(self) -> None:
        batch, slots, streams, vocab = 8, 9, 4, 5
        targets = torch.randint(0, vocab, (batch, slots))
        logits = torch.randn(batch, slots, streams, vocab)
        # Give each stream a different share of the answers.
        for stream in range(streams):
            share = slots * (stream + 1) // (streams + 1)
            correct = torch.nn.functional.one_hot(targets, vocab).float() * 9.0
            logits[:, :share, stream] = correct[:, :share]
        weights = torch.full((batch, slots, streams), 1.0 / streams)
        report = core_coalition.coalition_report(logits, weights, targets)
        self.assertGreater(report["solo_spread"], 0.05)

    def test_detects_a_single_dominant_solver(self) -> None:
        # One stream solves everything; the others are noise. Complementarity
        # gain should be ~0, distinguishing this from a real coalition.
        batch, slots, streams, vocab = 8, 9, 3, 5
        targets = torch.randint(0, vocab, (batch, slots))
        logits = torch.randn(batch, slots, streams, vocab) * 0.01
        logits[:, :, 0] = torch.nn.functional.one_hot(
            targets, vocab
        ).float() * 10.0
        weights = torch.zeros(batch, slots, streams)
        weights[:, :, 0] = 0.9
        weights[:, :, 1:] = 0.05
        report = core_coalition.coalition_report(logits, weights, targets)
        self.assertGreater(report["best_by_size"][1]["accuracy"], 0.95)
        self.assertLess(abs(report["complementarity_gain"]), 0.05)


class ConvKernelWidthTest(unittest.TestCase):
    """Reach per relay is (k-1)/2, so the kernel sets what the model can see.

    The reference architecture gets full-range reach in a single relay from
    its [L, L] relation kernel, which is why four relays suffice for it.
    Replacing that kernel without widening the convolution leaves the model
    unable to reach the writers a long-range target depends on. Production
    language configs use k=32.
    """

    def test_even_kernel_allowed_only_in_causal_mode(self) -> None:
        # Causal padding is entirely on the left, so even widths are fine.
        core_module.GatorLinearConfig(conv_kernel=32, causal=True)
        with self.assertRaises(ValueError):
            core_module.GatorLinearConfig(conv_kernel=32, causal=False)

    def test_wide_causal_kernel_runs_and_stays_causal(self) -> None:
        torch.manual_seed(0)
        config = small_config(conv_kernel=32, causal=True, context_length=64)
        core = GatorLinearCore(config)
        length = 24
        hidden = torch.randn(2, length, config.streams, config.width)
        addresses = torch.randn(length, config.address_width)
        base, _, _ = core(hidden, addresses)
        changed = hidden.clone()
        changed[:, -1] += 5.0
        altered, _, _ = core(changed, addresses)
        delta = (altered - base).abs().mean(dim=(0, 2, 3))
        self.assertLess(float(delta[:-1].max()), 1e-5)


class HeterogeneousStreamTest(unittest.TestCase):
    """Different capacities make convergence structurally impossible.

    Per-stream adapters leave every stream the same shape, so nothing
    prevents them collapsing onto one function -- measured agreement rose
    from 0.13 to 0.86 during training, fastest with adapters enabled. Streams
    with different widths do not share a function space to collapse into.
    """

    def test_widths_are_distinct_and_span_the_configured_range(self) -> None:
        model = build_model(
            width=32, streams=8, heterogeneous_streams=True,
            stream_width_min_fraction=0.25,
        )
        widths = model.core.stream_widths
        self.assertEqual(len(set(widths)), len(widths))
        self.assertEqual(max(widths), 32)
        self.assertGreaterEqual(min(widths), 8)

    def test_inactive_dimensions_stay_exactly_zero(self) -> None:
        model = build_model(
            width=32, streams=8, heterogeneous_streams=True
        ).eval()
        observed = torch.randint(0, 16, (8, 6))
        hidden, addresses = model.initial_hidden(observed, output_slots=5)
        final, _, _ = model.core(hidden, addresses)
        active = (final.abs().mean(dim=(0, 1)) > 1e-8).sum(-1).tolist()
        self.assertEqual(active, model.core.stream_widths)

    def test_homogeneous_mode_leaves_every_dimension_active(self) -> None:
        model = build_model(width=32, streams=8, heterogeneous_streams=False)
        self.assertEqual(model.core.stream_widths, [32] * 8)

    def test_rejects_a_degenerate_narrowest_stream(self) -> None:
        with self.assertRaises(ValueError):
            core_module.GatorLinearConfig(
                width=4, streams=4, heterogeneous_streams=True,
                stream_width_min_fraction=0.1,
            )


class ScanTimescaleTest(unittest.TestCase):
    """Scan timescales must track the context, not a fixed constant.

    A fixed decay range silently goes blind once the context outgrows it:
    at a 55-slot ceiling, a token 620 slots back is retained at ~4e-6, so
    the decayed scans stop supplying distance structure entirely and only
    the positionless global mean carries long-range information.
    """

    def test_timescales_span_one_to_context(self) -> None:
        for context in (16, 64, 620, 2048):
            mixer = core_module.LinearSequenceMixer(
                small_config(context_length=context)
            )
            decay = torch.sigmoid(mixer.forward_decay.detach())
            tau = -1.0 / torch.log(decay.clamp(1e-9, 1 - 1e-9))
            self.assertLess(float(tau.min()), 2.0)
            self.assertGreater(float(tau.max()), 0.5 * context)

    def test_far_token_retention_is_scale_invariant(self) -> None:
        retentions = []
        for context in (16, 64, 620, 2048):
            mixer = core_module.LinearSequenceMixer(
                small_config(context_length=context)
            )
            decay = torch.sigmoid(mixer.forward_decay.detach())
            retentions.append(float(decay.max() ** context))
        # Every context retains the same fraction at its own far edge.
        self.assertLess(max(retentions) - min(retentions), 0.05)


class RouteLossTest(unittest.TestCase):
    """Sharpening and balance must oppose each other, and both must be live.

    Entropy alone collapses every packet onto a few nodes and starves the
    rest; balance alone permits the degenerate uniform occupancy that makes
    the graph unaddressable. The pair is what allows sharp per-slot routing
    with every node still alive.
    """

    def test_statistics_are_differentiable_and_reported(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5)
        statistics = model.latest_route_statistics
        self.assertIn("route_entropy", statistics)
        self.assertIn("route_balance", statistics)
        for value in statistics.values():
            self.assertTrue(value.requires_grad)

    def test_entropy_is_lower_for_sharper_occupancy(self) -> None:
        sharp = torch.tensor([[[[0.9, 0.05, 0.05]]]])
        flat = torch.full((1, 1, 1, 3), 1.0 / 3.0)
        self.assertLess(
            float(core_module.route_entropy(sharp)),
            float(core_module.route_entropy(flat)),
        )

    def test_balance_is_zero_only_for_uniform_marginal_load(self) -> None:
        uniform = torch.full((2, 4), 0.25)
        self.assertAlmostEqual(
            float(core_module.load_balance(uniform)), 0.0, places=6
        )
        lopsided = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
        self.assertGreater(float(core_module.load_balance(lopsided)), 0.1)

    def test_route_losses_reach_routing_parameters(self) -> None:
        model = build_model()
        observed = torch.randint(0, 16, (8, 6))
        model(observed, output_slots=5)
        statistics = model.latest_route_statistics
        (statistics["route_entropy"] + statistics["route_balance"]).backward()
        self.assertIsNotNone(model.core.ingress_key.grad)
        self.assertGreater(float(model.core.ingress_key.grad.abs().sum()), 0.0)


class LinearScalingTest(unittest.TestCase):
    """No operator may build an [L, L], [N, N], or [S, S] tensor."""

    def test_parameters_grow_linearly_in_node_count(self) -> None:
        counts = {
            nodes: build_model(nodes=nodes).parameter_count
            for nodes in (20, 40, 80)
        }
        # Node-count-dependent parameters must not grow faster than linearly.
        first = counts[40] - counts[20]
        second = counts[80] - counts[40]
        self.assertLess(second, 2.5 * first)

    def test_no_operator_materialises_a_node_by_node_tensor(self) -> None:
        config = small_config(nodes=24)
        core = GatorLinearCore(config)
        observed_slots, output_slots = 5, 4
        length = observed_slots + output_slots
        hidden = torch.randn(2, length, config.streams, config.width)
        addresses = torch.randn(length, config.address_width)

        forbidden: list[tuple[int, ...]] = []
        original = torch.Tensor.__repr__

        def record(tensor: torch.Tensor) -> str:
            return original(tensor)

        # Walk every intermediate via a hook on the autograd graph instead of
        # guessing: assert on the peak element count relative to the linear
        # budget. A single [L, L, N] or [N, N] tensor would blow past it.
        result, _, _ = core(hidden, addresses)
        linear_budget = (
            2 * length * config.streams * config.nodes * config.degree * 4
        )
        largest = 0
        stack = [result.grad_fn]
        seen = set()
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            for saved in getattr(node, "saved_tensors", ()):  # type: ignore[arg-type]
                if isinstance(saved, torch.Tensor):
                    largest = max(largest, saved.numel())
            stack.extend(child for child, _ in node.next_functions)
        self.assertLess(largest, linear_budget, f"forbidden={forbidden}")


class CausalModeTest(unittest.TestCase):
    def test_causal_mixer_cannot_read_future_slots(self) -> None:
        torch.manual_seed(0)
        core = GatorLinearCore(small_config(causal=True))
        length, streams = 8, small_config().streams
        hidden = torch.randn(4, length, streams, 8)
        addresses = torch.randn(length, 4)
        base, _, _ = core(hidden, addresses)

        changed = hidden.clone()
        changed[:, -1] += 5.0
        altered, _, _ = core(changed, addresses)
        # Only the final slot may change when the final slot is perturbed.
        delta = (altered - base).abs().mean(dim=(0, 2, 3))
        self.assertLess(float(delta[:-1].max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
