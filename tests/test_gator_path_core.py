from __future__ import annotations

import unittest

import torch

from gator_path.core import GatorPathCore
from gator_path.packet_state import PacketState
from gator_path.relay import GatorPathRelay
from gator_path.topology import fixed_ring_topology


class GatorPathCoreTests(unittest.TestCase):
    def _relay(self, *, cells: int = 7) -> GatorPathRelay:
        torch.manual_seed(31)
        return GatorPathRelay(
            topology=fixed_ring_topology(cells),
            payload_width=4,
            address_width=2,
            top_k=1,
            straight_through_hard=True,
        )

    def _state(
        self,
        *,
        requires_grad: bool = False,
    ) -> PacketState:
        payload = torch.randn(2, 4, 4)
        address = torch.randn(4, 2)
        if requires_grad:
            payload.requires_grad_(True)
            address.requires_grad_(True)
        return PacketState(
            payload=payload,
            address=address,
            cell_id=torch.tensor(
                [[0, 1, 2, 3], [6, 5, 4, 3]],
                dtype=torch.long,
            ),
            node_id=torch.tensor(
                [[0, 1, 2, 3], [3, 2, 1, 0]],
                dtype=torch.long,
            ),
            active=torch.ones(2, 4, dtype=torch.bool),
            write_gate=torch.tensor(
                [[1.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.5]]
            ),
        )

    def test_relay_depth_reuses_exactly_one_parameter_set(self) -> None:
        shallow_relay = self._relay()
        deep_relay = self._relay()
        shallow = GatorPathCore(shallow_relay, relay_steps=1)
        deep = GatorPathCore(deep_relay, relay_steps=9)

        shallow_count = sum(
            parameter.numel() for parameter in shallow.parameters()
        )
        deep_count = sum(
            parameter.numel() for parameter in deep.parameters()
        )

        self.assertEqual(shallow_count, deep_count)
        self.assertIs(deep.relay, deep_relay)
        self.assertEqual(tuple(deep.children()), (deep_relay,))
        self.assertEqual(
            len(list(deep.parameters())),
            len({id(parameter) for parameter in deep.parameters()}),
        )
        self.assertFalse(
            any(
                name.startswith("relays.")
                for name, _ in deep.named_parameters()
            )
        )

    def test_forward_is_functional_and_preserves_input_snapshot(self) -> None:
        core = GatorPathCore(self._relay(), relay_steps=3)
        state = self._state()
        before = {
            name: getattr(state, name).clone()
            for name in (
                "payload",
                "cell_id",
                "node_id",
                "active",
                "write_gate",
            )
        }

        result = core(state)

        self.assertIsNot(result.state, state)
        self.assertIs(result.state.address, state.address)
        for name, expected in before.items():
            torch.testing.assert_close(getattr(state, name), expected)
        self.assertIs(result.state.write_gate, state.write_gate)
        for relay_result in result.relay_results:
            self.assertIs(relay_result.state.write_gate, state.write_gate)

    def test_every_step_stays_compact_and_finite(self) -> None:
        core = GatorPathCore(self._relay(cells=101), relay_steps=5)

        result = core(self._state())

        self.assertEqual(result.relay_steps, 5)
        self.assertIs(result.diagnostics, result.relay_results)
        for relay_result in result.relay_results:
            self.assertEqual(
                tuple(relay_result.state.payload.shape),
                (2, 4, 4),
            )
            self.assertNotIn(
                core.relay.topology.num_cells,
                relay_result.state.payload.shape,
            )
            self.assertTrue(
                torch.isfinite(relay_result.state.payload).all()
            )
            self.assertTrue(
                torch.isfinite(relay_result.route.weights).all()
            )
            self.assertTrue(
                torch.isfinite(
                    relay_result.route.joint_probabilities
                ).all()
            )

    def test_write_gate_schedule_is_committed_before_each_relay(self) -> None:
        relay = self._relay()
        core = GatorPathCore(relay, relay_steps=3)
        state = self._state()
        schedule = torch.tensor(
            [
                [[1.0, 1.0, 0.0, 0.0], [1.0, 0.5, 0.0, 1.0]],
                [[0.5, 0.5, 0.5, 0.5], [0.0, 0.5, 1.0, 1.0]],
                [[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.5, 0.5]],
            ]
        )
        seen: list[torch.Tensor] = []
        seen_validation: list[bool] = []

        def record_gate(
            _module: GatorPathRelay,
            args: tuple[PacketState, ...],
            _kwargs: dict[str, object],
        ) -> None:
            seen.append(args[0].write_gate.detach().clone())
            seen_validation.append(
                bool(_kwargs["validate_coordinates"])
            )

        handle = relay.register_forward_pre_hook(
            record_gate,
            with_kwargs=True,
        )
        try:
            result = core(state, write_gate_schedule=schedule)
        finally:
            handle.remove()

        self.assertEqual(len(seen), 3)
        self.assertEqual(seen_validation, [True, False, False])
        for relay_step in range(3):
            torch.testing.assert_close(seen[relay_step], schedule[relay_step])
            torch.testing.assert_close(
                result.relay_results[relay_step].state.write_gate,
                schedule[relay_step],
            )
        torch.testing.assert_close(result.state.write_gate, schedule[-1])
        torch.testing.assert_close(
            state.write_gate,
            self._state().write_gate,
        )

    def test_multi_relay_backward_is_finite(self) -> None:
        relay = self._relay()
        core = GatorPathCore(relay, relay_steps=4)
        state = self._state(requires_grad=True)

        result = core(
            state,
            straight_through_hard=True,
        )
        loss = result.state.payload.square().mean()
        loss.backward()

        self.assertIsNotNone(state.payload.grad)
        self.assertIsNotNone(state.address.grad)
        self.assertTrue(torch.isfinite(state.payload.grad).all())
        self.assertTrue(torch.isfinite(state.address.grad).all())
        parameter_gradients = [
            parameter.grad
            for parameter in core.parameters()
            if parameter.grad is not None
        ]
        self.assertGreater(len(parameter_gradients), 0)
        for gradient in parameter_gradients:
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNotNone(relay.cell_router_weight.grad)
        self.assertIsNotNone(relay.node_router_weight.grad)
        self.assertGreater(
            float(relay.cell_router_weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(relay.node_router_weight.grad.abs().sum()), 0.0
        )

    def test_write_gate_schedule_validation(self) -> None:
        core = GatorPathCore(self._relay(), relay_steps=3)
        state = self._state()

        with self.assertRaisesRegex(ValueError, r"\[R, B, L\]"):
            core(
                state,
                write_gate_schedule=torch.ones(2, 2, 4),
            )
        with self.assertRaisesRegex(TypeError, "floating-point"):
            core(
                state,
                write_gate_schedule=torch.ones(
                    3, 2, 4, dtype=torch.long
                ),
            )


if __name__ == "__main__":
    unittest.main()
