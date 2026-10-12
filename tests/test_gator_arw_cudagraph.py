import unittest

import torch
import torch.nn.functional as F

from gator.compute import (
    CompileConfig,
    compile_module,
    mark_compiled_iteration_begin,
)
from gator.models.language import GatorLanguageConfig, GatorLanguageModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class GatorARWCUDAGraphTests(unittest.TestCase):
    @staticmethod
    def _model() -> GatorLanguageModel:
        return GatorLanguageModel(
            GatorLanguageConfig(
                vocab_size=32,
                d_model=4,
                embedding_dim=8,
                lexical_embedding_dim=4,
                n_cells=3,
                n_nodes=2,
                relays=1,
                neighbors=2,
                mask_token_id=3,
                lexical_mode="private_hash",
                cell_input_mode="shared_projection",
                cell_output_mode="shared_pool",
                cell_transport_mode="broadcast_mix",
                stewardship_execution="packed_auto",
                receiver_memory_backend="linear_scan",
                block_causal=True,
                block_size=2,
                position_reference_length=16,
                node_topology="symmetric",
                dense_kernel_padding=8,
            )
        ).cuda().train()

    @staticmethod
    def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator().manual_seed(314159)
        return [
            (
                torch.randint(4, 32, (1, 2), generator=generator).cuda(),
                torch.randint(4, 32, (1, 2), generator=generator).cuda(),
            )
            for _ in range(3)
        ]

    @staticmethod
    def _accumulate(
        model: GatorLanguageModel,
        invocation,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        compile_config: CompileConfig,
    ) -> list[dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        accumulated_gradients = []
        for context, targets in batches:
            mask = torch.ones_like(context, dtype=torch.bool)
            mark_compiled_iteration_begin(compile_config, context.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = invocation(context, mask, output_slots=2)
                loss = F.cross_entropy(
                    logits.transpose(1, 2),
                    targets,
                ) / len(batches)
            loss.backward()
            accumulated_gradients.append(
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )
        torch.cuda.synchronize()
        return accumulated_gradients

    def test_max_autotune_accumulation_matches_eager_gradients(self) -> None:
        # This uses a private inspection hook deliberately: the target-GPU gate
        # must fail if the eager ingress graph break makes Inductor silently
        # abandon CUDA graphs for the remaining model body.
        from torch._inductor.cudagraph_trees import get_manager

        torch.manual_seed(2718)
        eager = self._model()
        compiled_model = self._model()
        compiled_model.load_state_dict(eager.state_dict())
        batches = self._batches()

        eager_gradient_steps = self._accumulate(
            eager,
            eager,
            batches,
            CompileConfig(),
        )
        compile_config = CompileConfig(enabled=True, mode="max-autotune")
        cuda_device = torch.cuda.current_device()
        manager_before = get_manager(
            cuda_device,
            create_if_none_exists=False,
        )
        recorded_before = (
            0
            if manager_before is None
            else sum(
                1 + root.num_descendants()
                for root in manager_before.get_roots()
            )
        )
        backward_functions_before = (
            0
            if manager_before is None
            else sum(
                mode.name == "BACKWARD"
                for mode in manager_before.id_to_mode.values()
            )
        )
        compiled = compile_module(compiled_model, compile_config)
        compiled_gradient_steps = self._accumulate(
            compiled_model,
            compiled,
            batches,
            compile_config,
        )

        manager_after = get_manager(
            cuda_device,
            create_if_none_exists=False,
        )
        self.assertIsNotNone(
            manager_after,
            "max-autotune created no CUDA-graph tree manager",
        )
        assert manager_after is not None
        recorded_after = sum(
            1 + root.num_descendants()
            for root in manager_after.get_roots()
        )
        self.assertGreater(
            recorded_after,
            recorded_before,
            "the compiled model recorded no CUDA-graph nodes",
        )
        backward_functions_after = sum(
            mode.name == "BACKWARD"
            for mode in manager_after.id_to_mode.values()
        )
        self.assertGreater(
            backward_functions_after,
            backward_functions_before,
            "the differentiable model body registered no CUDA-graph backward",
        )
        self.assertEqual(
            manager_after.path_state.name,
            "EXECUTION",
            "the third micro-batch did not replay a recorded CUDA graph",
        )
        self.assertIsNotNone(manager_after.current_node)
        assert manager_after.current_node is not None
        self.assertEqual(
            manager_after.id_to_mode[
                manager_after.current_node.wrapped_function.id
            ].name,
            "BACKWARD",
            "the replayed node was not a CUDA-graph backward region",
        )

        self.assertEqual(
            len(compiled_gradient_steps),
            len(eager_gradient_steps),
        )
        for micro_step, (compiled_gradients, eager_gradients) in enumerate(
            zip(compiled_gradient_steps, eager_gradient_steps),
            start=1,
        ):
            self.assertEqual(
                compiled_gradients.keys(),
                eager_gradients.keys(),
            )
            for name in eager_gradients:
                with self.subTest(micro_step=micro_step, parameter=name):
                    self.assertTrue(
                        torch.isfinite(compiled_gradients[name]).all()
                    )
                    torch.testing.assert_close(
                        compiled_gradients[name],
                        eager_gradients[name],
                        rtol=5e-2,
                        atol=5e-3,
                    )


if __name__ == "__main__":
    unittest.main()
