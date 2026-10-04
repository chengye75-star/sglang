"""A layer stack connects the stages appended to it in order."""

import unittest

import test_declared_decoder_boundary as fixture

from sglang.srt.layers.layer_boundary import (
    ProducerReduction,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def layer(sparse=False):
    """One decoder layer's stages: attention, then an FFN."""
    return append_stages(
        (declare_attn(), fixture.Norm()),
        (declare_ffn(sparse=sparse, next_layer_sparse=sparse), fixture.Norm()),
    )


def mixer():
    """A single-stage mixer whose exit depends on the stage after it."""
    return declare_attn(
        reduction=ProducerReduction.EXIT_SCOPED, gathers_attn_tp_input=False
    )


class TestAppendStages(CustomTestCase):
    def setUp(self):
        self.planning = fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2))
        self.planning.__enter__()
        self.addCleanup(self.planning.__exit__, None, None, None)

    def test_only_the_stacks_last_stage_ends_it(self):
        with layer_stack():
            stages = [s for _ in range(3) for s in layer()]
        self.assertEqual([s.plan.terminal for s in stages], [False] * 5 + [True])
        self.assertEqual([s.plan.enters_stack for s in stages], [True] + [False] * 5)

    def test_a_one_layer_stack_starts_and_ends_on_that_layer(self):
        # A NextN / MTP draft builds its single layer in a stack of its own.
        with layer_stack():
            attention, ffn = layer()
        self.assertTrue(attention.plan.enters_stack)
        self.assertTrue(ffn.plan.terminal)

    def test_every_stage_binds_when_the_stack_closes(self):
        with layer_stack():
            stages = [s for _ in range(3) for s in layer()]
            # Declarations are usable at once; plans wait for the neighbours.
            self.assertTrue(all(s.declaration is not None for s in stages))
            self.assertTrue(all(s.plan is None for s in stages))
        self.assertTrue(all(s.plan is not None for s in stages))

    def test_appending_needs_an_open_stack(self):
        with self.assertRaisesRegex(RuntimeError, "open layer stack"):
            layer()

    def test_the_previous_layer_gives_the_first_stage_its_producer(self):
        built = []

        def previous_layer():
            built.append(None)
            layer(sparse=True)

        with layer_stack(previous_layers=[previous_layer]):
            first, _ = layer()
        self.assertEqual(len(built), 1)
        self.assertTrue(first.declaration.previous.sparse)
        self.assertFalse(first.plan.enters_stack)

    def test_the_next_layer_gives_the_last_stage_its_consumer(self):
        with layer_stack(
            next_layers=[lambda: append_stages((declare_ffn(), fixture.Norm()))]
        ):
            (stage,) = append_stages((mixer(), fixture.Norm()))
        self.assertFalse(stage.plan.terminal)
        # It leaves its sum to the FFN that follows it, which completes it.
        for edge in stage.plan.edges.values():
            self.assertTrue(edge.outgoing.produced.always_partial)

    def test_a_stack_without_stages_builds_no_neighbour(self):
        def unexpected():
            self.fail("a neighbour was built for a stack without stages")

        with layer_stack(previous_layers=[unexpected], next_layers=[unexpected]):
            pass

    def test_neighbours_are_built_after_the_stacks_own_layers(self):
        order = []

        def neighbour():
            order.append("neighbour")
            layer()

        with layer_stack(previous_layers=[neighbour], next_layers=[neighbour]):
            order.append("own")
            layer()
        self.assertEqual(order, ["own", "neighbour", "neighbour"])

    def test_a_neighbour_without_stages_is_passed_over(self):
        with layer_stack(previous_layers=[lambda: None, lambda: layer(sparse=True)]):
            first, _ = layer()
        self.assertTrue(first.declaration.previous.sparse)
        with layer_stack(previous_layers=[lambda: None], next_layers=[lambda: None]):
            first, last = layer()
        # No earlier layer declares a stage: the stack starts here, and ends
        # here when no later one does.
        self.assertTrue(first.plan.enters_stack)
        self.assertTrue(last.plan.terminal)

    def test_a_neighbour_is_read_without_joining_the_stack(self):
        def previous_layer():
            # A layer with a dense FFN, then one with a MoE: the producer is
            # the nearest stage, and neither joins this stack.
            layer(sparse=False)
            layer(sparse=True)

        with layer_stack(previous_layers=[previous_layer]):
            stages = layer()
        self.assertTrue(stages[0].declaration.previous.sparse)
        self.assertIsNone(stages[0].declaration.previous.previous)

    def test_a_branch_neither_extends_the_stack_nor_waits(self):
        with layer_stack():
            attention, moe = layer(sparse=True)
            branch = append_stages(
                (declare_ffn(), fixture.Norm()),
                (declare_attn(), fixture.Norm()),
                (declare_ffn(), fixture.Norm()),
                prepared_from=moe.declaration,
            )
            following = layer()
        self.assertTrue(all(s.plan is not None for s in branch))
        # The branch reads the MoE's input; the next layer follows the main
        # line's MoE, not the branch's dense FFN.
        self.assertEqual(branch[0].declaration.prepared_from, moe.declaration)
        self.assertTrue(following[0].declaration.previous.sparse)

    def test_a_nested_stack_leaves_the_outer_one_untouched(self):
        with layer_stack():
            outer = layer()
            with layer_stack():
                inner = layer()
            later = layer()
        self.assertTrue(inner[0].plan.enters_stack and inner[1].plan.terminal)
        # The outer stack resumes from its own last stage.
        self.assertFalse(outer[1].plan.terminal)
        self.assertFalse(later[0].plan.enters_stack)
        self.assertTrue(later[1].plan.terminal)


if __name__ == "__main__":
    unittest.main()
