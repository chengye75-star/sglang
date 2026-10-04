"""Layers that append stages are built inside an open layer stack."""

import ast
import unittest
from pathlib import Path

import sglang.srt.models
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _calls(node):
    return {
        func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for func in [call.func]
    }


def _stage_classes(trees):
    """Classes whose construction appends stages: a method reaches
    append_stages, directly or through a module-level helper, or a base class
    does. Derived from the code, so a new model joins without being listed."""
    helpers, functions = {"append_stages"}, {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                functions.setdefault(node.name, []).append(node)
    changed = True
    while changed:
        changed = False
        for name, nodes in functions.items():
            if name not in helpers and any(_calls(fn) & helpers for fn in nodes):
                helpers.add(name)
                changed = True
    # Same-named classes in different files: keep every one, so a name
    # appends if any class of that name does (fails closed).
    classes = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
                classes.setdefault(node.name, []).append((node, bases))
    stage = set()
    changed = True
    while changed:
        changed = False
        for name, defs in classes.items():
            if name in stage:
                continue
            if any(
                bases & stage
                or any(
                    isinstance(m, ast.FunctionDef) and _calls(m) & helpers
                    for m in node.body
                )
                for node, bases in defs
            ):
                stage.add(name)
                changed = True
    return stage


def _opens_a_stack(node):
    if isinstance(node, ast.With):
        return any("layer_stack" in ast.unparse(i.context_expr) for i in node.items)
    return isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
        "make_pp_layers",
    )


def _unstacked_constructions(trees, stage):
    found = []
    for path, tree in trees.items():
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        stacked = [n for n in ast.walk(tree) if _opens_a_stack(n)]
        # A named factory used inside a stack (e.g. handed to make_pp_layers).
        factories = {
            x.id for s in stacked for x in ast.walk(s) if isinstance(x, ast.Name)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in stage
            ):
                continue
            chain, x = [], node
            while x in parents:
                x = parents[x]
                chain.append(x)
            if any(_opens_a_stack(c) for c in chain):
                continue
            enclosing = next(
                (c for c in chain if isinstance(c, (ast.FunctionDef, ast.Lambda))),
                None,
            )
            if (
                isinstance(enclosing, ast.FunctionDef)
                and enclosing.name != "__init__"
                and enclosing.name in factories
            ):
                continue
            found.append(f"{path.name}:{node.lineno} {node.func.id}")
    return found


class TestStageLayersBuiltInAStack(CustomTestCase):
    def test_every_stage_layer_is_built_inside_a_layer_stack(self):
        # append_stages raises outside a stack, but only when that model is
        # built: a draft or wrapper model nobody constructs in CI would fail
        # only in production. This checks every construction site instead.
        (models,) = map(Path, sglang.srt.models.__path__)
        trees = {p: ast.parse(p.read_text()) for p in sorted(models.rglob("*.py"))}
        stage = _stage_classes(trees)
        self.assertGreater(len(stage), 40)
        self.assertEqual(_unstacked_constructions(trees, stage), [])

    def test_a_layer_built_outside_a_stack_is_reported(self):
        source = (
            "def make_stage_boundary(norm):\n"
            "    return append_stages((decl, norm))\n"
            "class Layer:\n"
            "    def __init__(self):\n"
            "        self.boundary = make_stage_boundary(None)\n"
            "class DraftLayer(Layer):\n"
            "    pass\n"
            "class Draft:\n"
            "    def __init__(self):\n"
            "        self.decoder = DraftLayer()\n"
            "class Stacked:\n"
            "    def __init__(self):\n"
            "        with layer_stack():\n"
            "            self.decoder = DraftLayer()\n"
        )
        # Another file defines a class of the same name that appends nothing.
        other = "class Layer:\n    def __init__(self):\n        pass\n"
        trees = {Path("draft.py"): ast.parse(source), Path("z.py"): ast.parse(other)}
        stage = _stage_classes(trees)
        # Reached through a helper, inherited by a subclass, and not hidden by
        # a same-named class elsewhere.
        self.assertEqual(stage, {"Layer", "DraftLayer"})
        self.assertEqual(
            _unstacked_constructions(trees, stage), ["draft.py:10 DraftLayer"]
        )


if __name__ == "__main__":
    unittest.main()
