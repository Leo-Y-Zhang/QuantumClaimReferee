"""Guards on examples/run_example.py, which nothing else watches.

The example is outside ``testpaths`` and was outside the lint scope, so it drifted:
it certified a hypothesis labelled S=2.88, above the Tsirelson bound the library
itself refuses, because it read ``.p_memory_robust`` off a result whose ``.status``
was already ASSUMPTIONS_UNMET. A worked example that contradicts the tool is worse
than no example, so the two halves of that mistake are pinned here.

These read the file rather than running it: act 3 of the example runs 200,000
Monte-Carlo trials, which does not belong in a unit suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from qcref import chsh
from qcref.status import ASSUMPTIONS_UNMET

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "run_example.py"


def _example_tree() -> ast.Module:
    return ast.parse(EXAMPLE.read_text(encoding="utf-8"), filename=str(EXAMPLE))


def _chsh_calls() -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_example_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "chsh"
    ]


def test_the_example_file_is_where_the_docs_say_it_is():
    assert EXAMPLE.is_file()


@pytest.mark.parametrize("call", _chsh_calls(), ids=lambda c: f"line{c.lineno}")
def test_example_chsh_arguments_are_physical(call):
    # Every (wins, rounds) the example feeds to chsh must survive the certifier's
    # own assumption gates. A pair implying S above the Tsirelson bound comes back
    # ASSUMPTIONS_UNMET with a p-value of ~1e-33, so an example that reads only the
    # p-value certifies an impossible claim without ever printing a warning.
    wins, rounds = (ast.literal_eval(arg) for arg in call.args)
    result = chsh(wins, rounds, setting_randomness_declared=True)
    assert result.status != ASSUMPTIONS_UNMET, (
        f"{EXAMPLE.name}:{call.lineno} chsh({wins}, {rounds}) is "
        f"{result.status}: {result.unmet_reason}"
    )


def test_example_carries_the_chsh_status_into_the_study():
    # The other half: reading .p_memory_robust off a CHSH result and dropping the
    # status is what let the bad case through. Any hypothesis built from a CHSH
    # p-value must also pass assumptions_met, so a refusal reaches the report.
    adds = [
        node
        for node in ast.walk(_example_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add"
    ]
    from_chsh = [
        node
        for node in adds
        if any(
            isinstance(sub, ast.Attribute) and sub.attr == "p_memory_robust"
            for arg in node.args
            for sub in ast.walk(arg)
        )
    ]
    assert from_chsh, "expected the example to build hypotheses from CHSH p-values"
    for node in from_chsh:
        kwargs = {kw.arg for kw in node.keywords}
        assert "assumptions_met" in kwargs, (
            f"{EXAMPLE.name}:{node.lineno} adds a CHSH p-value without its status"
        )
