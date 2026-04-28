"""Agentic testcase-ID catalog for previously unmapped tests.

This file assigns deterministic testcase IDs with prefix:
AI_HOSTCPU_AGENTIC_TEST_ID_###

to tests that are not covered by existing FUNC/PERF/ML catalogs.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pytest


@dataclass(frozen=True)
class AgenticPlanCase:
    case_id: str
    selector: str
    title: str


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_all_test_selectors(root: Path) -> List[str]:
    selectors: List[str] = []
    tests_dir = root / "tests"

    for py_file in sorted(tests_dir.glob("test_*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        rel = str(py_file.relative_to(root)).replace("\\", "/")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_"):
                        selectors.append(f"{rel}::{node.name}::{fn.name}")
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                selectors.append(f"{rel}::{node.name}")

    return sorted(selectors)


def _read_assignments(path: Path) -> Dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node.value
    return out


def _selectors_with_existing_ids(root: Path) -> Set[str]:
    mapped: Set[str] = set()

    # Functional mappings
    func_assign = _read_assignments(root / "tests" / "test_functional_test_plan_mapping.py")
    func_selectors = ast.literal_eval(func_assign["_MAPPED_SELECTORS"])
    for values in func_selectors.values():
        mapped.update(values)

    # Performance and ML suite selectors are mapped through parametrized IDs.
    mapped.add("tests/test_performance_suite.py::test_performance_case_suite")
    mapped.add("tests/test_mlperf_suite.py::test_mlperf_case_suite")

    # Explicit testcase IDs embedded in docstrings are considered mapped.
    pattern = re.compile(r"AI_HOSTCPU_(?:FUNC|PERF|ML)_TEST_ID_\d{3}")
    for py_file in sorted((root / "tests").glob("test_*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        rel = str(py_file.relative_to(root)).replace("\\", "/")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_"):
                        doc = ast.get_docstring(fn) or ""
                        if pattern.search(doc):
                            mapped.add(f"{rel}::{node.name}::{fn.name}")
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                if pattern.search(doc):
                    mapped.add(f"{rel}::{node.name}")

    return mapped


def _selector_title(selector: str) -> str:
    test_name = selector.split("::")[-1]
    if test_name.startswith("test_"):
        test_name = test_name[5:]
    return test_name.replace("_", " ")


def _build_agentic_cases() -> List[AgenticPlanCase]:
    root = _workspace_root()
    all_selectors = _discover_all_test_selectors(root)
    covered = _selectors_with_existing_ids(root)

    unmapped = [s for s in all_selectors if s not in covered]

    cases: List[AgenticPlanCase] = []
    for idx, selector in enumerate(unmapped, start=1):
        cases.append(
            AgenticPlanCase(
                case_id=f"AI_HOSTCPU_AGENTIC_TEST_ID_{idx:03d}",
                selector=selector,
                title=_selector_title(selector),
            )
        )

    return cases


AGENTIC_PLAN_CASES: List[AgenticPlanCase] = _build_agentic_cases()


@pytest.mark.node_validation
@pytest.mark.parametrize("case", AGENTIC_PLAN_CASES, ids=[c.case_id for c in AGENTIC_PLAN_CASES])
def test_agentic_plan_mapping_catalog(case: AgenticPlanCase):
    """Validate shape and referential integrity of agentic testcase mappings."""
    assert re.fullmatch(r"AI_HOSTCPU_AGENTIC_TEST_ID_\d{3}", case.case_id)
    assert case.selector.strip()
    assert case.title.strip()

    file_part = case.selector.split("::", 1)[0]
    assert (_workspace_root() / file_part).exists(), f"Mapped selector file does not exist: {case.selector}"


def test_agentic_plan_ids_unique_and_sequential():
    """Ensure generated AI_HOSTCPU_AGENTIC_TEST_ID values are unique and sequential."""
    ids = [c.case_id for c in AGENTIC_PLAN_CASES]
    assert len(ids) == len(set(ids))
    if not ids:
        return

    expected = [f"AI_HOSTCPU_AGENTIC_TEST_ID_{i:03d}" for i in range(1, len(ids) + 1)]
    assert ids == expected
