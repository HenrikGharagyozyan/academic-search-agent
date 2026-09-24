"""Guards the dependency direction between layers.

A layered layout only holds if something checks it. These tests read the
imports of every module and fail when an inner layer reaches outward — the
failure mode that turns "ports and adapters" back into a pile of files.
"""

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app"

# Each layer may import itself, the layers listed here, and nothing else of ours.
ALLOWED: dict[str, set[str]] = {
    # Entities and pure algorithms. Depends on nothing of ours but itself.
    "domain": set(),
    # Contracts. They speak in domain types, so they may see the domain.
    "ports": {"domain"},
    # Adapters: implement ports using real services and settings.
    "infrastructure": {"domain", "ports", "core"},
    # Orchestration: depends on contracts, not on vendors — except where the
    # graph picks the default adapters to wire in.
    "application": {"domain", "ports", "core", "infrastructure"},
    # Transport: request/response shapes plus the services behind them.
    "api": {"domain", "ports", "core", "application"},
    "core": set(),
    "scripts": {"domain", "ports", "core", "application", "infrastructure"},
}


def module_layer(path: pathlib.Path) -> str:
    return path.relative_to(APP).parts[0]


def imported_layers(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    layers = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = node.names[0].name.split(".")
        else:
            continue

        if len(parts) >= 2 and parts[0] == "app":
            layers.add(parts[1])

    return layers


def source_files() -> list[pathlib.Path]:
    # main.py is the composition root: assembling the layers is its job, so it
    # is the one module allowed to reach across all of them.
    return sorted(
        p for p in APP.rglob("*.py")
        if "__pycache__" not in p.parts and p.name not in {"__init__.py", "main.py"}
    )


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(APP)))
def test_module_only_imports_layers_it_is_allowed_to(path):
    layer = module_layer(path)
    permitted = ALLOWED[layer] | {layer}
    violations = imported_layers(path) - permitted

    assert not violations, (
        f"{path.relative_to(APP)} is in '{layer}', which may import "
        f"{sorted(permitted)} — but it imports {sorted(violations)}"
    )


def test_domain_is_free_of_third_party_service_clients():
    """The domain holds entities and pure text algorithms. A vendor SDK in
    there means business rules have been welded to a service again."""
    forbidden = {"chromadb", "firecrawl", "langchain", "langchain_core",
                 "langchain_openai", "langchain_google_genai", "fastapi"}

    offenders = []
    for path in APP.joinpath("domain").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split(".")[0] in forbidden for n in names):
                offenders.append(f"{path.relative_to(APP)} -> {names}")

    assert not offenders, offenders


def test_every_layer_directory_is_declared():
    # A new top-level package must be placed in the hierarchy deliberately,
    # not silently escape these checks.
    on_disk = {p.name for p in APP.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert on_disk == set(ALLOWED)
