"""Fail-closed launch guard for the canonical direct-RQ ODT program.

This module intentionally imports no project model code.  A runner can install
the runtime guard before importing any model, compiler, or checkpoint helper.
The static audit follows local imports from the runner and hashes every source
file it inspected.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from typing import Any, Iterable


class DirectOnlyComplianceError(RuntimeError):
    """Raised before a prohibited numerical route can execute."""


_PROHIBITED_TERMINALS = frozenset(
    {
        "svd",
        "svdvals",
        "svd_lowrank",
        "pca_lowrank",
        "pinv",
        "pinverse",
        "lstsq",
        "cholesky",
        "cholesky_ex",
        "cholesky_inverse",
        "matrix_power",
        "matrix_rank",
        "cond",
        "polar",
        "cov",
        "covariance",
        "gram",
        "normal_equation",
        "inv",
        "inverse",
        "solve",
        "lu",
        "lu_factor",
        "lu_factor_ex",
        "lu_solve",
        "tensorinv",
        "tensorsolve",
    }
)

_PRODUCTION_MODULE = "xvla.train.implicit_sparse_projective_odt"
_INDEPENDENT_REFERENCE_MODULE = "xvla.train.direct_odt_clone_reference"

_DIRECT_QR_CALLERS = frozenset(
    {
        (_PRODUCTION_MODULE, "_positive_diagonal_direct_rq_rows"),
        (_PRODUCTION_MODULE, "_direct_tsqr_panel"),
        (_PRODUCTION_MODULE, "_independent_dense_clone_rq"),
        (_INDEPENDENT_REFERENCE_MODULE, "_independent_dense_clone_rq"),
    }
)
_TRIANGULAR_CALLERS = frozenset(
    {(_PRODUCTION_MODULE, "_solve_compact_q")}
)
_ALGORITHM3_EVD_CALLERS = frozenset(
    {
        (_PRODUCTION_MODULE, "diagonalize_implicit_dag_full_rank"),
        (
            _PRODUCTION_MODULE,
            "apply_shared_eigenbases_to_explicit_clone_occurrences_control",
        ),
        (
            _PRODUCTION_MODULE,
            "diagonalize_shared_and_explicit_clone_independently",
        ),
        (
            _INDEPENDENT_REFERENCE_MODULE,
            "diagonalize_shared_and_explicit_clone_independently",
        ),
    }
)
_APPROVED_ALGORITHM2_SELF_OVERLAP_EINSUMS = frozenset(
    {
        (_PRODUCTION_MODULE, "_role_environment", "oi,op,pj->ij"),
        (_PRODUCTION_MODULE, "_role_environment", "oia,op,pja->ij"),
        (_PRODUCTION_MODULE, "_role_environment", "oai,op,paj->ij"),
        (_PRODUCTION_MODULE, "_role_environment", "ti,si->ts"),
        (_INDEPENDENT_REFERENCE_MODULE, "_role_environment", "oi,op,pj->ij"),
        (_INDEPENDENT_REFERENCE_MODULE, "_role_environment", "oia,op,pja->ij"),
        (_INDEPENDENT_REFERENCE_MODULE, "_role_environment", "oai,op,paj->ij"),
        (
            _INDEPENDENT_REFERENCE_MODULE,
            "contract_clone_environments",
            "ao,ab,bp->op",
        ),
    }
)
_APPROVED_ALGORITHM2_SELF_OVERLAP_MATMULS = frozenset(
    {
        (
            _PRODUCTION_MODULE,
            "_role_environment",
            "core.matrix.T @ downstream @ core.matrix",
        ),
        (
            _PRODUCTION_MODULE,
            "_role_environment",
            "core.output_factor.T @ downstream @ core.output_factor",
        ),
        (
            _PRODUCTION_MODULE,
            "_role_environment",
            "selected.T @ weighted @ selected",
        ),
        (
            _PRODUCTION_MODULE,
            "reverse_implicit_environments",
            "network.head.T @ output_metric @ network.head",
        ),
        (
            _PRODUCTION_MODULE,
            "diagonalize_implicit_dag_full_rank",
            "work.head.T @ metric @ work.head",
        ),
    }
)
_CONTROLLED_TERMINALS = frozenset(
    {"qr", "solve_triangular", "eigh", "eigvalsh", "eig", "eigvals"}
)
_LOW_LEVEL_SVD_TERMINALS = frozenset({"_linalg_svd", "linalg_svd"})
_PATCHED_NORM_ENTRYPOINTS = frozenset(
    {
        "numpy.linalg.linalg.norm",
        "numpy.linalg.norm",
        "torch.Tensor.norm",
        "torch.functional.norm",
        "torch.linalg.norm",
        "torch.norm",
    }
)

# This is an exact, version-pinned guard surface, not a lower bound.  Missing
# wrappers and silently-added wrappers are both launch failures.  Keep this
# list independent of the patch loop below so a typo or API drift cannot make
# the installer and its validator agree by construction.
EXPECTED_RUNTIME_GUARD_ENTRYPOINTS = frozenset(
    {
        "numpy.linalg.cholesky",
        "numpy.linalg.cond",
        "numpy.linalg.eig",
        "numpy.linalg.eigh",
        "numpy.linalg.eigvals",
        "numpy.linalg.eigvalsh",
        "numpy.linalg.inv",
        "numpy.linalg.linalg.cholesky",
        "numpy.linalg.linalg.cond",
        "numpy.linalg.linalg.eig",
        "numpy.linalg.linalg.eigh",
        "numpy.linalg.linalg.eigvals",
        "numpy.linalg.linalg.eigvalsh",
        "numpy.linalg.linalg.inv",
        "numpy.linalg.linalg.lstsq",
        "numpy.linalg.linalg.matrix_power",
        "numpy.linalg.linalg.matrix_rank",
        "numpy.linalg.linalg.norm",
        "numpy.linalg.linalg.pinv",
        "numpy.linalg.linalg.solve",
        "numpy.linalg.linalg.svd",
        "numpy.linalg.linalg.tensorinv",
        "numpy.linalg.linalg.tensorsolve",
        "numpy.linalg.lstsq",
        "numpy.linalg.matrix_power",
        "numpy.linalg.matrix_rank",
        "numpy.linalg.norm",
        "numpy.linalg.pinv",
        "numpy.linalg.solve",
        "numpy.linalg.svd",
        "numpy.linalg.tensorinv",
        "numpy.linalg.tensorsolve",
        "torch.Tensor.cholesky",
        "torch.Tensor.cholesky_inverse",
        "torch.Tensor.cov",
        "torch.Tensor.inverse",
        "torch.Tensor.lstsq",
        "torch.Tensor.lu",
        "torch.Tensor.lu_solve",
        "torch.Tensor.matrix_power",
        "torch.Tensor.norm",
        "torch.Tensor.pinverse",
        "torch.Tensor.solve",
        "torch.Tensor.svd",
        "torch.cholesky",
        "torch.cholesky_inverse",
        "torch.cond",
        "torch.cov",
        "torch.functional.norm",
        "torch.inverse",
        "torch.linalg.cholesky",
        "torch.linalg.cholesky_ex",
        "torch.linalg.cond",
        "torch.linalg.eig",
        "torch.linalg.eigh",
        "torch.linalg.eigvals",
        "torch.linalg.eigvalsh",
        "torch.linalg.inv",
        "torch.linalg.lstsq",
        "torch.linalg.lu",
        "torch.linalg.lu_factor",
        "torch.linalg.lu_factor_ex",
        "torch.linalg.lu_solve",
        "torch.linalg.matrix_norm",
        "torch.linalg.matrix_power",
        "torch.linalg.matrix_rank",
        "torch.linalg.norm",
        "torch.linalg.pinv",
        "torch.linalg.qr",
        "torch.linalg.solve",
        "torch.linalg.solve_triangular",
        "torch.linalg.svd",
        "torch.linalg.svdvals",
        "torch.linalg.tensorinv",
        "torch.linalg.tensorsolve",
        "torch.lstsq",
        "torch.lu",
        "torch.lu_solve",
        "torch.matrix_power",
        "torch.matrix_rank",
        "torch.norm",
        "torch.pca_lowrank",
        "torch.pinverse",
        "torch.polar",
        "torch.solve",
        "torch.svd",
        "torch.svd_lowrank",
    }
)
EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT = 87
if len(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS) != EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT:
    raise RuntimeError("the direct-only guard surface constant is internally inconsistent")

ALLOWED_RUNTIME_CALLS = frozenset(
    {
        f"{module}.{caller}:torch.linalg.qr"
        for module, caller in _DIRECT_QR_CALLERS
    }
    | {
        f"{module}.{caller}:torch.linalg.solve_triangular"
        for module, caller in _TRIANGULAR_CALLERS
    }
    | {
        f"{module}.{caller}:torch.linalg.eigh"
        for module, caller in _ALGORITHM3_EVD_CALLERS
    }
)

DIRECT_RQ_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}._positive_diagonal_direct_rq_rows:torch.linalg.qr"
)
STREAMED_QR_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}._direct_tsqr_panel:torch.linalg.qr"
)
TRIANGULAR_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}._solve_compact_q:torch.linalg.solve_triangular"
)
PRODUCTION_ALGORITHM3_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}.diagonalize_implicit_dag_full_rank:torch.linalg.eigh"
)
PRODUCTION_CLONE_QR_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}._independent_dense_clone_rq:torch.linalg.qr"
)
PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL = (
    f"{_PRODUCTION_MODULE}.diagonalize_shared_and_explicit_clone_independently:"
    "torch.linalg.eigh"
)
REFERENCE_CLONE_QR_RUNTIME_CALL = (
    f"{_INDEPENDENT_REFERENCE_MODULE}._independent_dense_clone_rq:torch.linalg.qr"
)
REFERENCE_ALGORITHM3_RUNTIME_CALL = (
    f"{_INDEPENDENT_REFERENCE_MODULE}."
    "diagonalize_shared_and_explicit_clone_independently:torch.linalg.eigh"
)
STREAMED_CLONE_ORACLE_RUNTIME_CALLS = frozenset(
    {
        DIRECT_RQ_RUNTIME_CALL,
        STREAMED_QR_RUNTIME_CALL,
        TRIANGULAR_RUNTIME_CALL,
        PRODUCTION_ALGORITHM3_RUNTIME_CALL,
        REFERENCE_CLONE_QR_RUNTIME_CALL,
        REFERENCE_ALGORITHM3_RUNTIME_CALL,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_cli_option(arguments: Iterable[str], option: str) -> str | None:
    """Return one CLI option value and reject duplicate spellings.

    This parser is deliberately small so launchers can use it before importing
    Torch or deciding whether a production manifest is required.  ``argparse``
    otherwise accepts repeated flags and silently keeps the last value.
    """

    values: list[str] = []
    items = tuple(arguments)
    for index, argument in enumerate(items):
        if argument == option:
            if index + 1 >= len(items) or items[index + 1].startswith("--"):
                raise DirectOnlyComplianceError(f"{option} requires one value")
            values.append(items[index + 1])
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
            if not value:
                raise DirectOnlyComplianceError(f"{option} requires one value")
            values.append(value)
    if len(values) > 1:
        raise DirectOnlyComplianceError(
            f"{option} may be specified at most once"
        )
    return values[0] if values else None


def _resolve_local_module(project_root: Path, module: str) -> Path | None:
    if not module or any(not part for part in module.split(".")):
        return None
    base = project_root.joinpath(*module.split("."))
    source = base.with_suffix(".py")
    if source.is_file():
        return source.resolve()
    package = base / "__init__.py"
    if package.is_file():
        return package.resolve()
    return None


def _local_imports(path: Path, project_root: Path) -> tuple[Path, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[Path] = set()
    relative = path.resolve().relative_to(project_root.resolve())
    module_parts = list(relative.with_suffix("").parts)
    package_parts = module_parts[:-1]
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module
            else:
                ascend = node.level - 1
                if ascend > len(package_parts):
                    raise DirectOnlyComplianceError(
                        f"relative import escapes the project package: {path}:{node.lineno}"
                    )
                prefix = package_parts[: len(package_parts) - ascend]
                suffix = [] if node.module is None else node.module.split(".")
                base = ".".join((*prefix, *suffix))
            if base:
                modules.append(base)
                # ``from package import submodule`` can load either the package
                # or the named submodule.  Include both when local files exist.
                modules.extend(f"{base}.{alias.name}" for alias in node.names)
        for module in modules:
            resolved = _resolve_local_module(project_root, module)
            if resolved is not None:
                found.add(resolved)
                parts = module.split(".")
                for stop in range(1, len(parts)):
                    package = project_root.joinpath(*parts[:stop], "__init__.py")
                    if package.is_file():
                        found.add(package.resolve())
    return tuple(sorted(found))


def _module_name(path: Path, project_root: Path) -> str:
    relative = path.resolve().relative_to(project_root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def transitive_local_sources(project_root: Path, entrypoints: Iterable[Path]) -> tuple[Path, ...]:
    """Return the complete local Python import closure of ``entrypoints``."""

    project_root = project_root.resolve()
    pending = [path.resolve() for path in entrypoints]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file() or project_root not in path.parents:
            raise DirectOnlyComplianceError(f"audit source is outside the project or missing: {path}")
        visited.add(path)
        pending.extend(item for item in _local_imports(path, project_root) if item not in visited)
    return tuple(sorted(visited))


def canonical_direct_only_entrypoints(
    project_root: Path, runner: Path
) -> tuple[Path, ...]:
    """Authoritative runner plus its direct-only regression helpers."""

    project_root = project_root.resolve()
    paths = (
        runner.resolve(),
        project_root / "scripts/run_direct_odt_clone_oracle.py",
        project_root / "scripts/run_implicit_sparse_projective_odt.py",
        project_root / "scripts/run_implicit_sparse_projective_odt_all_tokens.py",
        project_root / "scripts/run_implicit_sparse_projective_odt_vla.py",
        project_root / "tests/test_implicit_sparse_projective_odt.py",
        project_root / "tests/test_implicit_sparse_projective_odt_all_tokens.py",
        project_root / "tests/test_implicit_sparse_projective_odt_heterogeneous.py",
        project_root / "tests/test_implicit_sparse_projective_odt_vla.py",
        project_root / "tests/test_direct_odt_clone_reference.py",
    )
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise DirectOnlyComplianceError(
            "canonical direct-only audit entrypoint is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return tuple(dict.fromkeys(paths))


def _expression_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _expression_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    # Resolve simple callable aliases such as ``factor = torch.linalg.qr``.
    sensitive = (
        _PROHIBITED_TERMINALS
        | _CONTROLLED_TERMINALS
        | _LOW_LEVEL_SVD_TERMINALS
        | {"matrix_norm", "norm"}
    )
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            name = _expression_name(value, result) if value is not None else None
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if name is None:
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                previous = result.get(target.id)
                name_is_sensitive = (
                    any(part in sensitive for part in name.split("."))
                    or name.startswith("<odt-sensitive-alias")
                )
                previous_is_sensitive = previous is not None and (
                    any(part in sensitive for part in previous.split("."))
                    or previous.startswith("<odt-sensitive-alias")
                )
                if previous is None:
                    result[target.id] = name
                    changed = True
                elif name_is_sensitive and not previous_is_sensitive:
                    result[target.id] = name
                    changed = True
                elif name_is_sensitive and previous_is_sensitive and name != previous:
                    marker = "<odt-sensitive-alias-conflict>"
                    if previous != marker:
                        result[target.id] = marker
                        changed = True
    return result


def _constant_argument(
    call: ast.Call, key: str, positional_index: int
) -> object | None:
    for keyword in call.keywords:
        if keyword.arg == key and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    if len(call.args) > positional_index and isinstance(
        call.args[positional_index], ast.Constant
    ):
        return call.args[positional_index].value
    return None


def _argument_node(
    call: ast.Call, key: str, positional_index: int
) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == key:
            return keyword.value
    if len(call.args) > positional_index:
        return call.args[positional_index]
    return None


def _same_expression(first: ast.AST, second: ast.AST) -> bool:
    return ast.dump(first, include_attributes=False) == ast.dump(
        second, include_attributes=False
    )


def _transpose_base(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Attribute) and node.attr in {"T", "mT"}:
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "transpose" and len(node.args) >= 2:
            dimensions = node.args[-2:]
            if all(isinstance(value, ast.Constant) for value in dimensions):
                pair = tuple(value.value for value in dimensions)
                if pair in {(-1, -2), (-2, -1), (0, 1), (1, 0)}:
                    return node.func.value
    return None


def _matmul_operands(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return [*_matmul_operands(node.left), *_matmul_operands(node.right)]
    return [node]


def _is_self_overlap_matmul(node: ast.BinOp) -> bool:
    if not isinstance(node.op, ast.MatMult):
        return False
    operands = _matmul_operands(node)
    for first, left in enumerate(operands):
        for right in operands[first + 1 :]:
            right_base = _transpose_base(right)
            if right_base is not None and _same_expression(left, right_base):
                return True
            left_base = _transpose_base(left)
            if left_base is not None and _same_expression(left_base, right):
                return True
    return False


def _call_is_self_overlap(call: ast.Call) -> bool:
    if len(call.args) < 2:
        return False
    first, second = call.args[:2]
    second_base = _transpose_base(second)
    if second_base is not None and _same_expression(first, second_base):
        return True
    first_base = _transpose_base(first)
    return first_base is not None and _same_expression(first_base, second)


def _einsum_reuses_an_operand(call: ast.Call) -> bool:
    if len(call.args) < 3:
        return False
    specification = call.args[0]
    if not isinstance(specification, ast.Constant) or not isinstance(
        specification.value, str
    ):
        return False
    equation = specification.value.replace(" ", "")
    if "->" not in equation:
        return True
    inputs_text, output = equation.split("->", 1)
    inputs = inputs_text.split(",")
    operands = call.args[1:]
    if len(inputs) != len(operands) or any("..." in item for item in inputs):
        return True
    for first in range(len(operands)):
        for second in range(first + 1, len(operands)):
            if not _same_expression(operands[first], operands[second]):
                continue
            first_unique = set(inputs[first]) - set(inputs[second])
            second_unique = set(inputs[second]) - set(inputs[first])
            # A genuine self-overlap leaves one distinct mode from each copy
            # in the output.  A coefficient-mediated polynomial such as
            # A[m,j,l] c[...,j] c[...,l] contracts both distinct polynomial
            # indices and is not an overlap construction.
            if first_unique.intersection(output) and second_unique.intersection(output):
                return True
    return False


def audit_direct_only_launch(
    project_root: Path,
    entrypoints: Iterable[Path],
    *,
    require_direct_qr: bool = True,
) -> dict[str, Any]:
    """Audit the transitive local launch sources and fail on forbidden routes."""

    project_root = project_root.resolve()
    sources = transitive_local_sources(project_root, entrypoints)
    violations: list[str] = []
    call_sites: list[str] = []
    guarded_spectral_norm_sites: list[str] = []
    self_overlap_sites: list[str] = []
    duplicate_definition_sites: list[str] = []
    direct_qr_sites = 0

    for path in sources:
        relative = path.relative_to(project_root).as_posix()
        module_name = _module_name(path, project_root)
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = _aliases(tree)
        top_level_definitions: dict[str, int] = {}
        for node in tree.body:
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            previous_line = top_level_definitions.get(node.name)
            if previous_line is not None:
                site = (
                    f"{relative}:{node.lineno}:duplicate_top_level_definition:"
                    f"{node.name}:first_line={previous_line}"
                )
                duplicate_definition_sites.append(site)
                violations.append(site)
            else:
                top_level_definitions[node.name] = node.lineno

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.functions: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_BinOp(self, node: ast.BinOp) -> None:
                caller = self.functions[-1] if self.functions else "<module>"
                if _is_self_overlap_matmul(node):
                    site = f"{relative}:{node.lineno}:{caller}:self_overlap_matmul"
                    expression = ast.unparse(node)
                    if (
                        module_name,
                        caller,
                        expression,
                    ) not in _APPROVED_ALGORITHM2_SELF_OVERLAP_MATMULS:
                        self_overlap_sites.append(site)
                        violations.append(site)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                nonlocal direct_qr_sites
                resolved = _expression_name(node.func, aliases) or "<dynamic>"
                terminal = resolved.rsplit(".", 1)[-1]
                caller = self.functions[-1] if self.functions else "<module>"
                site = f"{relative}:{node.lineno}:{caller}:{resolved}"
                call_sites.append(site)

                if resolved.startswith("<odt-sensitive-alias"):
                    violations.append(site)
                elif any(
                    part in _LOW_LEVEL_SVD_TERMINALS
                    for part in resolved.split(".")
                ):
                    violations.append(site)
                elif terminal in _PROHIBITED_TERMINALS:
                    violations.append(site)
                elif terminal == "qr":
                    direct_qr_sites += 1
                    if (
                        resolved != "torch.linalg.qr"
                        or (module_name, caller) not in _DIRECT_QR_CALLERS
                    ):
                        violations.append(site)
                elif terminal == "solve_triangular":
                    if (
                        resolved != "torch.linalg.solve_triangular"
                        or (module_name, caller) not in _TRIANGULAR_CALLERS
                    ):
                        violations.append(site)
                elif terminal in {"eigh", "eigvalsh", "eig", "eigvals"}:
                    if (
                        resolved != "torch.linalg.eigh"
                        or (module_name, caller) not in _ALGORITHM3_EVD_CALLERS
                    ):
                        violations.append(site)
                elif terminal == "matrix_norm":
                    order = _constant_argument(node, "ord", 1)
                    if order in {2, -2, "nuc"}:
                        # The model's training-only spectral clipping helper is
                        # imported but not called by ODT.  The runtime guard
                        # makes this dormant site unexecutable during a launch.
                        guarded_spectral_norm_sites.append(site)
                elif terminal == "norm":
                    explicit_numerical_namespace = (
                        resolved.startswith(("torch.", "numpy."))
                        or ".linalg." in resolved
                    )
                    if (
                        explicit_numerical_namespace
                        and resolved not in _PATCHED_NORM_ENTRYPOINTS
                    ):
                        violations.append(
                            f"{relative}:{node.lineno}:{caller}:"
                            f"unguarded_norm_namespace:{resolved}"
                        )
                    else:
                        order_key = (
                            "ord"
                            if resolved
                            in {
                                "torch.linalg.norm",
                                "numpy.linalg.norm",
                                "numpy.linalg.linalg.norm",
                            }
                            else "p"
                        )
                        order_node = _argument_node(node, order_key, 1)
                        if order_node is not None and not isinstance(
                            order_node, ast.Constant
                        ):
                            violations.append(
                                f"{relative}:{node.lineno}:{caller}:dynamic_norm_order"
                            )
                        elif isinstance(order_node, ast.Constant):
                            order = order_node.value
                            linalg_spectral = resolved in {
                                "torch.linalg.norm",
                                "numpy.linalg.norm",
                                "numpy.linalg.linalg.norm",
                            } and order in {2, -2, "nuc"}
                            legacy_nuclear = order == "nuc"
                            if linalg_spectral or legacy_nuclear:
                                guarded_spectral_norm_sites.append(site)
                elif terminal == "einsum" and _einsum_reuses_an_operand(node):
                    specification = node.args[0]
                    equation = (
                        specification.value
                        if isinstance(specification, ast.Constant)
                        and isinstance(specification.value, str)
                        else "<dynamic>"
                    )
                    if (
                        module_name,
                        caller,
                        equation,
                    ) not in _APPROVED_ALGORITHM2_SELF_OVERLAP_EINSUMS:
                        overlap_site = f"{relative}:{node.lineno}:{caller}:self_overlap_einsum"
                        self_overlap_sites.append(overlap_site)
                        violations.append(overlap_site)
                elif terminal == "einsum" and (
                    not node.args
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)
                ):
                    violations.append(f"{relative}:{node.lineno}:{caller}:dynamic_einsum_spec")
                elif terminal in {"matmul", "mm", "bmm"} and _call_is_self_overlap(node):
                    overlap_site = f"{relative}:{node.lineno}:{caller}:self_overlap_call"
                    self_overlap_sites.append(overlap_site)
                    violations.append(overlap_site)
                elif terminal == "getattr" and len(node.args) >= 2:
                    requested = node.args[1]
                    if isinstance(requested, ast.Constant) and requested.value in (
                        _PROHIBITED_TERMINALS
                        | _CONTROLLED_TERMINALS
                        | _LOW_LEVEL_SVD_TERMINALS
                    ):
                        violations.append(site)
                self.generic_visit(node)

        Visitor().visit(tree)

    if require_direct_qr and direct_qr_sites == 0:
        violations.append("launch:no_direct_qr_call_site")
    report = {
        "scope": "transitive_local_import_closure",
        "entrypoints": sorted(
            path.resolve().relative_to(project_root).as_posix() for path in entrypoints
        ),
        "source_sha256": {
            path.relative_to(project_root).as_posix(): _sha256(path) for path in sources
        },
        "source_count": len(sources),
        "direct_qr_call_sites": direct_qr_sites,
        "direct_qr_required": require_direct_qr,
        "guarded_dormant_spectral_norm_sites": sorted(guarded_spectral_norm_sites),
        "duplicate_top_level_definition_sites": sorted(
            set(duplicate_definition_sites)
        ),
        "prohibited_self_overlap_sites": sorted(set(self_overlap_sites)),
        "prohibited_calls_found": sorted(set(violations)),
        "call_site_count": len(call_sites),
    }
    if violations:
        raise DirectOnlyComplianceError(
            "transitive direct-only source audit failed: " + ", ".join(sorted(set(violations)))
        )
    return report


_RUNTIME_STATE: dict[str, Any] = {
    "installed": False,
    "patched_entrypoints": [],
    "allowed_calls": [],
    "prohibited_attempts": [],
}
_RUNTIME_WRAPPERS: dict[str, Any] = {}


def _caller_identity() -> tuple[str, str]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return "<unknown>", "<unknown>"
    caller = frame.f_back.f_back
    return caller.f_globals.get("__name__", "<unknown>"), caller.f_code.co_name


def _prohibited_wrapper(qualified_name: str):
    def blocked(*args: Any, **kwargs: Any):
        module, caller = _caller_identity()
        _RUNTIME_STATE["prohibited_attempts"].append(f"{module}.{caller}:{qualified_name}")
        raise DirectOnlyComplianceError(
            f"prohibited numerical route {qualified_name} called by {module}.{caller}"
        )

    return blocked


def _controlled_wrapper(
    original: Any,
    qualified_name: str,
    allowed_callers: frozenset[tuple[str, str]],
):
    def controlled(*args: Any, **kwargs: Any):
        module, caller = _caller_identity()
        if (module, caller) not in allowed_callers:
            _RUNTIME_STATE["prohibited_attempts"].append(
                f"{module}.{caller}:{qualified_name}"
            )
            raise DirectOnlyComplianceError(
                f"{qualified_name} is not allowed from {module}.{caller}"
            )
        _RUNTIME_STATE["allowed_calls"].append(f"{module}.{caller}:{qualified_name}")
        return original(*args, **kwargs)

    return controlled


_MISSING = object()


def _take_argument(
    positional: list[Any],
    keywords: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    if positional:
        if name in keywords:
            raise DirectOnlyComplianceError(f"duplicate norm argument {name}")
        return positional.pop(0)
    return keywords.pop(name, default)


def _norm_input(
    positional: list[Any], keywords: dict[str, Any], names: tuple[str, ...]
) -> Any:
    if positional:
        if any(name in keywords for name in names):
            raise DirectOnlyComplianceError("duplicate norm input argument")
        return positional.pop(0)
    present = [name for name in names if name in keywords]
    if len(present) != 1:
        raise DirectOnlyComplianceError("norm requires exactly one input")
    return keywords.pop(present[0])


def _forbidden_spectral_order(order: Any) -> bool:
    return (
        (isinstance(order, str) and order == "nuc")
        or (
            isinstance(order, (int, float))
            and not isinstance(order, bool)
            and order in {2, -2}
        )
    )


def _frobenius_order(order: Any) -> bool:
    return order is None or isinstance(order, str) and order == "fro"


def _raise_prohibited_norm(
    qualified_name: str, order: Any, module: str, caller: str
) -> None:
    detail = f"{qualified_name}(order={order!r})"
    _RUNTIME_STATE["prohibited_attempts"].append(f"{module}.{caller}:{detail}")
    raise DirectOnlyComplianceError(
        f"spectral/nuclear norm route {detail} is prohibited"
    )


def _dimension_kind(value: Any, dimension: Any) -> str:
    if dimension is None:
        rank = getattr(value, "ndim", None)
        if rank == 1:
            return "vector"
        if rank == 2:
            return "matrix"
        return "flattened"
    if isinstance(dimension, int):
        return "vector"
    if isinstance(dimension, (tuple, list)):
        if len(dimension) == 1:
            return "vector"
        if len(dimension) == 2:
            return "matrix"
    return "unsupported"


def _torch_linalg_norm_wrapper(vector_norm: Any, qualified_name: str):
    """Implement only Frobenius/vector routes using ``vector_norm``."""

    def controlled(*args: Any, **kwargs: Any):
        positional = list(args)
        keywords = dict(kwargs)
        value = _norm_input(positional, keywords, ("A", "input"))
        order = _take_argument(positional, keywords, "ord", None)
        dimension = _take_argument(positional, keywords, "dim", None)
        keepdim = _take_argument(positional, keywords, "keepdim", False)
        if positional:
            raise DirectOnlyComplianceError("too many torch.linalg.norm arguments")
        output = keywords.pop("out", _MISSING)
        dtype = keywords.pop("dtype", _MISSING)
        if keywords:
            raise DirectOnlyComplianceError(
                f"unsupported torch.linalg.norm arguments: {sorted(keywords)}"
            )

        kind = _dimension_kind(value, dimension)
        if order == "nuc" or (
            kind == "matrix" and _forbidden_spectral_order(order)
        ):
            module, caller = _caller_identity()
            _raise_prohibited_norm(qualified_name, order, module, caller)
        if kind == "unsupported":
            raise DirectOnlyComplianceError("ambiguous torch.linalg.norm dimensions")
        if kind == "flattened" and order is not None:
            raise DirectOnlyComplianceError(
                "an explicit norm order requires a vector or matrix dimension"
            )
        if kind == "matrix" and not _frobenius_order(order):
            raise DirectOnlyComplianceError(
                "only Frobenius matrix norms are allowed in canonical ODT"
            )
        if kind in {"vector", "flattened"} and isinstance(order, str):
            raise DirectOnlyComplianceError(
                "string-valued vector norm orders are not allowed"
            )
        safe_order = 2 if _frobenius_order(order) else order
        safe_keywords: dict[str, Any] = {
            "ord": safe_order,
            "dim": dimension,
            "keepdim": keepdim,
        }
        if output is not _MISSING:
            safe_keywords["out"] = output
        if dtype is not _MISSING:
            safe_keywords["dtype"] = dtype
        return vector_norm(value, **safe_keywords)

    return controlled


def _torch_legacy_norm_wrapper(
    vector_norm: Any, qualified_name: str, *, tensor_method: bool
):
    """Preserve legacy Frobenius/vector norms without retaining ``torch.norm``."""

    def controlled(*args: Any, **kwargs: Any):
        positional = list(args)
        keywords = dict(kwargs)
        value = _norm_input(positional, keywords, ("input", "self"))
        order = _take_argument(positional, keywords, "p", "fro")
        dimension = _take_argument(positional, keywords, "dim", None)
        keepdim = _take_argument(positional, keywords, "keepdim", False)
        if order == "nuc":
            module, caller = _caller_identity()
            _raise_prohibited_norm(qualified_name, order, module, caller)
        if isinstance(order, str) and order != "fro":
            raise DirectOnlyComplianceError(
                "only Frobenius or numeric legacy norm orders are allowed"
            )
        output = _MISSING
        if not tensor_method:
            output = _take_argument(positional, keywords, "out", _MISSING)
        dtype = _take_argument(positional, keywords, "dtype", _MISSING)
        if positional or keywords:
            raise DirectOnlyComplianceError("unsupported legacy norm arguments")
        safe_keywords: dict[str, Any] = {
            "ord": 2 if _frobenius_order(order) else order,
            "dim": dimension,
            "keepdim": keepdim,
        }
        if output is not _MISSING:
            safe_keywords["out"] = output
        if dtype is not _MISSING:
            safe_keywords["dtype"] = dtype
        return vector_norm(value, **safe_keywords)

    return controlled


def _numpy_linalg_norm_wrapper(numpy_module: Any, qualified_name: str):
    """Implement NumPy Frobenius/vector norms without retaining linalg.norm."""

    def controlled(*args: Any, **kwargs: Any):
        positional = list(args)
        keywords = dict(kwargs)
        value = _norm_input(positional, keywords, ("x",))
        order = _take_argument(positional, keywords, "ord", None)
        axis = _take_argument(positional, keywords, "axis", None)
        keepdims = _take_argument(positional, keywords, "keepdims", False)
        if positional or keywords:
            raise DirectOnlyComplianceError("unsupported numpy.linalg.norm arguments")
        array = numpy_module.asarray(value)
        kind = _dimension_kind(array, axis)
        if order == "nuc" or (
            kind == "matrix" and _forbidden_spectral_order(order)
        ):
            module, caller = _caller_identity()
            _raise_prohibited_norm(qualified_name, order, module, caller)
        if kind == "unsupported":
            raise DirectOnlyComplianceError("ambiguous numpy.linalg.norm axes")
        if kind == "flattened" and order is not None:
            raise DirectOnlyComplianceError(
                "an explicit norm order requires a vector or matrix axis"
            )
        if kind == "matrix" and not _frobenius_order(order):
            raise DirectOnlyComplianceError(
                "only Frobenius matrix norms are allowed in canonical ODT"
            )
        if kind in {"vector", "flattened"} and isinstance(order, str):
            raise DirectOnlyComplianceError(
                "string-valued vector norm orders are not allowed"
            )
        absolute = numpy_module.abs(array)
        safe_order = 2 if _frobenius_order(order) else order
        if not isinstance(safe_order, (int, float)) or isinstance(safe_order, bool):
            raise DirectOnlyComplianceError("unsupported numpy vector norm order")
        if safe_order == float("inf"):
            return numpy_module.max(absolute, axis=axis, keepdims=keepdims)
        if safe_order == float("-inf"):
            return numpy_module.min(absolute, axis=axis, keepdims=keepdims)
        if safe_order == 0:
            return numpy_module.sum(absolute != 0, axis=axis, keepdims=keepdims)
        return numpy_module.sum(
            absolute ** safe_order, axis=axis, keepdims=keepdims
        ) ** (1.0 / safe_order)

    return controlled


def _live_runtime_entrypoint(qualified_name: str) -> Any:
    import numpy
    import torch

    if qualified_name.startswith("numpy.linalg.linalg."):
        owner = numpy.linalg.linalg
        name = qualified_name.removeprefix("numpy.linalg.linalg.")
    elif qualified_name.startswith("numpy.linalg."):
        owner = numpy.linalg
        name = qualified_name.removeprefix("numpy.linalg.")
    elif qualified_name.startswith("torch.functional."):
        owner = torch.functional
        name = qualified_name.removeprefix("torch.functional.")
    elif qualified_name.startswith("torch.linalg."):
        owner = torch.linalg
        name = qualified_name.removeprefix("torch.linalg.")
    elif qualified_name.startswith("torch.Tensor."):
        owner = torch.Tensor
        name = qualified_name.removeprefix("torch.Tensor.")
    elif qualified_name.startswith("torch."):
        owner = torch
        name = qualified_name.removeprefix("torch.")
    else:
        raise DirectOnlyComplianceError(
            f"unknown direct-only runtime entrypoint {qualified_name}"
        )
    if not hasattr(owner, name):
        raise DirectOnlyComplianceError(
            f"required direct-only runtime entrypoint is unavailable: {qualified_name}"
        )
    return getattr(owner, name)


def assert_direct_only_runtime_guard(
    report: dict[str, Any] | None = None,
    *,
    exact_allowed_calls: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail unless the exact guard surface and runtime ledger are intact.

    ``exact_allowed_calls`` constrains the distinct semantic call identities,
    while ``allowed_call_count`` may be larger because repeated QR/EVD calls
    are expected.  This function inspects the live wrappers as well as the
    ledger so replacing a wrapper after installation cannot fail open.
    """

    snapshot = direct_only_runtime_report() if report is None else report
    errors: list[str] = []
    if snapshot.get("installed") is not True:
        errors.append("runtime guard is not installed")

    patched_raw = snapshot.get("patched_entrypoints", ())
    patched = frozenset(patched_raw) if isinstance(patched_raw, (list, tuple)) else frozenset()
    if patched != EXPECTED_RUNTIME_GUARD_ENTRYPOINTS:
        missing = sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS - patched)
        extra = sorted(patched - EXPECTED_RUNTIME_GUARD_ENTRYPOINTS)
        errors.append(f"guard surface differs: missing={missing}, extra={extra}")
    if snapshot.get("patched_entrypoint_count") != EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT:
        errors.append(
            "guard count differs: "
            f"{snapshot.get('patched_entrypoint_count')!r} != "
            f"{EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT}"
        )

    registered = frozenset(_RUNTIME_WRAPPERS)
    if registered != EXPECTED_RUNTIME_GUARD_ENTRYPOINTS:
        errors.append(
            "private wrapper registry differs: "
            f"missing={sorted(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS - registered)}, "
            f"extra={sorted(registered - EXPECTED_RUNTIME_GUARD_ENTRYPOINTS)}"
        )
    missing_live_wrappers = sorted(
        qualified_name
        for qualified_name in EXPECTED_RUNTIME_GUARD_ENTRYPOINTS
        if _live_runtime_entrypoint(qualified_name)
        is not _RUNTIME_WRAPPERS.get(qualified_name)
    )
    if missing_live_wrappers:
        errors.append(f"live guard wrappers are missing: {missing_live_wrappers}")
    wrapped_escape_hatches = sorted(
        qualified_name
        for qualified_name, wrapper in _RUNTIME_WRAPPERS.items()
        if hasattr(wrapper, "__wrapped__")
    )
    if wrapped_escape_hatches:
        errors.append(
            f"runtime wrappers expose __wrapped__: {wrapped_escape_hatches}"
        )
    blocked_wrappers_retaining_callables = sorted(
        qualified_name
        for qualified_name, wrapper in _RUNTIME_WRAPPERS.items()
        if wrapper.__name__ == "blocked"
        and any(
            callable(cell.cell_contents)
            for cell in (wrapper.__closure__ or ())
        )
    )
    if blocked_wrappers_retaining_callables:
        errors.append(
            "blocked wrappers retain callable originals: "
            f"{blocked_wrappers_retaining_callables}"
        )

    prohibited_attempts = snapshot.get("prohibited_attempts", ())
    if not isinstance(prohibited_attempts, (list, tuple)):
        errors.append("prohibited-attempt ledger is malformed")
        prohibited_attempts = ()
    if snapshot.get("prohibited_attempt_count") != len(prohibited_attempts):
        errors.append("prohibited-attempt count disagrees with its ledger")
    if prohibited_attempts:
        errors.append(f"prohibited numerical routes were attempted: {prohibited_attempts}")

    allowed_raw = snapshot.get("allowed_calls", ())
    allowed = frozenset(allowed_raw) if isinstance(allowed_raw, (list, tuple)) else frozenset()
    unexpected = sorted(allowed - ALLOWED_RUNTIME_CALLS)
    if unexpected:
        errors.append(f"unexpected controlled numerical callers: {unexpected}")
    allowed_count = snapshot.get("allowed_call_count")
    if not isinstance(allowed_count, int) or allowed_count < len(allowed):
        errors.append("allowed-call count is malformed or smaller than its distinct ledger")
    if exact_allowed_calls is not None:
        expected_calls = frozenset(exact_allowed_calls)
        if not expected_calls <= ALLOWED_RUNTIME_CALLS:
            errors.append("requested exact call set contains a non-allowlisted identity")
        if allowed != expected_calls:
            errors.append(
                "distinct runtime call set differs: "
                f"missing={sorted(expected_calls - allowed)}, "
                f"extra={sorted(allowed - expected_calls)}"
            )
        if not expected_calls and allowed_count != 0:
            errors.append("import-phase runtime call count is not zero")

    if errors:
        raise DirectOnlyComplianceError("; ".join(errors))
    return snapshot


def _patch(owner: Any, name: str, wrapper_factory: Any, qualified_name: str) -> None:
    if not hasattr(owner, name):
        return
    if qualified_name in _RUNTIME_WRAPPERS:
        if getattr(owner, name) is _RUNTIME_WRAPPERS[qualified_name]:
            return
        raise DirectOnlyComplianceError(
            f"runtime wrapper was replaced during installation: {qualified_name}"
        )
    original = getattr(owner, name)
    wrapper = wrapper_factory(original, qualified_name)
    setattr(owner, name, wrapper)
    _RUNTIME_WRAPPERS[qualified_name] = wrapper
    _RUNTIME_STATE["patched_entrypoints"].append(qualified_name)


def install_direct_only_runtime_guard() -> dict[str, Any]:
    """Install process-wide guards before importing any ODT model code."""

    if _RUNTIME_STATE["installed"]:
        return assert_direct_only_runtime_guard(direct_only_runtime_report())

    import numpy
    import torch

    discard_original = lambda _original, qualified: _prohibited_wrapper(qualified)
    for name in sorted(_PROHIBITED_TERMINALS):
        _patch(torch.linalg, name, discard_original, f"torch.linalg.{name}")
        _patch(torch, name, discard_original, f"torch.{name}")
        _patch(numpy.linalg, name, discard_original, f"numpy.linalg.{name}")
        _patch(torch.Tensor, name, discard_original, f"torch.Tensor.{name}")
        _patch(
            numpy.linalg.linalg,
            name,
            discard_original,
            f"numpy.linalg.linalg.{name}",
        )

    # NumPy eigensolvers are never part of canonical Algorithm 3.  Patch both
    # public namespaces separately because NumPy 1.21 retains independent
    # attribute bindings to the same original callables.
    for name in ("eig", "eigh", "eigvals", "eigvalsh"):
        _patch(numpy.linalg, name, discard_original, f"numpy.linalg.{name}")
        _patch(
            numpy.linalg.linalg,
            name,
            discard_original,
            f"numpy.linalg.linalg.{name}",
        )

    _patch(
        torch.linalg,
        "qr",
        lambda original, name: _controlled_wrapper(original, name, _DIRECT_QR_CALLERS),
        "torch.linalg.qr",
    )
    _patch(
        torch.linalg,
        "solve_triangular",
        lambda original, name: _controlled_wrapper(original, name, _TRIANGULAR_CALLERS),
        "torch.linalg.solve_triangular",
    )
    for name in ("eigh", "eigvalsh", "eig", "eigvals"):
        _patch(
            torch.linalg,
            name,
            lambda original, qualified: _controlled_wrapper(
                original, qualified, _ALGORITHM3_EVD_CALLERS
            ),
            f"torch.linalg.{name}",
        )
    _patch(
        torch.linalg,
        "matrix_norm",
        lambda _original, qualified: _prohibited_wrapper(qualified),
        "torch.linalg.matrix_norm",
    )
    _patch(
        torch.linalg,
        "norm",
        lambda _original, qualified: _torch_linalg_norm_wrapper(
            torch.linalg.vector_norm, qualified
        ),
        "torch.linalg.norm",
    )
    _patch(
        numpy.linalg,
        "norm",
        lambda _original, qualified: _numpy_linalg_norm_wrapper(numpy, qualified),
        "numpy.linalg.norm",
    )
    _patch(
        numpy.linalg.linalg,
        "norm",
        lambda _original, qualified: _numpy_linalg_norm_wrapper(numpy, qualified),
        "numpy.linalg.linalg.norm",
    )
    _patch(
        torch,
        "norm",
        lambda _original, qualified: _torch_legacy_norm_wrapper(
            torch.linalg.vector_norm, qualified, tensor_method=False
        ),
        "torch.norm",
    )
    _patch(
        torch.functional,
        "norm",
        lambda _original, qualified: _torch_legacy_norm_wrapper(
            torch.linalg.vector_norm, qualified, tensor_method=False
        ),
        "torch.functional.norm",
    )
    _patch(
        torch.Tensor,
        "norm",
        lambda _original, qualified: _torch_legacy_norm_wrapper(
            torch.linalg.vector_norm, qualified, tensor_method=True
        ),
        "torch.Tensor.norm",
    )

    _RUNTIME_STATE["installed"] = True
    return assert_direct_only_runtime_guard(
        direct_only_runtime_report(), exact_allowed_calls=()
    )


def direct_only_runtime_report() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the guard ledger."""

    return {
        "installed": bool(_RUNTIME_STATE["installed"]),
        "patched_entrypoints": sorted(set(_RUNTIME_STATE["patched_entrypoints"])),
        "patched_entrypoint_count": len(set(_RUNTIME_STATE["patched_entrypoints"])),
        "allowed_call_count": len(_RUNTIME_STATE["allowed_calls"]),
        "allowed_calls": sorted(set(_RUNTIME_STATE["allowed_calls"])),
        "prohibited_attempt_count": len(_RUNTIME_STATE["prohibited_attempts"]),
        "prohibited_attempts": tuple(_RUNTIME_STATE["prohibited_attempts"]),
    }


__all__ = [
    "DirectOnlyComplianceError",
    "ALLOWED_RUNTIME_CALLS",
    "DIRECT_RQ_RUNTIME_CALL",
    "EXPECTED_RUNTIME_GUARD_ENTRYPOINTS",
    "EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT",
    "PRODUCTION_ALGORITHM3_RUNTIME_CALL",
    "PRODUCTION_CLONE_QR_RUNTIME_CALL",
    "PRODUCTION_SHARED_CLONE_ALGORITHM3_RUNTIME_CALL",
    "REFERENCE_ALGORITHM3_RUNTIME_CALL",
    "REFERENCE_CLONE_QR_RUNTIME_CALL",
    "STREAMED_CLONE_ORACLE_RUNTIME_CALLS",
    "STREAMED_QR_RUNTIME_CALL",
    "TRIANGULAR_RUNTIME_CALL",
    "assert_direct_only_runtime_guard",
    "audit_direct_only_launch",
    "canonical_direct_only_entrypoints",
    "direct_only_runtime_report",
    "install_direct_only_runtime_guard",
    "single_cli_option",
    "transitive_local_sources",
]
