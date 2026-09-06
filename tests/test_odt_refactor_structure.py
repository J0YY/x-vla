"""Import-free architectural gates for the production ODT module split.

These tests parse source only. They do not instantiate a model or run numerical
legacy code. The project-source closure includes package initializers and local
dependencies, not the internal implementation of external tensor libraries.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "xvla/train/odt_engine_v2"
FACADE = ROOT / "xvla/train/implicit_sparse_projective_odt.py"
MODULES = ("core", "compiler", "validation", "diagnostics", "oracles", "types", "constants",
           "ops", "factorization", "graph", "compat", "__init__")
PROHIBITED = {"svd", "svdvals", "pinv", "pinverse", "lstsq", "cholesky", "polar",
              "matrix_rank", "matrix_power", "inv", "inverse", "cov", "covariance",
              "gram", "normal_equations"}
EVD = {"eig", "eigh", "eigvals", "eigvalsh"}
PRIMARY = {"canonicalize_implicit_dag_direct_rq", "reverse_implicit_environments",
           "diagonalize_implicit_dag_full_rank"}


def source_tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def module_path(name):
    base = ROOT.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file() and (base / "__init__.py").is_file():
        raise AssertionError(f"ambiguous module/package collision: {name}")
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def imported_modules(tree, path):
    parts = path.relative_to(ROOT).with_suffix("").parts
    package = list(parts[:-1])
    dynamic_aliases = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            dynamic_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package)-node.level+1] if node.level else []
            base = ".".join(prefix + ([node.module] if node.module else []))
            if base:
                yield base
            for alias in node.names:
                if alias.name != "*":
                    yield base + "." + alias.name if base else alias.name
        elif isinstance(node, ast.Call) and node.args:
            target = ast.unparse(node.func)
            if target in dynamic_aliases and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value


def project_source_closure(roots):
    """Follow every static local import and the package initializers it loads."""
    pending, found = list(roots), set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for parent in path.parents:
            if parent == ROOT:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer not in found:
                pending.append(initializer)
        for name in imported_modules(source_tree(path), path):
            candidate = module_path(name)
            if candidate is not None and candidate not in found:
                pending.append(candidate)
    return found


def numerical_references(tree):
    """Resolve direct import aliases and constant-name getattr dispatches."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module or "") + "." + alias.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return resolve(node.value) + "." + node.attr
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                return resolve(node.args[0]) + "." + node.args[1].value
        return ""

    references = set(aliases.values())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Name)):
            references.add(resolve(node))
        if isinstance(node, ast.Call):
            references.add(resolve(node.func))
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                references.add(resolve(node))
    return {name for name in references if name.rsplit(".", 1)[-1] in PROHIBITED}


def function_arguments(function):
    args = function.args
    return {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


class ODTRefactorStructureTests(unittest.TestCase):
    def test_required_modules_and_thin_legacy_facade(self):
        self.assertFalse(PACKAGE.with_suffix(".py").exists(), "canonical package must not shadow a legacy module")
        for name in MODULES:
            self.assertTrue((PACKAGE / (name + ".py")).is_file(), name)
        tree = source_tree(FACADE)
        definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        self.assertLessEqual(set(definitions), {"__getattr__", "__dir__"})
        numerical_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr in EVD | {"qr"}]
        self.assertEqual(numerical_calls, [])

    def test_primary_entry_points_live_in_core_without_fault_or_metric_switches(self):
        tree = source_tree(PACKAGE / "core.py")
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertLessEqual(PRIMARY, functions.keys())
        for name in PRIMARY:
            parameters = function_arguments(functions[name])
            self.assertNotIn("output_metric", parameters, name)
            self.assertFalse(any("omit" in p or "fault" in p for p in parameters), (name, parameters))
        # Injection belongs in oracle code, not in a private primary helper.
        for function in functions.values():
            self.assertFalse(any("omit" in p or "fault" in p for p in function_arguments(function)), function.name)

    def test_retired_public_oracle_exports_have_an_explicit_migration(self):
        tree = source_tree(FACADE)
        exports = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets))
        imported = {alias.asname or alias.name for node in tree.body
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        migration = (PACKAGE / "README.md").read_text()
        for retired in ("CloneEVDTrace", "apply_shared_eigenbases_to_explicit_clone_occurrences_control"):
            self.assertNotIn(retired, exports)
            self.assertNotIn(retired, imported)
            self.assertIn(retired, migration)
        for replacement in ("IndependentCloneEVDTrace", "diagonalize_shared_and_explicit_clone_independently"):
            self.assertIn(replacement, exports)
            self.assertIn(replacement, imported)
            self.assertIn(replacement, migration)

    def test_production_project_closure_has_no_oracle_or_testing_dependency(self):
        roots = [PACKAGE / (name + ".py") for name in ("__init__", "core", "compiler")]
        closure = project_source_closure(roots)
        forbidden = [str(path.relative_to(ROOT)) for path in closure if path == FACADE
                     or path.name in {"oracles.py", "compat.py"} or "tests" in path.parts or "research" in path.parts
                     or "clone_reference" in path.stem]
        self.assertEqual(forbidden, [])
        self.assertIn(ROOT / "xvla/nn/normalization.py", closure)
        self.assertIn(ROOT / "xvla/__init__.py", closure)

    def test_prohibited_numerics_absent_from_entire_project_source_closure(self):
        roots = [PACKAGE / (name + ".py") for name in ("__init__", "core", "compiler", "validation", "diagnostics")]
        violations = {str(path.relative_to(ROOT)): sorted(numerical_references(source_tree(path)))
                      for path in project_source_closure(roots)}
        self.assertEqual({path: names for path, names in violations.items() if names}, {})

    def test_exactly_one_primary_evd_call_site_owned_by_core(self):
        roots = [PACKAGE / (name + ".py") for name in ("__init__", "core", "compiler", "validation", "diagnostics")]
        sites = []
        for path in project_source_closure(roots):
            for node in ast.walk(source_tree(path)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in EVD:
                    sites.append((path, node.func.attr))
        self.assertEqual(sites, [(PACKAGE / "core.py", "eigh")])

    def test_static_scanner_catches_aliases_without_importing_them(self):
        examples = ("from torch.linalg import svd as disguised\ndisguised(x)",
                    "import numpy as n\nf = n.linalg.pinv\nf(x)",
                    "import torch as t\ngetattr(t.linalg, 'lstsq')(a,b)",
                    "from scipy.linalg import polar as helper\nhelper(x)")
        for source in examples:
            with self.subTest(source=source):
                self.assertTrue(numerical_references(ast.parse(source)))
        self.assertEqual(numerical_references(ast.parse("import torch\nq,r=torch.linalg.qr(x)")), set())

    def test_closure_resolves_relative_and_literal_dynamic_imports(self):
        source = "from . import validation\nfrom .. import sibling\nfrom importlib import import_module as load\nload('xvla.nn.tree_mixing')"
        imports = set(imported_modules(ast.parse(source), PACKAGE / "core.py"))
        self.assertIn("xvla.train.odt_engine_v2.validation", imports)
        self.assertIn("xvla.train.sibling", imports)
        self.assertIn("xvla.nn.tree_mixing", imports)

    def test_symmetry_is_declared_at_compile_and_checked_before_qr(self):
        compiler = source_tree(PACKAGE / "compiler.py")
        builder = next(node for node in compiler.body if isinstance(node, ast.ClassDef) and node.name == "_Builder")
        cp = next(node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "cp")
        tied = next(node for node in cp.body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "children[0] is children[1]")
        assigned = {name.id for node in tied.body for target in getattr(node, "targets", [])
                    for name in ast.walk(target) if isinstance(name, ast.Name)}
        self.assertLessEqual({"output", "left", "right"}, assigned)
        constructed = [node.lineno for node in ast.walk(cp) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "_make_cp_core"]
        self.assertLess(tied.end_lineno, min(constructed))
        factorization = source_tree(PACKAGE / "factorization.py")
        bounded = next(node for node in factorization.body if isinstance(node, ast.FunctionDef) and node.name == "_direct_rq_bounded_cp")
        calls = {node.func.id: node.lineno for node in ast.walk(bounded)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertLess(calls["_validate_tied_input_symmetry"], calls["_positive_diagonal_direct_rq_rows"])
        self.assertIn("_packed_cp_unfolding", calls)
        self.assertIn("_unpack_symmetric_q", calls)
        core = source_tree(PACKAGE / "core.py")
        sweep = next(node for node in core.body if isinstance(node, ast.FunctionDef) and node.name == "canonicalize_implicit_dag_direct_rq")
        calls = {node.func.id: node.lineno for node in ast.walk(sweep)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertLess(calls["_validate_symmetric_lift"], calls["_direct_rq_core"])

    def test_retired_streamed_route_fails_before_any_numerical_call(self):
        tree = source_tree(PACKAGE / "factorization.py")
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("_solve_compact_q", functions)
        self.assertNotIn("_direct_tsqr_panel", functions)
        streamed = functions["_direct_rq_cp_streamed"]
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(streamed)))
        calls = {ast.unparse(node.func) for node in ast.walk(streamed) if isinstance(node, ast.Call)}
        self.assertLessEqual(calls, {"_binary_columns", "ValueError"})


if __name__ == "__main__":
    unittest.main()
