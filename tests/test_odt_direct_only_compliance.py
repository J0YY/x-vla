"""Regression gates for fit-free Padé construction and launch compliance."""

from __future__ import annotations

import ast
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT,
    EXPECTED_RUNTIME_GUARD_ENTRYPOINTS,
    assert_direct_only_runtime_guard,
    audit_direct_only_launch,
    canonical_direct_only_entrypoints,
    transitive_local_sources,
)
from xvla.nn.normalization import RationalNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_implicit_sparse_projective_odt.py"
VLA_RUNNER = PROJECT_ROOT / "scripts/run_implicit_sparse_projective_odt_vla.py"
ALL_TOKENS_RUNNER = (
    PROJECT_ROOT / "scripts/run_implicit_sparse_projective_odt_all_tokens.py"
)
CLONE_RUNNER = PROJECT_ROOT / "scripts/run_direct_odt_clone_oracle.py"


def _float32_bits(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        struct.unpack(">I", struct.pack(">f", float(value)))[0]
        for value in values.detach().cpu()
    )


def test_default_pade_coefficients_are_checkpoint_exact_and_fit_free():
    norm = RationalNorm(variant="pade")
    assert _float32_bits(norm.pa) == (0x40AA8F16, 0x40F122F9, 0x3EA7906C)
    assert _float32_bits(norm.pb) == (0x3F800000, 0x411A5F50, 0x4026D326)

    normalization_source = (PROJECT_ROOT / "xvla/nn/normalization.py").read_text()
    tree = ast.parse(normalization_source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "numpy" not in imports
    assert ".lstsq(" not in normalization_source


def test_nondefault_pade_requires_explicit_valid_coefficients():
    with pytest.raises(ValueError, match="no checked-in"):
        RationalNorm(variant="pade", deg=3)
    with pytest.raises(ValueError, match="both numerator and denominator"):
        RationalNorm(variant="pade", deg=1, pade_numerator=(1.0, 0.5))
    with pytest.raises(ValueError, match="2 entries"):
        RationalNorm(
            variant="pade",
            deg=1,
            pade_numerator=(1.0,),
            pade_denominator=(1.0, 0.5),
        )
    with pytest.raises(ValueError, match="finite"):
        RationalNorm(
            variant="pade",
            deg=1,
            pade_numerator=(1.0, float("nan")),
            pade_denominator=(1.0, 0.5),
        )
    custom = RationalNorm(
        variant="pade",
        deg=1,
        pade_numerator=(1.0, 0.5),
        pade_denominator=(1.0, 0.25),
    )
    assert custom.pa.shape == custom.pb.shape == (2,)


def test_pade_buffers_still_load_from_a_checkpoint_state_dict():
    source = RationalNorm(
        variant="pade",
        deg=1,
        pade_numerator=(0.75, 0.125),
        pade_denominator=(1.0, -0.0625),
    )
    source.running_ms.fill_(1.75)
    source.initialized.fill_(True)
    target = RationalNorm(
        variant="pade",
        deg=1,
        pade_numerator=(1.0, 0.0),
        pade_denominator=(1.0, 0.0),
    )
    target.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(target.pa, source.pa, rtol=0, atol=0)
    torch.testing.assert_close(target.pb, source.pb, rtol=0, atol=0)
    torch.testing.assert_close(target.running_ms, source.running_ms, rtol=0, atol=0)
    assert bool(target.initialized)


def test_transitive_authoritative_launch_source_audit_is_green_and_complete():
    report = audit_direct_only_launch(
        PROJECT_ROOT, canonical_direct_only_entrypoints(PROJECT_ROOT, RUNNER)
    )
    audited = set(report["source_sha256"])
    assert report["prohibited_calls_found"] == []
    assert report["direct_qr_call_sites"] == 4
    assert {
        "scripts/run_direct_odt_clone_oracle.py",
        "scripts/run_implicit_sparse_projective_odt.py",
        "scripts/odt_direct_only_compliance.py",
        "xvla/nn/normalization.py",
        "xvla/models/vla.py",
        "xvla/models/vit.py",
        "xvla/train/implicit_sparse_projective_odt.py",
        "xvla/train/implicit_sparse_projective_odt_all_tokens.py",
        "xvla/train/implicit_sparse_projective_odt_vla.py",
        "tests/test_implicit_sparse_projective_odt.py",
        "tests/test_implicit_sparse_projective_odt_all_tokens.py",
        "tests/test_implicit_sparse_projective_odt_heterogeneous.py",
        "tests/test_implicit_sparse_projective_odt_vla.py",
        "tests/test_direct_odt_clone_reference.py",
        "xvla/train/direct_odt_clone_reference.py",
    } <= audited
    assert "tests/test_odt_direct_only_compliance.py" not in audited
    assert len(report["guarded_dormant_spectral_norm_sites"]) == 3


def test_every_canonical_runner_has_the_same_authenticated_source_closure():
    closures = []
    for runner in (RUNNER, VLA_RUNNER, ALL_TOKENS_RUNNER, CLONE_RUNNER):
        report = audit_direct_only_launch(
            PROJECT_ROOT,
            canonical_direct_only_entrypoints(PROJECT_ROOT, runner),
        )
        assert report["prohibited_calls_found"] == []
        closures.append(report["source_sha256"])
    assert all(closure == closures[0] for closure in closures[1:])


def test_every_canonical_runner_uses_the_central_exact_runtime_guard():
    for runner in (RUNNER, VLA_RUNNER, ALL_TOKENS_RUNNER, CLONE_RUNNER):
        source = runner.read_text()
        assert "assert_direct_only_runtime_guard(" in source
        assert "exact_allowed_calls=" in source
        assert "ArgumentParser(allow_abbrev=False)" in source
        assert '["patched_entrypoint_count"] > 0' not in source


def test_vla_manifest_verification_precedes_every_project_import():
    source = VLA_RUNNER.read_text()
    tree = ast.parse(source)
    project_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            name.startswith(("scripts", "xvla"))
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
        )
    ]
    verification_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name)
            and target.id == "_PREIMPORT_SOURCE_VERIFICATION"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    )
    assert project_import_lines
    assert verification_line < min(project_import_lines)
    assert "from scripts.odt_direct_only_compliance import single_cli_option" not in source


def test_all_token_checkpoint_identity_is_fixed_in_source():
    source = ALL_TOKENS_RUNNER.read_text()
    assert "--expected-checkpoint-sha256" not in source
    assert "checkpoint_sha != EXPECTED_CHECKPOINT_SHA256" in source
    assert '"synthetic_tiny_uncheckpointed"' in source


@pytest.mark.parametrize("runner", (RUNNER, VLA_RUNNER, ALL_TOKENS_RUNNER))
def test_duplicate_lane_flags_fail_before_numerical_execution(runner: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--lane",
            "tiny",
            "--lane=tiny",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--lane may be specified at most once" in completed.stderr


@pytest.mark.parametrize(
    "source",
    (
        "import torch.linalg as la\nfactor = la.svd\nfactor(x)\n",
        "def harmless(x):\n    return x\nfactor = harmless\nimport torch\nfactor = torch.linalg.svd\nfactor(x)\n",
        "def harmless(x):\n    return x\nfactor = harmless\nimport torch\nfactor = torch._VF.norm\nfactor(x, p='nuc')\n",
        "def harmless(x):\n    return x\nfactor = harmless\nimport torch\nfactor = torch.linalg.matrix_norm\nfactor(x, ord=2)\n",
        "from numpy.linalg import lstsq as fit\nfit(a, b)\n",
        "import torch\ndef f(x):\n    return getattr(torch.linalg, 'pinv')(x)\n",
        "import torch\ndef wrong(x):\n    return torch.linalg.qr(x)\n",
        "import torch\ndef algorithm2(x):\n    return torch.linalg.eigh(x)\n",
        "import scipy.linalg\ndef diagnostic(x):\n    return scipy.linalg.norm(x, 2)\n",
        "import torch\ndef diagnostic(x):\n    return torch.ops.aten._linalg_svd.default(x)\n",
        "import torch\ndef diagnostic(x):\n    return torch.ops.aten.linalg_svd.default(x)\n",
        "def diagnostic(q):\n    return q @ q.T\n",
        "def diagnostic(q):\n    return q.transpose(-1, -2) @ q\n",
        "import torch\ndef diagnostic(q):\n    return torch.einsum('oi,pi->op', q, q)\n",
        "import torch\ndef _role_environment(q):\n    return torch.einsum('oi,pi->op', q, q)\n",
    ),
)
def test_static_audit_rejects_aliases_dynamic_access_and_wrong_stage_calls(
    tmp_path: Path, source: str
):
    entrypoint = tmp_path / "bad_launch.py"
    entrypoint.write_text(source)
    # The public transitive helper intentionally requires sources under the
    # chosen root.  This exercises the same auditor without executing any
    # prohibited numerical routine.
    with pytest.raises(DirectOnlyComplianceError):
        audit_direct_only_launch(tmp_path, (entrypoint,))


def test_transitive_source_discovery_follows_relative_package_imports(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("from . import helper\n")
    entrypoint = package / "entry.py"
    entrypoint.write_text("from .helper import value\n")
    helper = package / "helper.py"
    helper.write_text("value = 1\n")
    closure = transitive_local_sources(tmp_path, (entrypoint,))
    assert {path.relative_to(tmp_path).as_posix() for path in closure} == {
        "package/__init__.py",
        "package/entry.py",
        "package/helper.py",
    }


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    return environment


def test_runtime_guard_installs_without_exposing_wrapped_originals():
    program = """
import json
from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    _RUNTIME_WRAPPERS,
    _prohibited_wrapper,
    assert_direct_only_runtime_guard,
    install_direct_only_runtime_guard,
)
installed = assert_direct_only_runtime_guard(
    install_direct_only_runtime_guard(), exact_allowed_calls=()
)
blocked_sentinel = _prohibited_wrapper("sentinel.route")
blocked = 0
try:
    blocked_sentinel()
except DirectOnlyComplianceError:
    blocked = 1
print(json.dumps({
    "blocked": blocked,
    "patched_entrypoint_count": installed["patched_entrypoint_count"],
    "registry_count": len(_RUNTIME_WRAPPERS),
    "has_wrapped_escape_hatch": any(
        hasattr(wrapper, "__wrapped__") for wrapper in _RUNTIME_WRAPPERS.values()
    ),
    "blocked_wrapper_retains_callable": any(
        callable(cell.cell_contents)
        for wrapper in _RUNTIME_WRAPPERS.values()
        if wrapper.__name__ == "blocked"
        for cell in (wrapper.__closure__ or ())
    ),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["blocked"] == 1
    assert result["patched_entrypoint_count"] == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
    assert result["registry_count"] == EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT
    assert not result["has_wrapped_escape_hatch"]
    assert not result["blocked_wrapper_retains_callable"]


def test_runtime_guard_surface_constants_are_exact_and_live_wrappers_fail_closed():
    assert len(EXPECTED_RUNTIME_GUARD_ENTRYPOINTS) == 87
    assert EXPECTED_RUNTIME_GUARD_ENTRYPOINT_COUNT == 87
    program = """
import json
import torch
from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    assert_direct_only_runtime_guard,
    install_direct_only_runtime_guard,
)
report = assert_direct_only_runtime_guard(
    install_direct_only_runtime_guard(), exact_allowed_calls=()
)
original = torch.linalg.qr
torch.linalg.qr = lambda *args, **kwargs: None
failed_closed = False
try:
    assert_direct_only_runtime_guard(report, exact_allowed_calls=())
except DirectOnlyComplianceError:
    failed_closed = True
finally:
    torch.linalg.qr = original
print(json.dumps({
    "failed_closed": failed_closed,
    "patched_entrypoint_count": report["patched_entrypoint_count"],
    "allowed_call_count": report["allowed_call_count"],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "failed_closed": True,
        "patched_entrypoint_count": 87,
        "allowed_call_count": 0,
    }


def test_norm_dispatch_blocks_spectral_and_nuclear_before_safe_sentinels():
    program = """
import json
from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    _numpy_linalg_norm_wrapper,
    _prohibited_wrapper,
    _torch_legacy_norm_wrapper,
    _torch_linalg_norm_wrapper,
)
calls = []
def safe_vector_norm_sentinel(*args, **kwargs):
    calls.append((args, kwargs))
    return "safe"
class Matrix:
    ndim = 2
class FakeNumpy:
    @staticmethod
    def asarray(value):
        return value
    @staticmethod
    def abs(value):
        calls.append("numpy_abs")
        return value
linalg_norm = _torch_linalg_norm_wrapper(
    safe_vector_norm_sentinel, "sentinel.linalg_norm"
)
legacy_norm = _torch_legacy_norm_wrapper(
    safe_vector_norm_sentinel, "sentinel.legacy_norm", tensor_method=False
)
numpy_norm = _numpy_linalg_norm_wrapper(FakeNumpy, "sentinel.numpy_norm")
matrix_norm = _prohibited_wrapper("sentinel.matrix_norm")
spectral_failed_closed = False
try:
    linalg_norm(Matrix(), 2)
except DirectOnlyComplianceError:
    spectral_failed_closed = True
nuclear_failed_closed = False
try:
    legacy_norm(Matrix(), "nuc")
except DirectOnlyComplianceError:
    nuclear_failed_closed = True
numpy_spectral_failed_closed = False
try:
    numpy_norm(Matrix(), 2)
except DirectOnlyComplianceError:
    numpy_spectral_failed_closed = True
matrix_norm_failed_closed = False
try:
    matrix_norm(Matrix())
except DirectOnlyComplianceError:
    matrix_norm_failed_closed = True
safe_result = linalg_norm(Matrix())
print(json.dumps({
    "spectral_failed_closed": spectral_failed_closed,
    "nuclear_failed_closed": nuclear_failed_closed,
    "numpy_spectral_failed_closed": numpy_spectral_failed_closed,
    "matrix_norm_failed_closed": matrix_norm_failed_closed,
    "safe_result": safe_result,
    "safe_sentinel_call_count": sum(1 for value in calls if value != "numpy_abs"),
    "numpy_abs_call_count": calls.count("numpy_abs"),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "spectral_failed_closed": True,
        "nuclear_failed_closed": True,
        "numpy_spectral_failed_closed": True,
        "matrix_norm_failed_closed": True,
        "safe_result": "safe",
        "safe_sentinel_call_count": 1,
        "numpy_abs_call_count": 0,
    }


def test_live_numpy_and_torch_aliases_are_separately_guarded():
    program = """
import json
import numpy
import torch
from scripts.odt_direct_only_compliance import (
    DirectOnlyComplianceError,
    _RUNTIME_WRAPPERS,
    direct_only_runtime_report,
    install_direct_only_runtime_guard,
)
install_direct_only_runtime_guard()
numpy_alias_names = (
    "svd", "pinv", "lstsq", "cholesky", "matrix_power", "matrix_rank",
    "cond", "inv", "solve", "tensorinv", "tensorsolve", "eig", "eigh",
    "eigvals", "eigvalsh", "norm",
)
separate_alias_identities = all(
    getattr(numpy.linalg.linalg, name)
    is _RUNTIME_WRAPPERS[f"numpy.linalg.linalg.{name}"]
    for name in numpy_alias_names
)
direct_numpy_evd_identities = all(
    getattr(numpy.linalg, name) is _RUNTIME_WRAPPERS[f"numpy.linalg.{name}"]
    for name in ("eig", "eigh", "eigvals", "eigvalsh")
)
torch_functional_identity = (
    torch.functional.norm is _RUNTIME_WRAPPERS["torch.functional.norm"]
)
blocked = 0
for function, args, kwargs in (
    (numpy.linalg.linalg.svd, (object(),), {}),
    (numpy.linalg.eigh, (object(),), {}),
    (numpy.linalg.linalg.norm, ([[0.0, 0.0], [0.0, 0.0]], 2), {}),
    (torch.functional.norm, (object(),), {"p": "nuc"}),
):
    try:
        function(*args, **kwargs)
    except DirectOnlyComplianceError:
        blocked += 1
report = direct_only_runtime_report()
print(json.dumps({
    "separate_alias_identities": separate_alias_identities,
    "direct_numpy_evd_identities": direct_numpy_evd_identities,
    "torch_functional_identity": torch_functional_identity,
    "blocked": blocked,
    "prohibited_attempt_count": report["prohibited_attempt_count"],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "separate_alias_identities": True,
        "direct_numpy_evd_identities": True,
        "torch_functional_identity": True,
        "blocked": 4,
        "prohibited_attempt_count": 4,
    }


def test_exact_runtime_call_sets_reject_missing_and_extra_without_kernels():
    program = f"""
import json
from scripts.odt_direct_only_compliance import (
    DIRECT_RQ_RUNTIME_CALL,
    DirectOnlyComplianceError,
    assert_direct_only_runtime_guard,
    install_direct_only_runtime_guard,
)
report = install_direct_only_runtime_guard()
missing_rejected = False
try:
    assert_direct_only_runtime_guard(
        report, exact_allowed_calls={{DIRECT_RQ_RUNTIME_CALL}}
    )
except DirectOnlyComplianceError:
    missing_rejected = True
extra = dict(report)
extra["allowed_calls"] = [DIRECT_RQ_RUNTIME_CALL]
extra["allowed_call_count"] = 1
extra_rejected = False
try:
    assert_direct_only_runtime_guard(extra, exact_allowed_calls=())
except DirectOnlyComplianceError:
    extra_rejected = True
print(json.dumps({{
    "missing_rejected": missing_rejected,
    "extra_rejected": extra_rejected,
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "missing_rejected": True,
        "extra_rejected": True,
    }


def test_tiny_full_runner_uses_only_allowed_qr_and_algorithm3_evd(tmp_path: Path):
    output = tmp_path / "tiny_full_direct_only.json"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--lane",
            "tiny",
            "--stage",
            "full",
            "--device",
            "cpu",
            "--block-size",
            "64",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    runtime = result["runtime_direct_only_guard_final"]
    assert result["all_gates_pass"]
    assert runtime["prohibited_attempt_count"] == 0
    assert runtime["allowed_call_count"] > 0
    assert any(call.endswith(":torch.linalg.qr") for call in runtime["allowed_calls"])
    assert any(call.endswith(":torch.linalg.eigh") for call in runtime["allowed_calls"])
