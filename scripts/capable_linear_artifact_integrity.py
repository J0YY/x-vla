"""Small file-identity checks shared by the capable-linear attestation lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from typing import Any
from collections.abc import Mapping
from pathlib import Path


def _scalar_value(value: Any, label: str) -> Any:
    if hasattr(value, "numel") and int(value.numel()) != 1:
        raise RuntimeError(f"{label} is not scalar")
    try:
        return value.item() if hasattr(value, "item") else value
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"{label} could not be read as a scalar") from error


def _finite_vector(value: Any, label: str) -> tuple[float, ...]:
    try:
        if hasattr(value, "detach"):
            raw = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (tuple, list)):
            raw = list(value)
        else:
            raw = list(value)
        result = tuple(float(item) for item in raw)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{label} is not a numeric vector") from error
    if not result or any(not math.isfinite(item) for item in result):
        raise RuntimeError(f"{label} is empty or nonfinite")
    return result


def require_rational_norm_buffer_inventory(
    state: Mapping[str, Any],
    *,
    expected_count: int,
    expected_inactive_sites: tuple[str, ...],
    expected_inactive_running_ms: Mapping[str, float],
    expected_pade_numerator: tuple[float, ...],
    expected_pade_denominator: tuple[float, ...],
) -> dict[str, Any]:
    """Require an exact paired scalar norm-buffer inventory and inactive mask."""

    if expected_count <= 0 or any(not isinstance(name, str) for name in state):
        raise RuntimeError("normalization checkpoint inventory is malformed")
    if tuple(sorted(set(expected_inactive_sites))) != expected_inactive_sites:
        raise RuntimeError("expected inactive normalization sites are not canonical")
    if set(expected_inactive_running_ms) != set(expected_inactive_sites):
        raise RuntimeError("inactive running-statistic authority differs")

    initialized_suffix = ".initialized"
    running_suffix = ".running_ms"
    numerator_suffix = ".pa"
    denominator_suffix = ".pb"
    initialized = {
        name[: -len(initialized_suffix)]: value
        for name, value in state.items()
        if name.endswith(initialized_suffix)
    }
    running = {
        name[: -len(running_suffix)]: value
        for name, value in state.items()
        if name.endswith(running_suffix)
    }
    numerators = {
        name[: -len(numerator_suffix)]: value
        for name, value in state.items()
        if name.endswith(numerator_suffix)
    }
    denominators = {
        name[: -len(denominator_suffix)]: value
        for name, value in state.items()
        if name.endswith(denominator_suffix)
    }
    if (
        len(initialized) != expected_count
        or len(running) != expected_count
        or len(numerators) != expected_count
        or len(denominators) != expected_count
        or set(initialized) != set(running)
        or set(initialized) != set(numerators)
        or set(initialized) != set(denominators)
    ):
        raise RuntimeError("normalization buffer count or pairing differs")

    flags: dict[str, bool] = {}
    values: dict[str, float] = {}
    for site in sorted(initialized):
        flag = _scalar_value(initialized[site], f"{site}{initialized_suffix}")
        if type(flag) is not bool:
            raise RuntimeError(f"{site}{initialized_suffix} is not boolean")
        flags[site] = flag
        raw = _scalar_value(running[site], f"{site}{running_suffix}")
        if isinstance(raw, bool):
            raise RuntimeError(f"{site}{running_suffix} is not numeric")
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"{site}{running_suffix} is not numeric") from error
        if not math.isfinite(numeric):
            raise RuntimeError(f"{site}{running_suffix} is nonfinite")
        values[site] = numeric
        if _finite_vector(numerators[site], f"{site}{numerator_suffix}") != tuple(
            expected_pade_numerator
        ):
            raise RuntimeError(f"{site}{numerator_suffix} differs")
        if _finite_vector(denominators[site], f"{site}{denominator_suffix}") != tuple(
            expected_pade_denominator
        ):
            raise RuntimeError(f"{site}{denominator_suffix} differs")

    inactive = tuple(site for site in sorted(flags) if not flags[site])
    if inactive != expected_inactive_sites:
        raise RuntimeError(
            f"normalization inactive-site mask differs: observed {inactive}"
        )
    for site, expected in expected_inactive_running_ms.items():
        if values[site] != float(expected):
            raise RuntimeError(f"{site} inactive running statistic differs")
    if any(values[site] <= 0.0 for site in values if site not in inactive):
        raise RuntimeError("active normalization running statistic is not positive")
    return {
        "site_count": expected_count,
        "initialized_true_count": expected_count - len(inactive),
        "initialized_false_sites": list(inactive),
        "all_running_ms_scalar_and_finite": True,
        "all_active_running_ms_positive": True,
        "all_pade_coefficients_match_frozen_defaults": True,
        "inactive_running_ms": {site: values[site] for site in inactive},
    }


def _read_physical_bytes(path: Path, label: str) -> bytes:
    """Read one unchanged physical inode without following a symbolic link."""

    try:
        before_path = os.lstat(path)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing or nonphysical") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"{label} is missing or nonphysical")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} could not be opened as a physical file") from error
    try:
        before_fd = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read()
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} changed during authenticated read") from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before_fd, field) != getattr(after_fd, field) for field in identity_fields)
        or before_path.st_dev != before_fd.st_dev
        or before_path.st_ino != before_fd.st_ino
        or after_path.st_dev != after_fd.st_dev
        or after_path.st_ino != after_fd.st_ino
        or not stat.S_ISREG(after_path.st_mode)
        or len(encoded) != after_fd.st_size
    ):
        raise RuntimeError(f"{label} changed during authenticated read")
    return encoded


def assert_physical_hashes_unchanged(
    paths: Mapping[str, Path], expected: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Hash physical files and optionally require an exact prior snapshot."""

    if not paths or any(not isinstance(label, str) or not label for label in paths):
        raise ValueError("artifact identity inventory is empty or malformed")
    observed: dict[str, str] = {}
    for label, path in paths.items():
        encoded = _read_physical_bytes(path, f"attestation input {label}")
        observed[label] = hashlib.sha256(encoded).hexdigest()
    if expected is not None and observed != dict(expected):
        raise RuntimeError("attestation inputs changed across publication")
    return observed


def read_authenticated_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Read, hash, and decode one stable physical inode from the same bytes."""

    encoded = _read_physical_bytes(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> Any:
        raise RuntimeError(f"{label} contains a nonfinite JSON constant {token}")

    value = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value, hashlib.sha256(encoded).hexdigest()
