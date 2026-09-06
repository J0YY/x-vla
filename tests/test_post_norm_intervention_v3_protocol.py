"""Static protocol and schema tests for Stage 7 v3."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import athena.post_norm_intervention_v3_publication as publication
import athena.build_post_norm_intervention_v3_panel as panel_builder
import athena.post_norm_intervention_v3_common as common

from athena.post_norm_intervention_v3_common import (
    ALPHA_GRID,
    CACHE,
    CHECKPOINTS,
    CERTIFICATION_GATES,
    DATASET_TO_OFFICIAL_TASK,
    DATA_MANIFEST_KEYS,
    DEPLOYED_CONDITIONS,
    GLOBAL_CLAIM_BOUNDARY,
    GLOBAL_ODT_REQUIRED_KEYS,
    HAAR_CONDITIONS,
    HORIZON,
    INFEASIBLE_IDENTITY_KEYS,
    PROTOCOL,
    SAMPLE_NAMESPACE,
    SCHEMA_EXCLUSION,
    SCHEMA_CERTIFICATION_DATA,
    SCHEMA_GLOBAL_ODT,
    SCHEMA_PREPARATION_ANCESTRY,
    SOFTWARE_RUNTIME,
    STAGE5_RESULT_PATH,
    STAGE5_RESULT_SHA256,
    TERMINAL_CLAIM_BOUNDARY,
    canonical_sha256,
    file_sha256,
    claim_certification_start,
    namespaced_seed,
    physical_transaction_directory,
    physical_transaction_root,
    recover_json_alias_for_existing_final,
    validate_certification_start,
    validate_claim_track,
    validate_global_odt_certificate,
    validate_infeasibility_receipt,
    validate_infeasibility_transaction,
    validate_panel_development_membership,
    validate_predecessor_exclusion_manifest,
    validate_preparation_ancestry,
    validate_split_tensor_data,
    write_quantitative_infeasibility,
)
from athena.summarize_post_norm_intervention_v3 import (
    action_profile_from_dict,
    exact_results_inventory,
    validate_interrupted_failure_partial,
    validate_exact_phase_state,
    validate_certification_scientific_payload,
    publish_or_reproduce_summary,
)
from athena.post_norm_intervention_v3_publication import (
    PublicationArtifact,
    begin_composite_publication,
    incomplete_publication_phases,
    publish_artifact_group,
    recover_committed_transaction_aliases,
    transaction_root_inventory,
    validate_publication_commit,
)
from xvla.train.post_norm_intervention_v3 import action_profiles_pass, compare_action_profiles


ROOT = Path(__file__).resolve().parents[1]


def _accepted_stage5_records() -> dict[str, list[dict]]:
    def records(split: str) -> list[dict]:
        split_offset = 0 if split == "discovery" else 100_000
        return [
            {
                "dataset_task": task,
                "official_task": DATASET_TO_OFFICIAL_TASK[task],
                "episode": split_offset + task * 1000 + index,
                "frame": 0,
                "horizon": HORIZON,
                "row_sha256": hashlib.sha256(
                    f"row|{split}|{task}|{index}".encode()
                ).hexdigest(),
                "window_sha256": hashlib.sha256(
                    f"window|{split}|{task}|{index}".encode()
                ).hexdigest(),
            }
            for task in range(10) for index in range(64)
        ]
    return {split: records(split) for split in ("discovery", "evaluation")}


def _valid_exclusion() -> dict:
    accepted_stage5_records = _accepted_stage5_records()
    accepted_records = [
        record
        for split in ("discovery", "evaluation")
        for record in accepted_stage5_records[split]
    ]
    panel_sha256 = canonical_sha256(accepted_stage5_records)
    exclusion = {
        "schema": SCHEMA_EXCLUSION,
        "identity_sources": {STAGE5_RESULT_PATH: STAGE5_RESULT_SHA256},
        "accepted_stage5_panel_sha256": panel_sha256,
        "accepted_stage5_records": accepted_stage5_records,
        "excluded_window_sha256": sorted(
            record["window_sha256"] for record in accepted_records
        ),
        "excluded_frame_footprint_sha256": sorted(
            hashlib.sha256(
                f"{record['dataset_task']}|{record['episode']}|{record['frame'] + offset}".encode()
            ).hexdigest()
            for record in accepted_records for offset in range(HORIZON)
        ),
        "excluded_episode_sha256": sorted(
            hashlib.sha256(
                f"{record['dataset_task']}|{record['episode']}".encode()
            ).hexdigest()
            for record in accepted_records
        ),
        "contains_activations": False,
        "contains_actions": False,
        "contains_outcomes": False,
        "contains_certification_metrics": False,
        "source_hashes": {"source.py": "a" * 64},
        "runtime": SOFTWARE_RUNTIME,
        "content_sha256": None,
    }
    exclusion["content_sha256"] = canonical_sha256({
        key: value for key, value in exclusion.items() if key != "content_sha256"
    })
    return exclusion


def test_exclusion_rejects_noncanonical_stage5_identity_source(monkeypatch) -> None:
    exclusion = _valid_exclusion()
    monkeypatch.setattr(common, "STAGE5_PANEL_SHA256", exclusion["accepted_stage5_panel_sha256"])
    validate_predecessor_exclusion_manifest(exclusion)
    tampered = copy.deepcopy(exclusion)
    tampered["identity_sources"] = {"another/stage5.json": STAGE5_RESULT_SHA256}
    tampered["content_sha256"] = canonical_sha256({
        key: value for key, value in tampered.items() if key != "content_sha256"
    })
    with pytest.raises(RuntimeError, match="accepted Stage 5"):
        validate_predecessor_exclusion_manifest(tampered)


def test_exclusion_requires_exact_frozen_discovery_and_evaluation_union(monkeypatch) -> None:
    exclusion = _valid_exclusion()
    frozen_panel_sha = exclusion["accepted_stage5_panel_sha256"]
    monkeypatch.setattr(common, "STAGE5_PANEL_SHA256", frozen_panel_sha)
    validate_predecessor_exclusion_manifest(exclusion)
    tampered = copy.deepcopy(exclusion)
    omitted = tampered["accepted_stage5_records"]["evaluation"].pop()
    tampered["excluded_window_sha256"].remove(omitted["window_sha256"])
    tampered["accepted_stage5_panel_sha256"] = canonical_sha256(
        tampered["accepted_stage5_records"]
    )
    tampered["content_sha256"] = canonical_sha256({
        key: value for key, value in tampered.items() if key != "content_sha256"
    })
    with pytest.raises(RuntimeError, match="accepted Stage 5 panel identity"):
        validate_predecessor_exclusion_manifest(tampered)


def test_discovery_materialization_rejects_cache_identity_drift(monkeypatch) -> None:
    exclusion = _valid_exclusion()
    monkeypatch.setattr(common, "STAGE5_PANEL_SHA256", exclusion["accepted_stage5_panel_sha256"])
    windows = [
        {
            "task": record["dataset_task"],
            "episode": record["episode"],
            "frame": record["frame"],
            "row_sha256": record["row_sha256"],
            "window_sha256": record["window_sha256"],
        }
        for record in exclusion["accepted_stage5_records"]["discovery"]
    ]
    windows[0] = {**windows[0], "row_sha256": "f" * 64}
    monkeypatch.setattr(panel_builder, "file_sha256", lambda path: CACHE["sha256"])
    monkeypatch.setattr(
        panel_builder,
        "load_validated_windows",
        lambda path: ([], {"legacy_exclusive": windows}, {}),
    )
    with pytest.raises(RuntimeError, match="differs from frozen cache"):
        panel_builder.materialize_split_payloads(Path("cache.pkl"), {}, exclusion)


def test_discovery_materialization_preserves_frozen_order_and_downstream_binding(
    monkeypatch,
) -> None:
    exclusion = _valid_exclusion()
    monkeypatch.setattr(common, "STAGE5_PANEL_SHA256", exclusion["accepted_stage5_panel_sha256"])

    def window(record: dict) -> dict:
        return {
            "task": record["dataset_task"],
            "episode": record["episode"],
            "frame": record["frame"],
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
            "actions": np.zeros((HORIZON, 7), dtype=np.float32),
            "row_sha256": record["row_sha256"],
            "window_sha256": record["window_sha256"],
        }

    accepted = exclusion["accepted_stage5_records"]["discovery"]
    cache_windows = [window(record) for record in reversed(accepted)]
    panel_records = []
    for index, partition in enumerate(
        ["development_0", "development_1", "development_2", "development_3", "sealed_certification"]
    ):
        digest = hashlib.sha256(f"fresh|{partition}".encode()).hexdigest()
        cache_windows.append({
            "task": index,
            "episode": 900_000 + index,
            "frame": 0,
            "image": np.zeros((64, 64, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
            "actions": np.zeros((HORIZON, 7), dtype=np.float32),
            "row_sha256": hashlib.sha256(f"fresh-row|{partition}".encode()).hexdigest(),
            "window_sha256": digest,
        })
        panel_records.append({
            "task": index,
            "window_sha256": digest,
            "partition": partition,
        })
    stats = {
        "action_mean": np.zeros(7, dtype=np.float32),
        "action_std": np.ones(7, dtype=np.float32),
        "state_mean": np.zeros(8, dtype=np.float32),
        "state_std": np.ones(8, dtype=np.float32),
    }
    monkeypatch.setattr(panel_builder, "file_sha256", lambda path: CACHE["sha256"])
    monkeypatch.setattr(
        panel_builder,
        "load_validated_windows",
        lambda path: ([], {"legacy_exclusive": cache_windows}, {"legacy_exclusive": stats}),
    )
    panel = {"records": panel_records, "content_sha256": "d" * 64}
    discovery, _, _ = panel_builder.materialize_split_payloads(
        Path("cache.pkl"), panel, exclusion
    )
    expected_windows = [record["window_sha256"] for record in accepted]
    assert discovery["data"]["window_sha256"] == expected_windows
    validate_split_tensor_data(
        discovery,
        schema=common.SCHEMA_DISCOVERY_DATA,
        expected_panel_content_sha256=discovery["panel_content_sha256"],
        expected_records_by_partition={"discovery": accepted},
    )
    tampered = copy.deepcopy(discovery)
    tampered["data"]["window_sha256"][0:2] = reversed(
        tampered["data"]["window_sha256"][0:2]
    )
    with pytest.raises(RuntimeError, match="panel membership"):
        validate_split_tensor_data(
            tampered,
            schema=common.SCHEMA_DISCOVERY_DATA,
            expected_panel_content_sha256=discovery["panel_content_sha256"],
            expected_records_by_partition={"discovery": accepted},
        )


def test_discovery_materialization_does_not_call_mutable_stage5_chooser() -> None:
    source = (ROOT / "athena/build_post_norm_intervention_v3_panel.py").read_text()
    assert "choose_panel" not in source


def _phase_identities(phase: str) -> dict:
    identities = {key: None for key in INFEASIBLE_IDENTITY_KEYS}
    identities["source_hashes"] = {
        "xvla/train/post_norm_intervention_v3.py": "a" * 64,
    }
    present = {
        "prepare_development": {
            "checkpoint_sha256", "cache_sha256", "model_state_sha256", "panel_sha256",
            "exclusion_manifest_sha256", "discovery_data_sha256",
            "development_data_sha256", "data_manifest_sha256",
            "preparation_ancestry_sha256",
            "discovery_queries_sha256", "discovery_means_sha256",
        },
        "certify_once": INFEASIBLE_IDENTITY_KEYS - {"source_hashes"},
        "global_odt_prerequisite": {
            "checkpoint_sha256", "cache_sha256", "exclusion_manifest_sha256",
        },
        "panel_manifest": {"cache_sha256", "exclusion_manifest_sha256"},
    }[phase]
    identities.update({key: "1" * 64 for key in present})
    return identities


def test_split_certification_tensor_is_float_typed_and_exactly_panel_bound() -> None:
    records = [
        {"task": task, "window_sha256": f"{task:064x}"}
        for task in range(10)
    ]
    payload = {
        "schema": SCHEMA_CERTIFICATION_DATA,
        "cache_sha256": CACHE["sha256"],
        "panel_content_sha256": "b" * 64,
        "action_mean": torch.zeros(7, dtype=torch.float64),
        "action_std": torch.ones(7, dtype=torch.float64),
        "data": {
            "images": torch.zeros(10, 3, 64, 64, dtype=torch.float32),
            "tasks": torch.arange(10, dtype=torch.int64),
            "states": torch.zeros(10, 8, dtype=torch.float32),
            "embodiments": torch.zeros(10, dtype=torch.int64),
            "window_sha256": [record["window_sha256"] for record in records],
        },
    }
    validate_split_tensor_data(
        payload, schema=SCHEMA_CERTIFICATION_DATA,
        expected_panel_content_sha256="b" * 64,
        expected_records_by_partition={"sealed_certification": records},
    )
    tampered = copy.deepcopy(payload)
    tampered["data"]["tasks"][0] = 9
    with pytest.raises(RuntimeError, match="panel membership"):
        validate_split_tensor_data(
            tampered, schema=SCHEMA_CERTIFICATION_DATA,
            expected_panel_content_sha256="b" * 64,
            expected_records_by_partition={"sealed_certification": records},
        )


def test_panel_development_membership_rejects_cross_panel_substitution() -> None:
    records = []
    for fold in range(4):
        for task in range(10):
            records.append({
                "task": task, "episode_id": f"{fold}-{task}",
                "sample_id": f"{fold * 10 + task:064x}",
                "window_sha256": f"{fold * 10 + task:064x}",
                "frame_footprints": [f"{1000 + fold * 80 + task * 8 + h:064x}" for h in range(HORIZON)],
                "partition": f"development_{fold}",
            })
    panel = {"records": records}
    development_records = {
        f"development_{fold}": [
            record for record in records if record["partition"] == f"development_{fold}"
        ]
        for fold in range(4)
    }
    manifest = {
        "development_records": development_records,
        "development_membership_sha256": canonical_sha256(development_records),
    }
    validate_panel_development_membership(panel, manifest)
    substituted = copy.deepcopy(manifest)
    substituted["development_records"]["development_0"][0]["episode_id"] = "other-panel"
    substituted["development_membership_sha256"] = canonical_sha256(substituted["development_records"])
    with pytest.raises(RuntimeError, match="panel development partition"):
        validate_panel_development_membership(panel, substituted)


def global_certificate(checkpoint_sha256: str = "c" * 64) -> dict:
    value = {key: True for key in GLOBAL_ODT_REQUIRED_KEYS}
    value.update({
        "schema": SCHEMA_GLOBAL_ODT,
        "checkpoint_sha256": checkpoint_sha256,
        "source_hashes": {"xvla/full_policy_compiler.py": "a" * 64},
        "compiler_topology_sha256": "b" * 64,
        "formal_finite_precision_certificate": False,
        "numerical_scope": "a_posteriori_float64_davis_kahan_with_heuristic_roundoff_allowance",
        "roundoff_allowance": {
            "formula": "multiplier*epsilon*dimension*scale",
            "multiplier": 10_000,
            "epsilon": 2.220446049250313e-16,
            "dimension": 128,
            "scale": 1.0,
        },
        "content_sha256": None,
    })
    value["content_sha256"] = canonical_sha256({key: value[key] for key in value if key != "content_sha256"})
    return value


def test_global_track_fails_closed_without_full_policy_certificate() -> None:
    assert validate_claim_track(
        "terminal_balanced", checkpoint_sha256="c" * 64
    ) == TERMINAL_CLAIM_BOUNDARY
    with pytest.raises(RuntimeError, match="global_odt_prerequisite_unavailable"):
        validate_claim_track("global_odt", checkpoint_sha256="c" * 64)
    with pytest.raises(RuntimeError, match="global_odt_prerequisite_unavailable"):
        validate_claim_track(
            "global_odt", checkpoint_sha256="c" * 64,
            global_odt_certificate=global_certificate(),
        )


def test_global_certificate_rejects_tiny_bridge_missing_coverage_and_formal_precision_claim() -> None:
    certificate = global_certificate()
    validate_global_odt_certificate(certificate, expected_checkpoint_sha256="c" * 64)
    missing = copy.deepcopy(certificate)
    missing["full_deployed_precommit_map_covered"] = False
    missing["content_sha256"] = canonical_sha256({key: missing[key] for key in missing if key != "content_sha256"})
    with pytest.raises(RuntimeError, match="prerequisite gates failed"):
        validate_global_odt_certificate(missing, expected_checkpoint_sha256="c" * 64)
    overstated = copy.deepcopy(certificate)
    overstated["formal_finite_precision_certificate"] = True
    overstated["content_sha256"] = canonical_sha256({key: overstated[key] for key in overstated if key != "content_sha256"})
    with pytest.raises(RuntimeError, match="unsupported formal"):
        validate_global_odt_certificate(overstated, expected_checkpoint_sha256="c" * 64)


def test_quantitative_failure_receipt_is_exact_hashed_npz_bound_and_terminal(tmp_path: Path) -> None:
    gate = {
        "name": "aggregate_relative", "actual": 0.11, "reference": 0.0,
        "threshold": 0.10, "signed_margin": 0.10 - 0.11, "passed": False,
        "location": "sealed_certification/task=2",
    }
    identities = _phase_identities("certify_once")
    eligibility = np.zeros(91, dtype=np.bool_)
    eligibility[:5] = True
    numeric_arrays = {
        "sampling_eligibility": eligibility,
        "design_counts": np.asarray([0, 0, 160, 160, 160, 160, 160, 160], dtype=np.int64),
        "design_quintets": np.asarray([0, 0, 32, 32, 32, 32, 32, 32], dtype=np.int64),
    }
    design_counts = {alpha: int(value) for alpha, value in zip(ALPHA_GRID, numeric_arrays["design_counts"])}
    design_quintets = {
        alpha: int(value) for alpha, value in zip(ALPHA_GRID, numeric_arrays["design_quintets"])
    }
    proposal_seeds = {
        f"matched_haar_d{i}": tuple(
            namespaced_seed(SAMPLE_NAMESPACE, 2, i, horizon) for horizon in range(HORIZON)
        )
        for i in range(5)
    }
    receipt = write_quantitative_infeasibility(
        tmp_path / "failure.json", tmp_path / "failure.npz",
        failure_class="certification_infeasible", failure_phase="certify_once",
        reason="frozen quintet failed", seed=2, certification_attempt=1,
        identities=identities, chosen_beta=0.5,
        design_counts_by_alpha=design_counts,
        design_quintets_by_alpha=design_quintets,
        proposals_examined=91,
        selected_proposal_indices={f"matched_haar_d{i}": i for i in range(5)},
        selected_proposal_seeds=proposal_seeds,
        selected_operator_sha256={f"matched_haar_d{i}": "2" * 64 for i in range(5)},
        gate_results=[gate], numeric_arrays=numeric_arrays,
        claim_boundary=TERMINAL_CLAIM_BOUNDARY,
    )
    validate_infeasibility_receipt(receipt, artifact_root=tmp_path)
    assert not ((tmp_path / "failure.json").stat().st_mode & 0o222)
    assert not ((tmp_path / "failure.npz").stat().st_mode & 0o222)
    assert not ((tmp_path / "failure.commit.json").stat().st_mode & 0o222)
    assert not (tmp_path / "failure.partial.json").exists()
    assert (tmp_path / "failure.commit.json").is_file()
    assert validate_infeasibility_transaction(tmp_path / "failure.json") == receipt
    repeated = write_quantitative_infeasibility(
        tmp_path / "failure.json", tmp_path / "failure.npz",
        failure_class="certification_infeasible", failure_phase="certify_once",
        reason="frozen quintet failed", seed=2, certification_attempt=1,
        identities=identities, chosen_beta=0.5,
        design_counts_by_alpha=design_counts,
        design_quintets_by_alpha=design_quintets,
        proposals_examined=91,
        selected_proposal_indices={f"matched_haar_d{i}": i for i in range(5)},
        selected_proposal_seeds=proposal_seeds,
        selected_operator_sha256={f"matched_haar_d{i}": "2" * 64 for i in range(5)},
        gate_results=[gate], numeric_arrays=numeric_arrays,
        claim_boundary=TERMINAL_CLAIM_BOUNDARY,
    )
    assert repeated == receipt
    resumed = copy.deepcopy(receipt)
    resumed["resumed_sampling"] = True
    resumed["content_sha256"] = canonical_sha256({key: resumed[key] for key in resumed if key != "content_sha256"})
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_infeasibility_receipt(resumed, artifact_root=tmp_path)
    sparse = copy.deepcopy(receipt)
    del sparse["maximum_offender"]
    with pytest.raises(RuntimeError, match="field inventory"):
        validate_infeasibility_receipt(sparse, artifact_root=tmp_path)
    bad_margin = copy.deepcopy(receipt)
    bad_margin["gate_results"][0]["signed_margin"] = -0.02
    with pytest.raises(RuntimeError, match="threshold minus actual"):
        validate_infeasibility_receipt(bad_margin, artifact_root=tmp_path)
    wrong_phase_identities = copy.deepcopy(receipt)
    wrong_phase_identities["identities"]["certification_data_sha256"] = None
    wrong_phase_identities["content_sha256"] = canonical_sha256({
        key: wrong_phase_identities[key]
        for key in wrong_phase_identities if key != "content_sha256"
    })
    with pytest.raises(RuntimeError, match="exact phase"):
        validate_infeasibility_receipt(wrong_phase_identities, artifact_root=tmp_path)
    with np.load(tmp_path / "failure.npz", allow_pickle=False) as stored:
        rewritten = {name: stored[name].copy() for name in stored.files}
    rewritten["undeclared_numeric_stream"] = np.asarray([1], dtype=np.int64)
    (tmp_path / "failure.npz").chmod(0o600)
    np.savez_compressed(tmp_path / "failure.npz", **rewritten)
    extra_array = copy.deepcopy(receipt)
    extra_array["numeric_artifact"]["sha256"] = file_sha256(tmp_path / "failure.npz")
    extra_array["numeric_artifact"]["layout"] = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in rewritten.items()
    }
    extra_array["content_sha256"] = canonical_sha256({
        key: extra_array[key] for key in extra_array if key != "content_sha256"
    })
    with pytest.raises(RuntimeError, match="numeric stream inventory differs"):
        validate_infeasibility_receipt(extra_array, artifact_root=tmp_path)
    (tmp_path / "failure.npz").chmod(0o600)
    with (tmp_path / "failure.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="bytes differ"):
        validate_infeasibility_receipt(receipt, artifact_root=tmp_path)


def test_quantitative_failure_receipt_rejects_string_numeric_payload(tmp_path: Path) -> None:
    identities = _phase_identities("global_odt_prerequisite")
    with pytest.raises(RuntimeError, match="non-numeric"):
        write_quantitative_infeasibility(
            tmp_path / "bad.json", tmp_path / "bad.npz",
            failure_class="global_odt_prerequisite_unavailable",
            failure_phase="global_odt_prerequisite", reason="unavailable",
            seed=0, certification_attempt=0, identities=identities,
            chosen_beta=None, design_counts_by_alpha={}, design_quintets_by_alpha={},
            proposals_examined=0, selected_proposal_indices={}, selected_proposal_seeds={},
            selected_operator_sha256={}, gate_results=[],
            numeric_arrays={"bad_strings": np.asarray(["not numeric"])},
            claim_boundary=GLOBAL_CLAIM_BOUNDARY,
        )


def test_orphan_failure_npz_is_bound_to_exact_request_before_recovery(tmp_path: Path) -> None:
    orphan_npz = tmp_path / "orphan.npz"
    np.savez_compressed(orphan_npz, sampling_eligibility=np.asarray([True], dtype=np.bool_))
    identities = _phase_identities("prepare_development")
    shortfall_gate = {
        "name": "ordered_greedy_sample_quintet_shortfall", "actual": 5.0,
        "reference": 0.0, "threshold": 0.0, "signed_margin": -5.0,
        "passed": False, "location": "development_sample_stream",
    }
    with pytest.raises(RuntimeError, match="numeric request differs"):
        write_quantitative_infeasibility(
            tmp_path / "orphan.json", orphan_npz,
            failure_class="ordered_greedy_sample_shortfall", failure_phase="prepare_development",
            reason="orphan transaction", seed=0, certification_attempt=0,
            identities=identities, chosen_beta=1.0,
            design_counts_by_alpha={1.0: 160}, design_quintets_by_alpha={1.0: 32}, proposals_examined=0,
            selected_proposal_indices={}, selected_proposal_seeds={},
            selected_operator_sha256={}, gate_results=[shortfall_gate],
            numeric_arrays={
                "sampling_eligibility": np.asarray([], dtype=np.bool_),
                "design_counts": np.asarray([160, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                "design_quintets": np.asarray([32, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
            },
            claim_boundary=TERMINAL_CLAIM_BOUNDARY,
        )
    partial = tmp_path / "orphan.partial.json"
    assert partial.is_file()
    assert "infeasibility-partial-v3" in partial.read_text()
    assert "request_sha256" in json.loads(partial.read_text())


def test_partial_failure_recovery_is_phase_specific_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "recover.json"
    npz_path = tmp_path / "recover.npz"
    partial_path = tmp_path / "recover.partial.json"
    identities = _phase_identities("prepare_development")
    shortfall_gate = {
        "name": "ordered_greedy_sample_quintet_shortfall", "actual": 5.0,
        "reference": 0.0, "threshold": 0.0, "signed_margin": -5.0,
        "passed": False, "location": "development_sample_stream",
    }
    kwargs = dict(
        failure_class="ordered_greedy_sample_shortfall",
        failure_phase="prepare_development", reason="recover exact request", seed=0,
        certification_attempt=0, identities=identities, chosen_beta=1.0,
        design_counts_by_alpha={1.0: 160}, design_quintets_by_alpha={1.0: 32}, proposals_examined=0,
        selected_proposal_indices={}, selected_proposal_seeds={},
        selected_operator_sha256={}, gate_results=[shortfall_gate],
        numeric_arrays={
            "sampling_eligibility": np.asarray([], dtype=np.bool_),
            "design_counts": np.asarray([160, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
            "design_quintets": np.asarray([32, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        },
        claim_boundary=TERMINAL_CLAIM_BOUNDARY,
    )
    real_save = np.savez_compressed
    monkeypatch.setattr(np, "savez_compressed", lambda *args, **values: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError, match="kill"):
        write_quantitative_infeasibility(json_path, npz_path, **kwargs)
    assert partial_path.is_file() and not npz_path.exists()
    monkeypatch.setattr(np, "savez_compressed", real_save)
    changed = dict(kwargs)
    changed["reason"] = "different request"
    with pytest.raises(RuntimeError, match="another exact request"):
        write_quantitative_infeasibility(
            json_path, npz_path, **changed,
        )
    receipt = write_quantitative_infeasibility(json_path, npz_path, **kwargs)
    assert receipt["failure_phase"] == "prepare_development"
    assert not partial_path.exists()
    assert validate_infeasibility_transaction(json_path) == receipt


def test_protocol_physically_separates_proposal_and_certification_and_preserves_predecessors() -> None:
    tensor_source = (ROOT / "xvla/train/post_norm_intervention_v3.py").read_text()
    protocol = (ROOT / "athena/POST_NORM_INTERVENTION_V3_PROTOCOL.md").read_text()
    assert "certification_queries" not in tensor_source.split("def freeze_development_proposals", 1)[1].split("def certify_frozen_controls_once", 1)[0]
    assert "v2 proposal bundle" in (ROOT / "athena/post_norm_intervention_v3_common.py").read_text()
    assert "tests -> predecessor_exclusion -> panel_freeze" in protocol
    assert "fixed `1e-8` resolved-projector failure remains failed" in protocol
    assert "not a formal finite-precision certificate" in protocol
    assert PROTOCOL["certification_attempts"] == 1
    assert PROTOCOL["resumed_sampling_after_certification"] is False
    assert PROTOCOL["global_odt_track_enabled"] is False
    assert PROTOCOL["scope"] == "pre_rollout_remove_only_feasibility_and_one_shot_certification"
    assert "def lock_action_semantic_label" not in tensor_source
    assert "def match_fixed_comparator_on_development" not in tensor_source
    assert not {
        "panel_path", "panel_file_sha256", "panel_content_sha256",
        "certification_data_path", "certification_data_sha256",
        "sealed_data_manifest_path", "sealed_data_manifest_sha256",
    } & set(DATA_MANIFEST_KEYS)


def test_cohort_claim_precedes_every_certification_sealed_open() -> None:
    source = (ROOT / "athena/certify_post_norm_intervention_v3.py").read_text()
    claimant = (ROOT / "athena/claim_post_norm_intervention_v3_cohort.py").read_text()
    prepare_source = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    freeze_source = (ROOT / "athena/freeze_post_norm_intervention_v3.py").read_text()
    seal_source = (ROOT / "athena/seal_post_norm_intervention_v3.py").read_text()
    claim = claimant.index("token = claim_certification_attempt(")
    validation = source.index("validate_certification_attempt_token(")
    proposal_validation = source.index("validate_proposal_freeze(proposal_freeze")
    payload_open = source.index("torch.load(args.payload")
    panel_open = source.index("panel = json.loads(args.panel.read_text())")
    sealed_data_open = source.index("torch.load(args.certification_data")
    execution_start = source.index("claim_certification_start(")
    sealed_payload_hash = source.index("file_sha256(args.payload)")
    assert validation < payload_open
    assert validation < panel_open
    assert validation < sealed_data_open
    assert validation < execution_start < sealed_payload_hash < sealed_data_open
    assert "#SBATCH --no-requeue" in (
        ROOT / "athena/slurm_certify_post_norm_intervention_v3.sbatch"
    ).read_text()
    assert proposal_validation < validation
    assert "torch.load" not in claimant
    assert "file_sha256(args.certification_data" not in claimant
    assert 'parser.add_argument("--manifests", type=Path, nargs=3' in claimant
    assert '"seeds": sorted(CHECKPOINTS)' in (
        ROOT / "athena/post_norm_intervention_v3_common.py"
    ).read_text()
    assert "args.proposal_freeze: proposal_freeze_sha" in source
    assert "load_validated_windows" not in source
    assert "load_validated_windows" not in prepare_source
    assert "args.certification_data" not in prepare_source
    assert 'add_argument("--certification-data"' not in prepare_source
    assert 'add_argument("--cache"' not in prepare_source
    assert 'add_argument("--sealed-manifest"' not in prepare_source
    assert 'add_argument("--panel"' not in prepare_source
    assert "torch.load(args.certification_data" not in freeze_source
    assert 'add_argument("--certification-data"' not in freeze_source
    assert 'add_argument("--cache"' not in freeze_source
    assert 'add_argument("--sealed-manifest"' not in freeze_source
    assert 'add_argument("--panel"' not in freeze_source
    payload_block = seal_source[
        seal_source.index('payload = {'):seal_source.index('payload_file_sha = serialized_sha256')
    ]
    for forbidden in {
        "proposal_seeds", "proposal_indices", "design_counts_by_alpha",
        "design_quintets_by_alpha", "sampling_eligibility",
    }:
        assert forbidden not in payload_block
    assert 'add_argument("--attempt-token"' in source
    assert 'add_argument("--result"' not in source
    assert 'add_argument("--batch-size"' not in source
    assert source.index("torch.use_deterministic_algorithms(True)") < validation
    assert source.index("attempt_path = args.attempt_token") < validation
    canonical = source.index("canonical_paths = {")
    rejection = source.index('raise RuntimeError("v3 certification input path is not canonical")')
    first_read = source.index("manifest_bytes = args.manifest.read_bytes()")
    assert canonical < rejection < first_read
    assert source.index('require_read_only_regular(args.manifest') < first_read
    assert source.index('require_read_only_regular(args.proposal_freeze') < validation
    assert source.index('require_read_only_regular(args.final_freeze') < validation
    for field in (
        "manifest", "checkpoint", "cache", "panel", "exclusion", "payload",
        "discovery_data", "certification_data", "development_manifest",
        "sealed_manifest", "final_freeze", "proposal_freeze",
    ):
        assert f'"{field}"' in source[canonical:rejection]


def test_seal_broker_hashes_live_canonical_sealed_tensor_without_deserializing_it() -> None:
    source = (ROOT / "athena/seal_post_norm_intervention_v3.py").read_text()
    canonical = source.index("canonical_certification_data =")
    readonly = source.index('require_read_only_regular(path, label="sealing input")')
    hash_capture = source.index("hashes = {path: file_sha256(path) for path in inputs}")
    binding = source.index('sealed["certification_data_sha256"] != hashes[args.certification_data]')
    final_write = source.index("publish_artifact_group(")
    assert canonical < readonly < hash_capture < binding < final_write
    assert "torch.load(args.certification_data" not in source
    assert '--certification-data results/post_norm_intervention_v3_sealed_certification_data.pt' in (
        ROOT / "athena/slurm_seal_post_norm_intervention_v3.sbatch"
    ).read_text()


def test_summary_inventory_never_hashes_unclaimed_sealed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = tmp_path / "public.json"
    shared = tmp_path / "post_norm_intervention_v3_sealed_certification_data.pt"
    payload0 = tmp_path / "post_norm_intervention_v3_certification_input_s0.pt"
    panel = tmp_path / "post_norm_intervention_v3_panel.json"
    sealed_manifest = tmp_path / "post_norm_intervention_v3_sealed_data_manifest.json"
    for path in (public, panel, sealed_manifest, shared, payload0):
        path.write_bytes(path.name.encode())
        path.chmod(0o400 if path.suffix == ".pt" else 0o444)
    paths = {path.name: path for path in (public, panel, sealed_manifest, shared, payload0)}
    opened: list[str] = []

    def guarded(path: Path) -> str:
        opened.append(path.name)
        if path in (panel, sealed_manifest, shared, payload0):
            raise AssertionError("sealed byte opened before attempt")
        return "a" * 64

    monkeypatch.setattr("athena.summarize_post_norm_intervention_v3.file_sha256", guarded)
    inventory = exact_results_inventory(paths, claimed_seeds=set())
    assert inventory[shared.name] == {
        "mode": "0o400", "sha256": "sealed_bytes_unopened_before_attempt",
    }
    assert inventory[payload0.name] == {
        "mode": "0o400", "sha256": "sealed_bytes_unopened_before_attempt",
    }
    assert inventory[panel.name]["sha256"] == "sealed_bytes_unopened_before_attempt"
    assert inventory[sealed_manifest.name]["sha256"] == "sealed_bytes_unopened_before_attempt"
    assert opened == [public.name]


@pytest.mark.parametrize("partial", [{0}, {0, 1}])
def test_summary_inventory_never_opens_shared_bytes_for_partial_cohort_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial: set[int],
) -> None:
    shared = tmp_path / "post_norm_intervention_v3_sealed_certification_data.pt"
    shared.write_bytes(b"sealed")
    shared.chmod(0o400)

    def forbidden(path: Path) -> str:
        raise AssertionError(f"opened shared bytes under partial cohort: {path}")

    monkeypatch.setattr("athena.summarize_post_norm_intervention_v3.file_sha256", forbidden)
    inventory = exact_results_inventory({shared.name: shared}, claimed_seeds=partial)
    assert inventory[shared.name]["sha256"] == "sealed_bytes_unopened_before_attempt"


def test_publication_recovers_after_partial_final_link_and_rejects_changed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    artifacts = (
        PublicationArtifact(
            tmp_path / "results/post_norm_intervention_v3_tx_a.json", "json", {"a": 1},
        ),
        PublicationArtifact(
            tmp_path / "results/post_norm_intervention_v3_tx_b.pt", "torch", {"b": torch.arange(3)},
        ),
    )
    original = publication._link_final
    calls = 0

    def fail_second(staged: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication interruption")
        original(staged, final)

    monkeypatch.setattr(publication, "_link_final", fail_second)
    with pytest.raises(OSError, match="injected"):
        publish_artifact_group(tmp_path, phase="tx", request={"request": 1}, artifacts=artifacts)
    assert artifacts[0].path.is_file()
    assert not artifacts[1].path.exists()
    monkeypatch.setattr(publication, "_link_final", original)
    publish_artifact_group(tmp_path, phase="tx", request={"request": 1}, artifacts=artifacts)
    validate_publication_commit(tmp_path, "tx")
    with pytest.raises(RuntimeError, match="request changed"):
        publish_artifact_group(tmp_path, phase="tx", request={"request": 2}, artifacts=artifacts)


def test_publication_discards_unprepared_partial_stage_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    artifact = PublicationArtifact(
        tmp_path / "results/post_norm_intervention_v3_stage.pt", "torch", {"x": torch.ones(2)},
    )
    original = publication._write_bytes_exclusive
    interrupted = False

    def partial_once(path: Path, payload: bytes, *, mode: int) -> None:
        nonlocal interrupted
        if not interrupted and path.name.endswith(".staged"):
            interrupted = True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
            path.chmod(mode)
            raise OSError("injected staged interruption")
        original(path, payload, mode=mode)

    monkeypatch.setattr(publication, "_write_bytes_exclusive", partial_once)
    with pytest.raises(OSError, match="staged interruption"):
        publish_artifact_group(tmp_path, phase="stage", request={"request": 1}, artifacts=(artifact,))
    monkeypatch.setattr(publication, "_write_bytes_exclusive", original)
    publish_artifact_group(tmp_path, phase="stage", request={"request": 1}, artifacts=(artifact,))
    validate_publication_commit(tmp_path, "stage")


def test_publication_recovery_is_forbidden_after_cohort_claim(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    token = tmp_path / "results/post_norm_intervention_v3_certification_cohort_attempt.json"
    token.write_text("{}")
    token.chmod(0o444)
    artifact = PublicationArtifact(
        tmp_path / "results/post_norm_intervention_v3_late.json", "json", {"x": 1},
    )
    with pytest.raises(RuntimeError, match="forbidden after the cohort claim"):
        publish_artifact_group(tmp_path, phase="late", request={"request": 1}, artifacts=(artifact,))


def test_composite_intent_covers_interruption_between_panel_components(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    begin_composite_publication(
        tmp_path, phase="panel", request={"request": 1},
        components=("panel_development", "panel_sealed"),
    )
    artifact = PublicationArtifact(
        tmp_path / "results/post_norm_intervention_v3_dev.json", "json", {"x": 1},
    )
    publish_artifact_group(
        tmp_path, phase="panel_development", request={"request": 1}, artifacts=(artifact,),
    )
    assert incomplete_publication_phases(tmp_path) == ["panel"]


def test_transaction_root_rejects_symlinked_parent(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".post_norm_intervention_v3_transactions").symlink_to(
        outside, target_is_directory=True,
    )
    artifact = PublicationArtifact(
        tmp_path / "results/post_norm_intervention_v3_symlink.json", "json", {"x": 1},
    )
    with pytest.raises(RuntimeError, match="transaction root is a symlink"):
        publish_artifact_group(
            tmp_path, phase="symlink", request={"request": 1}, artifacts=(artifact,),
        )


def test_transaction_recovery_rejects_symlink_child_before_mutation(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    transaction_root = physical_transaction_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("must remain")
    (transaction_root / "atomic").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="nonphysical entry"):
        recover_committed_transaction_aliases(tmp_path)
    assert sentinel.read_text() == "must remain"


def test_transaction_recovery_removes_only_proven_final_hardlink_aliases(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    final = results / "post_norm_intervention_v3_cleanup.json"
    publish_artifact_group(
        tmp_path, phase="cleanup", request={"request": 1},
        artifacts=(PublicationArtifact(final, "json", {"x": 1}),),
    )
    phase = physical_transaction_directory(tmp_path, "cleanup")
    staged = phase / f"{final.name}.staged"
    os.link(final, staged)
    atomic = physical_transaction_directory(tmp_path, "atomic")
    temporary = atomic / f"{final.name}.1234.abcdef.tmp"
    os.link(final, temporary)
    assert any(transaction_root_inventory(tmp_path).values())
    assert recover_committed_transaction_aliases(tmp_path)
    assert not staged.exists() and not temporary.exists()
    assert final.is_file()


def test_terminal_summary_recovers_post_link_crash_alias_and_reproduces(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    physical_transaction_root(tmp_path)
    final = results / "post_norm_intervention_v3_terminal_summary.json"
    value = {"schema": "fixture", "complete": True}
    common.write_json_exclusive(final, value, mode=0o444)
    atomic = physical_transaction_directory(tmp_path, "atomic")
    alias = atomic / f"{final.name}.1234.deadbeef.tmp"
    os.link(final, alias)
    assert alias.stat().st_ino == final.stat().st_ino
    assert recover_json_alias_for_existing_final(tmp_path, final) is True
    assert not alias.exists()
    publish_or_reproduce_summary(
        tmp_path, final, value,
        expected_transaction_inventory=transaction_root_inventory(tmp_path),
    )
    assert not any(transaction_root_inventory(tmp_path).values())


def test_terminal_summary_alias_recovery_accepts_canonical_relative_cli_path(
    tmp_path: Path, monkeypatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    physical_transaction_root(tmp_path)
    final = results / "post_norm_intervention_v3_terminal_summary.json"
    common.write_json_exclusive(final, {"complete": True}, mode=0o444)
    atomic = physical_transaction_directory(tmp_path, "atomic")
    os.link(final, atomic / f"{final.name}.1234.feedface.tmp")
    monkeypatch.chdir(tmp_path)
    assert recover_json_alias_for_existing_final(
        tmp_path, tmp_path / Path("results/post_norm_intervention_v3_terminal_summary.json")
    ) is True


def test_engineering_summary_preserves_recorded_unrelated_transaction_aliases(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    unrelated_dir = physical_transaction_directory(tmp_path, "interrupted")
    unrelated = unrelated_dir / "unknown.staged"
    unrelated.write_bytes(b"metadata-only retained alias")
    unrelated.chmod(0o400)
    atomic = physical_transaction_directory(tmp_path, "atomic")
    expected = transaction_root_inventory(tmp_path)
    final = results / "post_norm_intervention_v3_terminal_summary.json"
    value = {"schema": "engineering-fixture", "transaction_root_inventory": expected}
    publish_or_reproduce_summary(
        tmp_path, final, value, expected_transaction_inventory=expected,
    )
    os.link(final, atomic / f"{final.name}.1234.cafebabe.tmp")
    assert recover_json_alias_for_existing_final(tmp_path, final) is True
    publish_or_reproduce_summary(
        tmp_path, final, value, expected_transaction_inventory=expected,
    )
    assert transaction_root_inventory(tmp_path) == expected


def test_claim_byte_validates_public_inputs_but_not_sealed_groups() -> None:
    source = (ROOT / "athena/claim_post_norm_intervention_v3_cohort.py").read_text()
    assert 'root, "proposal", verify_artifact_bytes=True' in source
    assert 'root, "panel_development", verify_artifact_bytes=True' in source
    assert 'root, "panel_sealed", verify_artifact_bytes=False' in source
    assert 'root, "seal", verify_artifact_bytes=False' in source
    assert 'payload_hashes[str(seed)] = payload_content_sha' in source
    assert 'manifest.get("payload_file_sha256")' in source
    assert "claim_certification_attempt(" in source


def test_certification_start_is_exclusive_and_bound_to_array_identity(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    path = tmp_path / "results/post_norm_intervention_v3_certification_start_s0.json"
    sources = {"source.py": "a" * 64}
    value = claim_certification_start(
        path, seed=0, attempt_token_sha256="b" * 64,
        certification_manifest_sha256="c" * 64,
        final_freeze_sha256="d" * 64, frozen_source_hashes=sources,
        slurm_array_job_id="123", slurm_array_task_id="0", slurm_job_id="456",
    )
    validate_certification_start(
        value, expected_seed=0, expected_attempt_token_sha256="b" * 64,
        expected_manifest_sha256="c" * 64, expected_final_freeze_sha256="d" * 64,
        expected_source_hashes=sources, expected_array_job_id="123",
    )
    with pytest.raises(FileExistsError, match="already started"):
        claim_certification_start(
            path, seed=0, attempt_token_sha256="b" * 64,
            certification_manifest_sha256="c" * 64,
            final_freeze_sha256="d" * 64, frozen_source_hashes=sources,
            slurm_array_job_id="123", slurm_array_task_id="0", slurm_job_id="456",
        )


def test_interrupted_failure_partial_is_publicly_classifiable(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    path = results / "post_norm_intervention_v3_prepare_infeasible_s1.partial.json"
    value = {
        "schema": "xvla-post-norm-intervention-infeasibility-partial-v3",
        "failure_phase": "prepare_development",
        "seed": 1,
        "planned_json_path": str(
            results / "post_norm_intervention_v3_prepare_infeasible_s1.json"
        ),
        "planned_npz_path": str(
            results / "post_norm_intervention_v3_prepare_infeasible_s1.npz"
        ),
        "request_sha256": "a" * 64,
    }
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    assert validate_interrupted_failure_partial(tmp_path, path) == value


def test_development_consumers_never_validate_the_sealed_panel_commit() -> None:
    prepare = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    freeze = (ROOT / "athena/freeze_post_norm_intervention_v3.py").read_text()
    assert 'validate_publication_commit(root, "panel_development")' in prepare
    assert 'validate_publication_commit(root, "panel_development")' in freeze
    assert "panel_sealed" not in prepare
    assert "panel_sealed" not in freeze


def test_exact_phase_classifier_rejects_partial_broker_and_panel_and_writable_artifacts(
    tmp_path: Path,
) -> None:
    def frozen(name: str) -> Path:
        path = tmp_path / name
        path.write_bytes(b"x")
        path.chmod(0o444)
        return path

    partial_broker = {
        path.name: path for path in (
            frozen("post_norm_intervention_v3_final_freeze.json"),
            frozen("post_norm_intervention_v3_certification_manifest_s0.json"),
        )
    }
    with pytest.raises(RuntimeError, match="broker"):
        validate_exact_phase_state(partial_broker, recoverable_certification_partials=set())
    partial_panel_path = frozen("post_norm_intervention_v3_panel.json")
    with pytest.raises(RuntimeError, match="panel is only partly"):
        validate_exact_phase_state(
            {partial_panel_path.name: partial_panel_path},
            recoverable_certification_partials=set(),
        )
    writable = tmp_path / "post_norm_intervention_v3_exclusion.json"
    writable.write_bytes(b"x")
    writable.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode differs"):
        validate_exact_phase_state({writable.name: writable}, recoverable_certification_partials=set())


@pytest.mark.parametrize(
    ("name", "wrong_mode"),
    [
        ("post_norm_intervention_v3_bundle_s0.pt", 0o444),
        ("post_norm_intervention_v3_prepare_s0.json", 0o400),
        ("post_norm_intervention_v3_prepare_s0.json", 0o555),
    ],
)
def test_inventory_rejects_noncanonical_frozen_modes(
    tmp_path: Path, name: str, wrong_mode: int,
) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    path.chmod(wrong_mode)
    with pytest.raises(RuntimeError, match="mode differs"):
        exact_results_inventory({name: path}, claimed_seeds=set())


def test_summary_static_open_order_guards_sealed_loads_with_prevalidated_claims() -> None:
    source = (ROOT / "athena/summarize_post_norm_intervention_v3.py").read_text()
    claim_validation = source.index("claimed_seeds = validate_public_attempt_claims(")
    inventory = source.index("inventory_start = exact_results_inventory(")
    sealed_load = source.index('certification_tensor = torch.load(', inventory)
    assert claim_validation < inventory < sealed_load
    early_start = source.index("if args.final_freeze.is_file():")
    early_final = source[early_start:source.index("elif args.final_freeze.exists():", early_start)]
    assert "torch.load(certification_data_path" not in early_final
    assert "torch.load(payload_path" not in early_final
    assert "file_sha256(certification_data_path)" not in early_final
    assert "file_sha256(payload_path)" not in early_final
    proposal_start = source.index("if not peer_preparation_scientific_stop and args.proposal_freeze.is_file():")
    claimed_final = source.index(
        "if args.final_freeze.is_file() and claimed_seeds == set(CHECKPOINTS):",
        proposal_start,
    )
    unclaimed_prefix = source[proposal_start:claimed_final]
    assert 'panel_path.read_text()' not in unclaimed_prefix
    assert 'sealed_manifest_path.read_text()' not in unclaimed_prefix
    prepared_start = source.index("def validate_live_prepared(")
    prepared_end = source.index("\ndef main()", prepared_start)
    prepared_helper = source[prepared_start:prepared_end]
    assert "post_norm_intervention_v3_panel.json" not in prepared_helper
    assert "validate_fresh_panel_manifest" not in prepared_helper
    assert "validate_panel_development_membership" not in prepared_helper


def test_production_benchmark_uses_exact_candidate_path_and_honest_extrapolation() -> None:
    source = (ROOT / "athena/benchmark_post_norm_intervention_v3.py").read_text()
    for call in (
        "_haar_projector(", "_unit_deltas_by_fold(", "_pooled_rms_from_unit_deltas(",
        "_cached_candidate_development_passes(", "projector_cosine(",
    ):
        assert call in source
    assert '"hidden_dim": HIDDEN_DIM' in source
    assert 'torch.randn(HIDDEN_DIM, ACTION_DIM' in source
    assert '"observable_rank": ACTION_DIM' in source
    assert '"complete_quintets_by_alpha"' in source
    assert '"pairwise_separation_checks"' in source
    assert '"pairwise_separation_path_exercised"' in source
    assert "torch.eye(HIDDEN_DIM" not in source
    assert '"horizon": HORIZON' in source
    assert '"elementwise_gates_per_fold_candidate_beta": 728' in source
    assert '"linear_extrapolated_16384_candidate_seconds"' in source
    assert "not a production wall-time guarantee" in source


def test_successful_prepare_to_freeze_gate_inventory_excludes_primary() -> None:
    tensor_source = (ROOT / "xvla/train/post_norm_intervention_v3.py").read_text()
    prepare_source = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    freeze_source = (ROOT / "athena/freeze_post_norm_intervention_v3.py").read_text()
    assert "gate_results: dict[str, tuple[GateResult, ...]] = {}" in tensor_source
    assert "for label, gates in frozen.development_gate_results.items()" in prepare_source
    assert 'set(bundle.get("development_gate_results", {})) != set(HAAR_CONDITIONS)' in freeze_source
    assert "{primary.label: ()}" not in tensor_source


def test_preparation_ancestry_is_frozen_before_search_and_rejects_query_tamper() -> None:
    sources = {"source.py": "a" * 64}
    ancestry = {
        "schema": SCHEMA_PREPARATION_ANCESTRY, "seed": 0,
        "checkpoint_sha256": CHECKPOINTS[0]["sha256"], "cache_sha256": CACHE["sha256"],
        "source_hashes": sources, "model_state_sha256": "1" * 64,
        "panel_sha256": "2" * 64, "exclusion_manifest_sha256": "3" * 64,
        "discovery_data_sha256": "4" * 64, "development_data_sha256": "5" * 64,
        "data_manifest_sha256": "6" * 64, "discovery_queries_sha256": "7" * 64,
        "discovery_means_sha256": "8" * 64, "simulator_rollouts_started": False,
        "content_sha256": None,
    }
    ancestry["content_sha256"] = canonical_sha256({
        key: value for key, value in ancestry.items() if key != "content_sha256"
    })
    validate_preparation_ancestry(ancestry, expected_seed=0, expected_source_hashes=sources)
    tampered = copy.deepcopy(ancestry)
    tampered["discovery_queries_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="content hash"):
        validate_preparation_ancestry(tampered, expected_seed=0, expected_source_hashes=sources)
    prepare = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    assert prepare.index("write_json_exclusive(args.ancestry") < prepare.index("freeze_development_proposals(")
    assert "--ancestry" in (ROOT / "athena/slurm_prepare_post_norm_intervention_v3.sbatch").read_text()
    assert "--ancestries" in (ROOT / "athena/slurm_freeze_post_norm_intervention_v3.sbatch").read_text()


def test_exclusive_writers_set_final_mode_before_publication() -> None:
    common = (ROOT / "athena/post_norm_intervention_v3_common.py").read_text()
    publication_source = (ROOT / "athena/post_norm_intervention_v3_publication.py").read_text()
    common_writer = common[common.index("def write_json_exclusive("):]
    assert common_writer.index("os.fchmod(handle.fileno(), mode)") < common_writer.index("os.link(temporary, path)")
    stage_writer = publication_source[
        publication_source.index("def _write_bytes_exclusive("):
        publication_source.index("def _content(")
    ]
    assert "os.fchmod(handle.fileno(), mode)" in stage_writer
    assert publication_source.index("_write_bytes_exclusive(staged") < publication_source.index("_link_final(staged, final)")
    assert "mode=0o444" in common[common.index("def claim_certification_attempt("):common.index("def validate_predecessor_exclusion_manifest(")]


def _stored_profile() -> dict:
    nested = {
        "aggregate_rms": 1.0,
        "coordinate_rms": [1.0] * 7,
        "group_rms": {"translation": 1.0, "rotation": 1.0, "gripper": 1.0},
        "horizon_coordinate_rms": [[1.0] * 7 for _ in range(HORIZON)],
        "coordinate_second_moment": [[1.0] * 7 for _ in range(7)],
        "gripper_flip_fraction": 0.1,
        "gripper_margin_reachable_fraction": 0.1,
        "task_profiles": {},
    }
    value = copy.deepcopy(nested)
    value["task_profiles"] = {str(task): copy.deepcopy(nested) for task in range(10)}
    return value


def test_terminal_summary_recomputes_every_certification_gate_from_profiles() -> None:
    profiles = {label: _stored_profile() for label in DEPLOYED_CONDITIONS}
    parsed = {label: action_profile_from_dict(profile) for label, profile in profiles.items()}
    gates = {
        label: [gate.as_dict() for gate in compare_action_profiles(
            parsed[label], parsed["balanced_top"], gates=CERTIFICATION_GATES,
            location=f"sealed_certification/condition={label}",
        )]
        for label in HAAR_CONDITIONS
    }
    value = {
        "status": "pass", "operator_sha256": {label: "a" * 64 for label in DEPLOYED_CONDITIONS},
        "alphas": {label: 0.5 for label in DEPLOYED_CONDITIONS},
        "gate_results": gates, "profiles": profiles, "reason": None, "maximum_offender": None,
    }
    validate_certification_scientific_payload(value)
    tampered = copy.deepcopy(value)
    tampered["gate_results"][HAAR_CONDITIONS[0]][0]["actual"] = 0.01
    tampered["gate_results"][HAAR_CONDITIONS[0]][0]["signed_margin"] -= 0.01
    with pytest.raises(RuntimeError, match="recompute"):
        validate_certification_scientific_payload(tampered)


def test_allocation_light_design_predicate_is_elementwise_equivalent() -> None:
    reference = action_profile_from_dict(_stored_profile())
    candidate = action_profile_from_dict(_stored_profile())
    materialized = compare_action_profiles(
        candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
    )
    assert len(materialized) == 728
    assert action_profiles_pass(
        candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
    ) is all(gate.passed for gate in materialized)


def test_vectorized_design_predicate_matches_every_materialized_gate_family() -> None:
    reference = action_profile_from_dict(_stored_profile())
    mutations = (
        lambda value: value.__setitem__("aggregate_rms", 2.0),
        lambda value: value["group_rms"].__setitem__("translation", 2.0),
        lambda value: value["coordinate_rms"].__setitem__(3, 2.0),
        lambda value: value["horizon_coordinate_rms"][4].__setitem__(2, 2.0),
        lambda value: value["coordinate_second_moment"][2].__setitem__(5, 2.0),
        lambda value: value.__setitem__("gripper_flip_fraction", 0.9),
        lambda value: value["task_profiles"]["4"]["group_rms"].__setitem__("rotation", 2.0),
        lambda value: value["task_profiles"]["4"]["horizon_coordinate_rms"][3].__setitem__(1, 2.0),
        lambda value: value["task_profiles"]["4"].__setitem__("gripper_margin_reachable_fraction", 0.9),
    )
    for mutate in mutations:
        raw = _stored_profile()
        mutate(raw)
        candidate = action_profile_from_dict(raw)
        materialized = compare_action_profiles(
            candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
        )
        assert action_profiles_pass(
            candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
        ) is all(gate.passed for gate in materialized)
    changed = _stored_profile()
    changed["task_profiles"]["9"]["horizon_coordinate_rms"][7][6] = 10.0
    candidate = action_profile_from_dict(changed)
    materialized = compare_action_profiles(
        candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
    )
    assert action_profiles_pass(
        candidate, reference, gates=CERTIFICATION_GATES, location="benchmark",
    ) is all(gate.passed for gate in materialized)


def test_launcher_is_held_transaction_and_contains_no_simulator_stage() -> None:
    source = (ROOT / "athena/launch_post_norm_intervention_v3.py").read_text()
    assert '"--hold"' in source
    assert 'reverse_release_order = list(reversed(DAG_ORDER))' in source
    assert 'write_json_exclusive(launch_path, launch, mode=0o444)' in source
    assert source.index('write_json_exclusive(launch_path, launch, mode=0o444)') < source.index('reverse_release_order = list(reversed(DAG_ORDER))')
    assert '"simulator_jobs_submitted": False' in source
    assert 'jobs["smoke"]' not in source
    assert 'jobs["rollout"]' not in source
    assert 'jobs["aggregate"]' not in source
    assert 'dependency=f"afterany:{all_jobs}"' in source
    assert '"--kill-on-invalid-dep=yes"' in source
    assert 'write_json_exclusive(release_path, release, mode=0o444)' in source
    assert 'write_json_exclusive(release_intent_path, release_intent, mode=0o444)' in source
    assert source.index('write_json_exclusive(release_path, release, mode=0o444)') < source.index('subprocess.run(["scontrol", "release", jobs["release_gate"]]')
    assert 'capture_output=True' in source
    assert 'jobs["release_gate"] = submit(root, "release_gate")' in source
    assert 'dependency=f"afterok:{jobs[\'release_gate\']}"' in source
    assert "os.chmod(stage5_path, 0o444)" in source
    assert "STAGE5_RESULT_SHA256" in source
    panel_source = (ROOT / "athena/build_post_norm_intervention_v3_panel.py").read_text()
    assert 'require_read_only_regular(stage5_path, label="accepted Stage 5 predecessor")' in panel_source
    verifier = (ROOT / "athena/verify_post_norm_intervention_v3_stage.py").read_text()
    summary = (ROOT / "athena/summarize_post_norm_intervention_v3.py").read_text()
    assert '_require_read_only(stage5_path, "accepted Stage 5 identity source")' in verifier
    assert "file_sha256(stage5_path) != STAGE5_RESULT_SHA256" in verifier
    assert '"stage5_result_sha256": STAGE5_RESULT_SHA256' in summary
    assert 'file_sha256(stage5_path) != launch_identity["stage5_result_sha256"]' in summary


def test_release_completion_is_required_by_every_downstream_stage() -> None:
    verifier = (ROOT / "athena/verify_post_norm_intervention_v3_stage.py").read_text()
    gate = (ROOT / "athena/slurm_release_gate_post_norm_intervention_v3.sbatch").read_text()
    summary = (ROOT / "athena/summarize_post_norm_intervention_v3.py").read_text()
    assert 'completion_path = results / "post_norm_intervention_v3_release_complete.json"' in verifier
    assert '_require_read_only(completion_path, "release completion")' in verifier
    assert "--release-gate" in gate
    assert "commit_post_norm_intervention_v3_release.py" in gate
    assert '"release_completion_sha256"' in verifier
    assert "validate_release_transaction(root, launch_record, release_gate=False)" in summary
    assert "preparation_completed_but_not_frozen_due_to_peer_scientific_stop" in summary
    assert "preparation_not_observed_due_to_peer_scientific_stop" in summary
    assert "preparation_completed_but_proposal_freeze_engineering_failure" in summary
    assert 'and args.proposal_freeze.is_file()' in summary
    assert "validate_failure_identities" in summary
    assert "failure receipt numeric companion path is not canonical" in summary
    assert "preparation failure coexists with same-seed success" in summary
    assert "panel failure coexists with successful or downstream artifacts" in summary
    assert 'Path(proposal_freeze["development_data_path"])' in summary
    assert "validate_panel_development_membership" in summary
    assert "proposal input s{seed}" in summary
    assert "terminal state contains an incomplete failure transaction" in summary
    assert "recoverable_certification_partials" in summary
    assert "recovered certification transaction retained a partial marker" in summary
    assert "result inventory changed outside exact summary transaction recovery" in summary
    assert "transaction_root_inventory(root)" in summary
    panel_branch = summary[
        summary.index("if args.panel_failure.exists():"):
        summary.index("terminal: dict[str, Any] = {}")
    ]
    assert '"transaction_root_inventory": transaction_inventory_start' in panel_branch
    assert "transaction_root_inventory(root) != transaction_inventory_start" in panel_branch
    assert "engineering_failure_after_certification_start" in summary
    assert "absent proposal freeze coexists with downstream artifacts" in summary
    assert "final freeze does not bind its complete live manifest chain" in summary
    assert "final freeze sealed-data public ancestry differs" in summary
    assert "claimed sealed certification tensor" in summary
    assert "engineering stop contains an incomplete preparation pair" in summary
    assert '"proposal_freeze_sha256": None' in summary
    assert "true_indices != ordered_indices" in summary
    assert "namespaced_seed(SAMPLE_NAMESPACE" in summary
    assert "frozen projector separation gate failed" in summary
    assert '"development_data_sha256", "data_manifest_sha256", "chosen_beta"' in summary
    assert "live prepared discovery/development normalization differs" in summary


def test_freeze_chain_artifacts_are_made_read_only_and_enforced() -> None:
    prepare = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    freeze = (ROOT / "athena/freeze_post_norm_intervention_v3.py").read_text()
    seal = (ROOT / "athena/seal_post_norm_intervention_v3.py").read_text()
    certify = (ROOT / "athena/certify_post_norm_intervention_v3.py").read_text()
    publication = (ROOT / "athena/post_norm_intervention_v3_publication.py").read_text()
    assert 'PublicationArtifact(args.bundle, "torch", bundle)' in prepare
    assert 'PublicationArtifact(args.record, "json", record)' in prepare
    assert "write_json_exclusive(args.ancestry, ancestry, mode=0o444)" in prepare
    assert 'PublicationArtifact(args.freeze, "json", freeze)' in freeze
    assert "require_read_only_regular(path, label=\"proposal-freeze input\")" in freeze
    assert 'PublicationArtifact(payload_path, "torch", payload)' in seal
    assert 'PublicationArtifact(manifest_path, "json", manifest)' in seal
    assert 'PublicationArtifact(args.final_freeze, "json", final)' in seal
    assert 'KINDS = {"json": 0o444, "torch": 0o400}' in publication
    assert "os.fchmod(handle.fileno(), mode)" in publication
    assert "_link_final(staged, final)" in publication
    assert "require_read_only_regular(path, label=label)" in certify
    assert '"alphas": record["alphas"]' in freeze
    assert 'raw["alphas"] != prepared["alphas"]' in seal
    assert '"operator_sha256": raw["operator_sha256"], "alphas": raw["alphas"]' in seal


def test_failure_class_phase_map_and_representative_preflight_are_frozen() -> None:
    common = (ROOT / "athena/post_norm_intervention_v3_common.py").read_text()
    benchmark = (ROOT / "athena/benchmark_post_norm_intervention_v3.py").read_text()
    assert '"ordered_greedy_design_shortfall": "prepare_development"' in common
    assert '"ordered_greedy_sample_shortfall": "prepare_development"' in common
    assert '"numerical_gate_infeasible": "prepare_development"' in common
    assert '"certification_infeasible": "certify_once"' in common
    assert 'default=32' in benchmark
    assert 'for proposal_index in range(args.measured_candidates)' in benchmark
    assert 'linear_extrapolated_16384_candidate_seconds' in benchmark
    assert 'results_root in resolved_output.parents' in benchmark
    panel = (ROOT / "athena/build_post_norm_intervention_v3_panel.py").read_text()
    prepare = (ROOT / "athena/prepare_post_norm_intervention_v3.py").read_text()
    assert "panel succeeded but an earlier failure transaction exists" in panel
    assert "preparation succeeded but an earlier failure transaction exists" in prepare


def test_every_v3_consumer_is_in_source_closure() -> None:
    common = (ROOT / "athena/post_norm_intervention_v3_common.py").read_text()
    for name in (
        "build_post_norm_intervention_v3_panel.py",
        "post_norm_intervention_v3_publication.py",
        "prepare_post_norm_intervention_v3.py",
        "freeze_post_norm_intervention_v3.py",
        "seal_post_norm_intervention_v3.py",
        "claim_post_norm_intervention_v3_cohort.py",
        "certify_post_norm_intervention_v3.py",
        "summarize_post_norm_intervention_v3.py",
        "launch_post_norm_intervention_v3.py",
        "verify_post_norm_intervention_v3_stage.py",
        "commit_post_norm_intervention_v3_release.py",
        "slurm_release_gate_post_norm_intervention_v3.sbatch",
        "slurm_post_norm_intervention_tests_v3.sbatch",
        "slurm_post_norm_intervention_exclusion_v3.sbatch",
        "slurm_post_norm_intervention_panel_v3.sbatch",
        "slurm_prepare_post_norm_intervention_v3.sbatch",
        "slurm_freeze_post_norm_intervention_v3.sbatch",
        "slurm_seal_post_norm_intervention_v3.sbatch",
        "slurm_claim_post_norm_intervention_v3_cohort.sbatch",
        "slurm_certify_post_norm_intervention_v3.sbatch",
        "slurm_summary_post_norm_intervention_v3.sbatch",
        "benchmark_post_norm_intervention_v3.py",
        "slurm_benchmark_post_norm_intervention_v3.sbatch",
    ):
        assert name in common


def test_athena_preflight_is_timing_only_and_outside_scientific_dag() -> None:
    launcher = (ROOT / "athena/launch_post_norm_intervention_v3.py").read_text()
    benchmark = (ROOT / "athena/benchmark_post_norm_intervention_v3.py").read_text()
    assert 'os.environ.get("SLURM_JOB_ID")' in benchmark
    assert "elementwise_gates_per_fold_candidate_beta" in benchmark
    assert "_cached_candidate_development_passes" in benchmark
    assert '"benchmark"' not in launcher.split("DAG_ORDER", 1)[1].split("SCRIPTS", 1)[0]


def test_every_sbatch_disables_bytecode_and_verifies_launch_manifest_first() -> None:
    names = (
        "slurm_release_gate_post_norm_intervention_v3.sbatch",
        "slurm_post_norm_intervention_tests_v3.sbatch",
        "slurm_post_norm_intervention_exclusion_v3.sbatch",
        "slurm_post_norm_intervention_panel_v3.sbatch",
        "slurm_prepare_post_norm_intervention_v3.sbatch",
        "slurm_freeze_post_norm_intervention_v3.sbatch",
        "slurm_seal_post_norm_intervention_v3.sbatch",
        "slurm_claim_post_norm_intervention_v3_cohort.sbatch",
        "slurm_certify_post_norm_intervention_v3.sbatch",
        "slurm_summary_post_norm_intervention_v3.sbatch",
    )
    for name in names:
        source = (ROOT / "athena" / name).read_text()
        assert "PYTHONDONTWRITEBYTECODE=1" in source
        assert "python -B athena/verify_post_norm_intervention_v3_stage.py" in source
        assert source.index("python -B athena/verify_post_norm_intervention_v3_stage.py") < source.rindex("python -B")
