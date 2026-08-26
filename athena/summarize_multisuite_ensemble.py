#!/usr/bin/env python3
"""Strictly aggregate the frozen three-checkpoint multi-suite mean ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from athena import summarize_multisuite_generalist as generalist


SHARDS = ((0, 3), (3, 6), (6, 8), (8, 10))
SCHEMA = "xvla_multisuite_mean_ensemble_manifest_v1"
RESULT_PREFIX = "multisuite_mean_ensemble_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multisuite_mean_ensemble_v1_summary.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_equal(actual: Any, expected: Any, label: str) -> None:
    generalist.require_equal(actual, expected, label)


def result_path(results_dir: Path, suite: str, start: int, end: int) -> Path:
    return results_dir / f"{RESULT_PREFIX}_{suite}_t{start}_{end}.json"


def validate_manifest_live(manifest: dict[str, Any]) -> None:
    require_equal(manifest.get("schema"), SCHEMA, "ensemble manifest schema")
    require_equal(
        manifest.get("status"),
        "frozen_after_member_evaluations_before_ensemble_outcomes",
        "ensemble manifest status",
    )
    require_equal(manifest["scope"]["seeds"], list(generalist.SEEDS), "ensemble seeds")
    require_equal(manifest["scope"]["suites"], list(generalist.SUITES), "ensemble suites")
    require_equal(
        manifest["generalist_manifest"]["sha256"],
        generalist.MANIFEST_SHA256,
        "generalist manifest pin",
    )
    require_equal(
        generalist.file_sha256(generalist.MANIFEST_PATH),
        generalist.MANIFEST_SHA256,
        "live generalist manifest",
    )
    for path, digest in manifest["source_files"].items():
        require_equal(generalist.file_sha256(Path(path)), digest, f"frozen source {path}")
    for seed in generalist.SEEDS:
        artifact = manifest["artifacts"][str(seed)]
        require_equal(
            generalist.file_sha256(Path(artifact["checkpoint_path"])),
            artifact["checkpoint_sha256"],
            f"seed {seed} checkpoint",
        )
        require_equal(
            generalist.file_sha256(Path(artifact["metadata_path"])),
            artifact["metadata_sha256"],
            f"seed {seed} metadata",
        )
        for source in manifest["member_results"][str(seed)]["source_files"]:
            require_equal(
                generalist.file_sha256(Path(source["path"])),
                source["sha256"],
                f"seed {seed} member result",
            )


def validate_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    suite: str,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_label = f"{suite} tasks {start}:{end}"
    artifacts = manifest["artifacts"]
    checkpoints = [artifacts[str(seed)]["checkpoint_path"] for seed in generalist.SEEDS]
    seed0_metadata = load_json(Path(artifacts["0"]["metadata_path"]))
    require_equal(
        {
            "mode": result.get("mode"),
            "architecture": result.get("architecture"),
            "vision_encoder": result.get("vision_encoder"),
            "suite": result.get("suite"),
            "training_suite": result.get("training_suite"),
            "checkpoint": result.get("checkpoint"),
            "ensemble_checkpoints": result.get("ensemble_checkpoints"),
            "ensemble_reduction": result.get("ensemble_reduction"),
            "model_metadata": result.get("model_metadata"),
            "cache": result.get("cache"),
            "precision": result.get("matmul_precision"),
        },
        {
            "mode": "ensemble_capability",
            "architecture": "chi",
            "vision_encoder": "vit",
            "suite": suite,
            "training_suite": suite,
            "checkpoint": checkpoints[0],
            "ensemble_checkpoints": checkpoints,
            "ensemble_reduction": "mean",
            "model_metadata": artifacts["0"]["metadata_path"],
            "cache": generalist.CACHES[suite]["path"],
            "precision": "highest",
        },
        f"{path_label} identity",
    )
    require_equal(result.get("training_metadata"), seed0_metadata, f"{path_label} metadata")
    require_equal(result.get("prediction_finite"), True, f"{path_label} finite prediction")
    require_equal(result.get("prediction_shape"), [1, 8, 7], f"{path_label} prediction shape")
    require_equal(
        result.get("profile", {}).get("parameters"),
        3 * generalist.EXPECTED_PARAMETERS["chi"],
        f"{path_label} parameter count",
    )
    require_equal(
        result.get("profile", {}).get("gpu"),
        generalist.EXPECTED_EVALUATION_GPU_NAME,
        f"{path_label} GPU",
    )
    require_equal(
        result.get("numerical_environment"),
        generalist.EXPECTED_NUMERICAL_ENVIRONMENT,
        f"{path_label} runtime",
    )
    require_equal(
        result.get("evaluation_protocol"),
        {
            "res": 64,
            "horizon": 8,
            "num_steps_wait": 10,
            "exec_h": 8,
            "eps_per_task": 50,
            "max_steps": 280,
        },
        f"{path_label} protocol",
    )
    source = result.get("source_identity", {})
    require_equal(source.get("expected_evaluator_sha256"), generalist.EVALUATOR_SHA256, f"{path_label} evaluator pin")
    require_equal(source.get("start"), source.get("end"), f"{path_label} source closure")
    require_equal(source["start"].get("evaluator_sha256"), generalist.EVALUATOR_SHA256, f"{path_label} evaluator")
    require_equal(source["start"].get("checkpoint_sha256"), artifacts["0"]["checkpoint_sha256"], f"{path_label} primary checkpoint")
    require_equal(source["start"].get("model_metadata_sha256"), artifacts["0"]["metadata_sha256"], f"{path_label} metadata hash")
    require_equal(source["start"].get("manifest_sha256"), generalist.MANIFEST_SHA256, f"{path_label} generalist manifest")
    require_equal(source["start"].get("evaluation_cache_sha256"), generalist.CACHES[suite]["sha256"], f"{path_label} cache")
    wrapper = result.get("ensemble_run_identity", {})
    require_equal(wrapper.get("manifest_sha256"), manifest_sha256, f"{path_label} ensemble manifest")
    require_equal(wrapper.get("source_files_start"), manifest["source_files"], f"{path_label} wrapper sources start")
    require_equal(wrapper.get("source_files_end"), manifest["source_files"], f"{path_label} wrapper sources end")
    expected_members = {
        str(seed): artifacts[str(seed)]["checkpoint_sha256"]
        for seed in generalist.SEEDS
    }
    require_equal(wrapper.get("member_checkpoint_sha256_start"), expected_members, f"{path_label} members start")
    require_equal(wrapper.get("member_checkpoint_sha256_end"), expected_members, f"{path_label} members end")

    capability = result.get("capability", {})
    tasks = list(range(start, end))
    require_equal(capability.get("task_indices"), tasks, f"{path_label} task indices")
    require_equal(capability.get("eps_per_task"), 50, f"{path_label} episodes per task")
    require_equal(capability.get("max_steps"), 280, f"{path_label} maximum steps")
    require_equal(capability.get("num_steps_wait"), 10, f"{path_label} settle steps")
    require_equal(capability.get("exec_h"), 8, f"{path_label} execution horizon")
    require_equal(capability.get("canonical_init_states"), True, f"{path_label} canonical states")
    protocol = capability.get("task_protocol", {})
    require_equal(set(protocol), {str(task) for task in tasks}, f"{path_label} task protocol")
    for task in tasks:
        require_equal(
            protocol[str(task)],
            manifest["task_protocol"][f"{suite}:{task}"],
            f"{path_label} task {task} assets",
        )
    episodes = capability.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError(f"{path_label} episode rows are missing")
    require_equal(len(episodes), len(tasks) * 50, f"{path_label} episode count")
    seen: set[tuple[int, int]] = set()
    successes = 0
    for episode in episodes:
        task = episode.get("task_index")
        index = episode.get("episode")
        success = episode.get("success")
        steps = episode.get("steps")
        if type(task) is not int or task not in tasks:
            raise RuntimeError(f"{path_label} invalid task row {episode}")
        if type(index) is not int or not 0 <= index < 50:
            raise RuntimeError(f"{path_label} invalid episode row {episode}")
        if type(success) is not bool:
            raise RuntimeError(f"{path_label} non-Boolean outcome {episode}")
        if type(steps) is not int or not 0 <= steps <= 280:
            raise RuntimeError(f"{path_label} invalid step count {episode}")
        key = (task, index)
        if key in seen:
            raise RuntimeError(f"{path_label} duplicate episode {key}")
        seen.add(key)
        successes += int(success)
    require_equal(capability.get("trials"), len(episodes), f"{path_label} trials")
    require_equal(capability.get("successes"), successes, f"{path_label} successes")
    require_equal(capability.get("overall"), successes / len(episodes), f"{path_label} overall")
    return episodes, protocol


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite ensemble summary {args.output}")
    require_equal(generalist.file_sha256(args.manifest), args.manifest_sha256, "ensemble manifest SHA-256")
    manifest = load_json(args.manifest)
    validate_manifest_live(manifest)
    provenance_start = generalist.validate_cache_provenance(args.results_dir)
    require_equal(provenance_start, manifest["cache_provenance"], "cache provenance")

    task_outcomes: dict[tuple[str, int], list[bool]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    source_files: list[dict[str, str]] = []
    for suite in generalist.SUITES:
        for start, end in SHARDS:
            path = result_path(args.results_dir, suite, start, end)
            digest = generalist.file_sha256(path)
            result = load_json(path)
            episodes, _ = validate_result(result, manifest, args.manifest_sha256, suite, start, end)
            source_files.append({"path": path.as_posix(), "sha256": digest})
            for episode in episodes:
                key = (suite, episode["task_index"], episode["episode"])
                if key in seen:
                    raise RuntimeError(f"duplicate ensemble episode {key}")
                seen.add(key)
                task_outcomes[(suite, episode["task_index"])].append(episode["success"])
    require_equal(len(seen), 2000, "ensemble episode coverage")
    expected_cells = {(suite, task) for suite in generalist.SUITES for task in range(10)}
    require_equal(set(task_outcomes), expected_cells, "ensemble task coverage")
    for key, outcomes in task_outcomes.items():
        require_equal(len(outcomes), 50, f"ensemble task {key} trials")

    task_success = {
        f"{suite}:{task}": sum(task_outcomes[(suite, task)]) / 50
        for suite in generalist.SUITES
        for task in range(10)
    }
    suite_success = {
        suite: sum(task_success[f"{suite}:{task}"] for task in range(10)) / 10
        for suite in generalist.SUITES
    }
    macro = sum(task_success.values()) / 40
    tasks_at_least = sum(value >= 0.50 for value in task_success.values())
    member_mean = manifest["member_mean_macro_success"]
    uplift = macro - member_mean
    frozen = manifest["frozen_gates"]
    gates = {
        "macro_task_success_at_least": macro >= frozen["macro_task_success_at_least"],
        "every_suite_macro_at_least": all(value >= frozen["every_suite_macro_at_least"] for value in suite_success.values()),
        "tasks_at_least_0p50_at_least": tasks_at_least >= frozen["tasks_at_least_0p50_at_least"],
        "uplift_over_member_mean_at_least": uplift >= frozen["uplift_over_member_mean_at_least"],
    }
    gates["ensemble_capability_pass"] = all(gates.values())

    validate_manifest_live(manifest)
    require_equal(generalist.file_sha256(args.manifest), args.manifest_sha256, "ensemble manifest closure")
    require_equal(generalist.validate_cache_provenance(args.results_dir), provenance_start, "cache provenance closure")
    for source in source_files:
        require_equal(generalist.file_sha256(Path(source["path"])), source["sha256"], "ensemble result closure")
    output = {
        "scope": manifest["scope"],
        "manifest": {"path": args.manifest.as_posix(), "sha256": args.manifest_sha256},
        "frozen_gates": frozen,
        "task_success": task_success,
        "suite_macro_success": suite_success,
        "macro_task_success": macro,
        "tasks_at_least_0p50": tasks_at_least,
        "member_mean_macro_success": member_mean,
        "uplift_over_member_mean": uplift,
        "gates": gates,
        "successes": sum(sum(outcomes) for outcomes in task_outcomes.values()),
        "trials": 2000,
        "source_files": source_files,
        "identity": {
            "manifest_source_files": manifest["source_files"],
            "manifest_source_bundle_sha256": manifest["source_bundle_sha256"],
            "member_artifacts": manifest["artifacts"],
            "cache_provenance": provenance_start,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
