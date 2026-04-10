from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path("/opt/project")
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
REGISTRY_DIR = MODELS_DIR / "registry"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = OUTPUT_DIR / "model_registry.json"

MODEL_CONFIG = {
    "customer_segmentation": {
        "artifacts": [
            MODELS_DIR / "customer_segmentation_kmeans.pkl",
            MODELS_DIR / "customer_segmentation_scaler.pkl",
            MODELS_DIR / "customer_segmentation_metrics.json",
        ],
        "metrics_file": MODELS_DIR / "customer_segmentation_metrics.json",
        "task_type": "clustering",
    },
    "customer_anomaly_detection": {
        "artifacts": [
            MODELS_DIR / "anomaly_isolation_forest.pkl",
            MODELS_DIR / "anomaly_lof.pkl",
            MODELS_DIR / "anomaly_ocsvm.pkl",
            MODELS_DIR / "anomaly_scaler.pkl",
            OUTPUT_DIR / "anomaly_summary.json",
        ],
        "metrics_file": OUTPUT_DIR / "anomaly_summary.json",
        "task_type": "anomaly_detection",
    },
    "customer_risk_scoring": {
        "artifacts": [
            OUTPUT_DIR / "risk_bucket_summary.json",
        ],
        "metrics_file": OUTPUT_DIR / "risk_bucket_summary.json",
        "task_type": "scoring",
    },
    "spend_prediction": {
        "artifacts": [
            MODELS_DIR / "spend_prediction_random_forest.pkl",
            OUTPUT_DIR / "spend_prediction_metrics.json",
            OUTPUT_DIR / "spend_feature_importance.csv",
        ],
        "metrics_file": OUTPUT_DIR / "spend_prediction_metrics.json",
        "task_type": "regression",
    },
}


def safe_read_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def safe_write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def next_version(existing_versions: list[str]) -> str:
    nums = []
    for v in existing_versions:
        if v.startswith("v"):
            try:
                nums.append(int(v[1:]))
            except ValueError:
                pass
    return f"v{max(nums, default=0) + 1}"


def copy_artifacts(model_name: str, version: str, artifact_paths: list[Path]) -> list[str]:
    target_dir = REGISTRY_DIR / model_name / version
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in artifact_paths:
        if src.exists():
            dst = target_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    print("===== Model Registry Started =====")

    registry = safe_read_json(REGISTRY_FILE)
    if "models" not in registry:
        registry["models"] = {}

    run_timestamp = datetime.now(timezone.utc).isoformat()

    for model_name, config in MODEL_CONFIG.items():
        print(f"\nProcessing model: {model_name}")

        model_entry = registry["models"].get(model_name, {})
        versions = model_entry.get("versions", {})
        new_version = next_version(list(versions.keys()))

        metrics = safe_read_json(config["metrics_file"])
        copied_artifacts = copy_artifacts(model_name, new_version, config["artifacts"])

        version_record = {
            "version": new_version,
            "registered_at_utc": run_timestamp,
            "task_type": config["task_type"],
            "metrics": metrics,
            "artifacts": copied_artifacts,
            "status": "candidate",
        }

        if not copied_artifacts:
            print(f"Skipped {model_name}: no artifacts found.")
            continue

        if not model_entry:
            registry["models"][model_name] = {
                "latest_version": new_version,
                "production_version": new_version,
                "versions": {
                    new_version: version_record
                },
            }
        else:
            registry["models"][model_name]["latest_version"] = new_version
            registry["models"][model_name]["versions"][new_version] = version_record

            if "production_version" not in registry["models"][model_name]:
                registry["models"][model_name]["production_version"] = new_version

        print(f"Registered {model_name} as {new_version}")
        print(f"Copied artifacts: {len(copied_artifacts)}")

    safe_write_json(REGISTRY_FILE, registry)

    print("\n===== Model Registry Complete =====")
    print(f"Saved registry to: {REGISTRY_FILE}")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()