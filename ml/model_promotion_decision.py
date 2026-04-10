from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_FILE = Path("/opt/project/outputs/model_registry.json")
OUTPUT_FILE = Path("/opt/project/outputs/model_promotion_decision.json")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_spend_model_info(registry: dict) -> tuple[dict, str, dict, str, dict]:
    models = registry.get("models", {})
    spend_entry = models.get("spend_prediction")
    if not spend_entry:
        raise ValueError("spend_prediction entry not found in model registry")

    latest_version = spend_entry.get("latest_version")
    production_version = spend_entry.get("production_version")

    versions = spend_entry.get("versions", {})
    latest_info = versions.get(latest_version)
    production_info = versions.get(production_version)

    if not latest_info or not production_info:
        raise ValueError("Missing latest or production version info for spend_prediction")

    return spend_entry, latest_version, latest_info, production_version, production_info


def main() -> None:
    print("===== Model Promotion Decision Started =====")

    registry = load_json(REGISTRY_FILE)

    spend_entry, latest_version, latest_info, production_version, production_info = get_spend_model_info(registry)

    latest_metrics = latest_info.get("metrics", {})
    production_metrics = production_info.get("metrics", {})

    latest_r2 = latest_metrics.get("r2")
    production_r2 = production_metrics.get("r2")

    should_promote = False
    reasons = []

    if latest_version == production_version:
        reasons.append("Latest version is already the production version.")
    elif latest_r2 is None or production_r2 is None:
        reasons.append("Missing R2 metric for comparison; no promotion performed.")
    elif float(latest_r2) > float(production_r2):
        should_promote = True
        reasons.append(
            f"Promoting latest version because latest R2 ({latest_r2:.6f}) > production R2 ({production_r2:.6f})."
        )
    else:
        reasons.append(
            f"Keeping current production version because latest R2 ({latest_r2:.6f}) <= production R2 ({production_r2:.6f})."
        )

    if should_promote:
        registry["models"]["spend_prediction"]["production_version"] = latest_version
        registry["models"]["spend_prediction"]["versions"][latest_version]["status"] = "production"
        if production_version in registry["models"]["spend_prediction"]["versions"]:
            registry["models"]["spend_prediction"]["versions"][production_version]["status"] = "candidate"

        save_json(REGISTRY_FILE, registry)

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "spend_prediction",
        "latest_version": latest_version,
        "production_version_before_decision": production_version,
        "production_version_after_decision": registry["models"]["spend_prediction"]["production_version"],
        "latest_r2": latest_r2,
        "production_r2_before_decision": production_r2,
        "should_promote": should_promote,
        "reasons": reasons,
    }

    save_json(OUTPUT_FILE, decision)

    print("\n===== Model Promotion Decision Complete =====")
    print(f"Saved promotion decision to: {OUTPUT_FILE}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()