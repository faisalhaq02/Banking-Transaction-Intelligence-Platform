from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_PATH = Path("/opt/project/outputs/model_registry.json")


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry file not found: {REGISTRY_PATH}")
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def promote_model(model_name: str, version: str, target_status: str = "production") -> None:
    registry = load_registry()

    if "models" not in registry or model_name not in registry["models"]:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model_entry = registry["models"][model_name]
    versions = model_entry.get("versions", {})

    if version not in versions:
        raise ValueError(f"Version '{version}' not found for model '{model_name}'.")

    if target_status not in {"candidate", "staging", "production", "archived"}:
        raise ValueError(
            "target_status must be one of: candidate, staging, production, archived"
        )

    # If promoting to production, archive old production version first
    old_production_version = model_entry.get("production_version")
    if target_status == "production" and old_production_version:
        if old_production_version in versions and old_production_version != version:
            versions[old_production_version]["status"] = "archived"
            versions[old_production_version]["archived_at_utc"] = utc_now()

    # Update selected version
    versions[version]["status"] = target_status
    versions[version]["last_promoted_at_utc"] = utc_now()

    # Update model-level pointers
    model_entry["latest_version"] = version
    if target_status == "production":
        model_entry["production_version"] = version

    save_registry(registry)

    print("===== Model Promotion Complete =====")
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"New status: {target_status}")
    print(f"Registry updated: {REGISTRY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model version in registry.")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--version", required=True, help="Version to promote, e.g. v1")
    parser.add_argument(
        "--status",
        default="production",
        help="Target status: candidate | staging | production | archived",
    )

    args = parser.parse_args()
    promote_model(args.model, args.version, args.status)


if __name__ == "__main__":
    main()