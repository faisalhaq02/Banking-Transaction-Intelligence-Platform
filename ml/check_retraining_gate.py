from __future__ import annotations

import json
import sys
from pathlib import Path


DECISION_FILE = Path("/opt/project/outputs/retraining_decision.json")


def main() -> None:
    if not DECISION_FILE.exists():
        raise FileNotFoundError(f"Missing decision file: {DECISION_FILE}")

    with open(DECISION_FILE, "r") as f:
        decision = json.load(f)

    should_retrain = bool(decision.get("should_retrain", False))

    print("===== Retraining Gate Check =====")
    print(json.dumps(decision, indent=2))

    if should_retrain:
        print("Retraining is required. Gate PASSED.")
        sys.exit(0)

    print("Retraining is not required. Gate BLOCKED.")
    sys.exit(1)


if __name__ == "__main__":
    main()