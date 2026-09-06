"""Root entrypoint for all NER experiments."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from ner_core.config import BootstrapSettings  # noqa: E402
from ner_core.runner import dispatch  # noqa: E402

def parse_args() -> argparse.Namespace:
    """Parse common runner options."""

    root_dir = ROOT_DIR
    bootstrap = BootstrapSettings(_env_file=root_dir / ".env")
    parser = argparse.ArgumentParser(description="Run a reproducible NER experiment stage.")
    parser.add_argument("--exp", default=bootstrap.exp_name or "exp_baseline")
    parser.add_argument(
        "--stage",
        choices=("train", "predict", "evaluate", "serve"),
        default="serve",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested YAML value; can be specified more than once.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute the selected experiment stage."""

    args = parse_args()
    root_dir = ROOT_DIR
    try:
        dispatch(
            root_dir=root_dir,
            experiment_name=args.exp,
            stage=args.stage,
            config_path=args.config,
            overrides=args.overrides,
            run_id=args.run_id,
            run_dir=args.run_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError, ModuleNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
