from __future__ import annotations

import argparse
from pathlib import Path

from .sqlite_store import backfill_run_directories, default_db_path, get_run_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="GreenNet run persistence helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create or migrate the SQLite run store.")
    init_parser.add_argument("--db-path", type=Path, default=None)

    backfill_parser = subparsers.add_parser("backfill", help="Import compatible run artifacts into SQLite.")
    backfill_parser.add_argument("--db-path", type=Path, default=None)
    backfill_parser.add_argument("--base", choices=["results", "runs", "both"], default="both")
    backfill_parser.add_argument("--results-dir", type=Path, default=None)
    backfill_parser.add_argument("--runs-dir", type=Path, default=None)

    args = parser.parse_args()
    db_path = args.db_path or default_db_path()

    if args.command == "init":
        repository = get_run_repository(db_path)
        repository.ensure_initialized()
        print(f"[persistence] SQLite run store ready at {db_path}")
        return

    report = backfill_run_directories(
        base=args.base,
        db_path=db_path,
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
    )
    print(
        "[persistence] backfill complete "
        f"(db={db_path}, scanned={report.scanned}, persisted={report.persisted}, "
        f"skipped={report.skipped}, failed={report.failed})"
    )


if __name__ == "__main__":
    main()
