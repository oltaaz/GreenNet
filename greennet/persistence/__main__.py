from __future__ import annotations

import argparse
import csv
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

    export_parser = subparsers.add_parser("export-summary", help="Export aggregated run summary rows from SQLite.")
    export_parser.add_argument("--db-path", type=Path, default=None)
    export_parser.add_argument("--base", choices=["results", "runs", "both"], default="both")
    export_parser.add_argument("--tag", type=str, default=None)
    export_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    db_path = args.db_path or default_db_path()

    if args.command == "init":
        repository = get_run_repository(db_path)
        repository.ensure_initialized()
        print(f"[persistence] SQLite run store ready at {db_path}")
        return

    if args.command == "export-summary":
        repository = get_run_repository(db_path)
        rows = repository.list_summary_rows(base=args.base, tag=args.tag)
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(str(key))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        print(f"[persistence] exported {len(rows)} rows to {args.output} from {db_path}")
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
