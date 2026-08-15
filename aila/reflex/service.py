from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from aila.queue import ObservationQueue
from aila.reflex.config import load_dedup_config, load_ingest_filter, load_ranking_rules
from aila.reflex.ingest import IngestReducer
from aila.reflex.store import EventStore

DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_BATCH_SIZE = 500
_RETENTION_EVERY = 300  # enforce retention roughly every N cycles


def build_reducer(store_dir: Path, rules_path: Path | None) -> IngestReducer:
    store = EventStore(store_dir)
    config_path = rules_path if rules_path is not None else Path()
    rules = load_ranking_rules(config_path)
    filter_config = load_ingest_filter(config_path)
    dedup_config = load_dedup_config(config_path)
    return IngestReducer(store, rules, filter_config=filter_config, dedup_config=dedup_config)


def run_once(queue: ObservationQueue, reducer: IngestReducer, *, batch_size: int | None = DEFAULT_BATCH_SIZE) -> int:
    events = reducer.drain_queue(queue, batch_size=batch_size)
    return len(events)


def run_forever(
    queue: ObservationQueue,
    reducer: IngestReducer,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    cycles = 0
    while True:
        drained = run_once(queue, reducer, batch_size=batch_size)
        cycles += 1
        if cycles % _RETENTION_EVERY == 0:
            reducer.store.enforce_retention()
        # When a full batch was drained there is likely more backlog; keep going
        # without sleeping so we work through it quickly.
        if batch_size is not None and drained >= batch_size:
            continue
        sleep(interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    queue = ObservationQueue(args.queue_dir)
    reducer = build_reducer(args.store_dir, args.rules)
    if args.once:
        run_once(queue, reducer, batch_size=args.batch_size)
        reducer.store.enforce_retention()
        return 0
    try:
        run_forever(queue, reducer, interval_seconds=args.interval_seconds, batch_size=args.batch_size)
    except KeyboardInterrupt:
        return 0
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aila-reflex-ingest")
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--once",
        action="store_true",
        help="drain the queue once, enforce retention, then exit",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
