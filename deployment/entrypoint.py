from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="source-inversion",
        description="Source inversion service and isolated worker launcher",
    )
    parser.add_argument("command", choices=("serve", "worker"))
    return parser


def _serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="source-inversion serve")
    parser.add_argument(
        "--host",
        default=os.environ.get("SOURCE_INVERSION_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SOURCE_INVERSION_PORT", "8000")),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("SOURCE_INVERSION_LOG_LEVEL", "info"),
    )
    return parser


def _run_server(argv: Sequence[str]) -> None:
    args = _serve_parser().parse_args(argv)
    import uvicorn

    from .api import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    command_args = _command_parser().parse_args(arguments[:1])
    remaining = arguments[1:]
    if command_args.command == "serve":
        _run_server(remaining)
        return

    from .worker_entry import main as worker_main

    worker_main(remaining)


if __name__ == "__main__":
    main()
