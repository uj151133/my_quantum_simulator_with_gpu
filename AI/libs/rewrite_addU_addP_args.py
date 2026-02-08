"""Rewrite some gate-call argument orders in generated C++.

This script ONLY rewrites these exact forms:

- qc.addU((a, b, c) qc.quantumRegister_[i][j]);
  -> qc.addU(qc.quantumRegister_[i][j], a, b, c);

- qc.addP((x) qc.quantumRegister_[i][j]);
  -> qc.addP(qc.quantumRegister_[i][j], x);

- qc.addCP((theta) q1, q2);
    -> qc.addCP(q1, q2, theta);

- qc.addRz((theta) qc.quantumRegister_[i][j]);
    -> qc.addRz(qc.quantumRegister_[i][j], theta);

It intentionally does NOT touch other gates (e.g. addRz/addCX/etc).

Usage examples:
  - Dry-run (prints rewritten file to stdout):
      python3 AI/libs/rewrite_addU_addP_args.py src/test/Shor/shor18MQT.cpp

  - In-place rewrite:
      python3 AI/libs/rewrite_addU_addP_args.py --in-place src/test/Shor/shor18MQT.cpp

  - Check-only (non-zero exit if rewrites are needed):
      python3 AI/libs/rewrite_addU_addP_args.py --check src/test/Shor/shor18MQT.cpp
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RewriteStats:
    addU_matches: int
    addP_matches: int
    addCP_matches: int
    addRz_matches: int
    changed: bool


_ADD_U_PATTERN = re.compile(
    r"qc\.addU\(\((?P<params>[^)]*)\)\s*(?P<qreg>qc\.quantumRegister_\[\d+\]\[\d+\])\s*\);"
)


def _rewrite_addU(text: str) -> tuple[str, int]:
    matches = list(_ADD_U_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        params = m.group("params")
        qreg = m.group("qreg")

        parts = [p.strip() for p in params.split(",")]
        if len(parts) != 3:
            raise ValueError(f"addU params not 3: {params!r}")

        a, b, c = parts
        return f"qc.addU({qreg}, {a}, {b}, {c});"

    return _ADD_U_PATTERN.sub(repl, text), len(matches)


_ADD_P_PATTERN = re.compile(
    r"qc\.addP\(\((?P<param>[^)]*)\)\s*(?P<qreg>qc\.quantumRegister_\[\d+\]\[\d+\])\s*\);"
)


def _rewrite_addP(text: str) -> tuple[str, int]:
    matches = list(_ADD_P_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        param = m.group("param").strip()
        qreg = m.group("qreg")
        return f"qc.addP({qreg}, {param});"

    return _ADD_P_PATTERN.sub(repl, text), len(matches)


def rewrite_text(text: str) -> tuple[str, RewriteStats]:
    original = text

    text, u_count = _rewrite_addU(text)
    text, p_count = _rewrite_addP(text)
    text, cp_count = _rewrite_addCP(text)
    text, rz_count = _rewrite_addRz(text)

    return text, RewriteStats(
        addU_matches=u_count,
        addP_matches=p_count,
        addCP_matches=cp_count,
        addRz_matches=rz_count,
        changed=(text != original),
    )


_ADD_CP_PATTERN = re.compile(
    r"qc\.addCP\(\((?P<param>[^)]*)\)\s*(?P<q1>[A-Za-z_][A-Za-z0-9_\[\]\.]+)\s*,\s*(?P<q2>[A-Za-z_][A-Za-z0-9_\[\]\.]+)\s*\);"
)


def _rewrite_addCP(text: str) -> tuple[str, int]:
    matches = list(_ADD_CP_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        param = m.group("param").strip()
        q1 = m.group("q1").strip()
        q2 = m.group("q2").strip()
        return f"qc.addCP({q1}, {q2}, {param});"

    return _ADD_CP_PATTERN.sub(repl, text), len(matches)


_ADD_RZ_PATTERN = re.compile(
    r"qc\.addRz\(\((?P<param>[^)]*)\)\s*(?P<qreg>qc\.quantumRegister_\[\d+\]\[\d+\])\s*\);"
)


def _rewrite_addRz(text: str) -> tuple[str, int]:
    matches = list(_ADD_RZ_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        param = m.group("param").strip()
        qreg = m.group("qreg")
        return f"qc.addRz({qreg}, {param});"

    return _ADD_RZ_PATTERN.sub(repl, text), len(matches)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite qc.addU/qc.addP argument order in a C++ file.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Target .cpp/.h file path",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the file in place (default is dry-run to stdout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check-only: exit 1 if rewrites are needed; does not write",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if args.in_place and args.check:
        print("error: --in-place and --check are mutually exclusive", file=sys.stderr)
        return 2

    if not args.path.exists():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    original = args.path.read_text(encoding="utf-8")
    rewritten, stats = rewrite_text(original)

    print(
        f"addU matches: {stats.addU_matches}, addP matches: {stats.addP_matches}, addCP matches: {stats.addCP_matches}, addRz matches: {stats.addRz_matches}, changed: {stats.changed}",
        file=sys.stderr,
    )

    if args.check:
        return 1 if stats.changed else 0

    if args.in_place:
        if stats.changed:
            args.path.write_text(rewritten, encoding="utf-8")
        return 0

    # dry-run: print file to stdout
    sys.stdout.write(rewritten)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
