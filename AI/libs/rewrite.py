"""Rewrite some gate-call argument orders in generated C++.

This script ONLY rewrites these exact forms:

- qc.addU((a, b, c) qc.quantumRegister_[i][j]);
  -> qc.addU(qc.quantumRegister_[i][j], a, b, c);

- qc.addP((x) qc.quantumRegister_[i][j]);
  -> qc.addP(qc.quantumRegister_[i][j], x);

- qc.addCP((theta) q1, q2);
    -> qc.addCP(q1, q2, theta);

- qc.addCU((a, b, c, d) q1, q2);
    -> qc.addCU(q1, q2, a, b, c, d);

- qc.addRz((theta) qc.quantumRegister_[i][j]);
    -> qc.addRz(qc.quantumRegister_[i][j], theta);

- qc.addRy((theta) qc.quantumRegister_[i][j]);
    -> qc.addRy(qc.quantumRegister_[i][j], theta);

And token-level replacements:

- pi -> M_PI
- q0, q1, q2, ... -> qIdx0, qIdx1, qIdx2, ...

It intentionally does NOT touch other gates (e.g. addRz/addCX/etc).

Usage examples:
  - Always rewrite target file in place:
      python3 AI/libs/rewrite.py src/test/Shor/shor18MQT.cpp
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
    addCU_matches: int
    addRz_matches: int
    addRy_matches: int
    pi_replacements: int
    qidx_replacements: int
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
    text, cu_count = _rewrite_addCU(text)
    text, rz_count = _rewrite_addRz(text)
    text, ry_count = _rewrite_addRy(text)

    return text, RewriteStats(
        addU_matches=u_count,
        addP_matches=p_count,
        addCP_matches=cp_count,
        addCU_matches=cu_count,
        addRz_matches=rz_count,
        addRy_matches=ry_count,
        pi_replacements=0,
        qidx_replacements=0,
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


_ADD_CU_PATTERN = re.compile(
    r"qc\.addCU\(\((?P<params>[^)]*)\)\s*(?P<q1>[A-Za-z_][A-Za-z0-9_\[\]\.]+)\s*,\s*(?P<q2>[A-Za-z_][A-Za-z0-9_\[\]\.]+)\s*\);"
)


def _rewrite_addCU(text: str) -> tuple[str, int]:
    matches = list(_ADD_CU_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        params = m.group("params")
        q1 = m.group("q1").strip()
        q2 = m.group("q2").strip()
        parts = [p.strip() for p in params.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"addCU params not 4: {params!r}")
        return f"qc.addCU({q1}, {q2}, {parts[0]}, {parts[1]}, {parts[2]}, {parts[3]});"

    return _ADD_CU_PATTERN.sub(repl, text), len(matches)


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


_ADD_RY_PATTERN = re.compile(
    r"(?P<prefix>qc\.\.?)addRy\(\((?P<param>[^)]*)\)\s*(?P<qreg>qc\.quantumRegister_\[\d+\]\[\d+\])\s*\);"
)


def _rewrite_addRy(text: str) -> tuple[str, int]:
    matches = list(_ADD_RY_PATTERN.finditer(text))

    def repl(m: re.Match[str]) -> str:
        prefix = m.group("prefix")
        param = m.group("param").strip()
        qreg = m.group("qreg")
        return f"{prefix}addRy({qreg}, {param});"

    return _ADD_RY_PATTERN.sub(repl, text), len(matches)


_PI_PATTERN = re.compile(r"\bpi\b")
_QIDX_PATTERN = re.compile(r"\bq(?P<idx>\d+)\b")


def _rewrite_tokens(text: str) -> tuple[str, int, int]:
    text, pi_count = _PI_PATTERN.subn("M_PI", text)
    text, qidx_count = _QIDX_PATTERN.subn(r"qIdx\g<idx>", text)
    return text, pi_count, qidx_count


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite selected qc gate argument orders and tokens in a C++ file (always in-place).",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Target .cpp/.h file path",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if not args.path.exists():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    original = args.path.read_text(encoding="utf-8")
    preprocessed, pi_count, qidx_count = _rewrite_tokens(original)
    rewritten, base_stats = rewrite_text(preprocessed)
    stats = RewriteStats(
        addU_matches=base_stats.addU_matches,
        addP_matches=base_stats.addP_matches,
        addCP_matches=base_stats.addCP_matches,
        addCU_matches=base_stats.addCU_matches,
        addRz_matches=base_stats.addRz_matches,
        addRy_matches=base_stats.addRy_matches,
        pi_replacements=pi_count,
        qidx_replacements=qidx_count,
        changed=(rewritten != original),
    )

    print(
        f"addU matches: {stats.addU_matches}, addP matches: {stats.addP_matches}, addCP matches: {stats.addCP_matches}, addCU matches: {stats.addCU_matches}, addRz matches: {stats.addRz_matches}, addRy matches: {stats.addRy_matches}, pi replacements: {stats.pi_replacements}, qIdx replacements: {stats.qidx_replacements}, changed: {stats.changed}",
        file=sys.stderr,
    )

    # always in-place rewrite
    if stats.changed:
        args.path.write_text(rewritten, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
