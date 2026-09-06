#!/usr/bin/env python3
"""Refuse to regenerate the superseded noncanonical SVHN ODT figure."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "withheld: the legacy SVHN result used a noncanonical downstream-Gram "
        "surrogate. Use only a prospectively verified canonical ODT result"
    )


if __name__ == "__main__":
    main()
