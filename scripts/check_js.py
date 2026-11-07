"""Crude sanity checks for web/app.js without Node: bracket balance (ignoring
strings, template literals and comments) and element ids referenced from JS that
do not exist in index.html."""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")


def bracket_check(src: str) -> int:
    i, n, line = 0, len(src), 1
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    errors = 0
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
        if c == "/" and src[i + 1:i + 2] == "/":
            j = src.find("\n", i)
            i = j if j >= 0 else n
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            j = src.find("*/", i)
            line += src[i:j].count("\n")
            i = j + 2
            continue
        if c in ('"', "'"):
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    print(f"unterminated string at line {line}")
                    errors += 1
                    break
                j += 1
            i = j + 1
            continue
        if c == "`":
            j = i + 1
            depth = 0
            while j < n:
                ch = src[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "$" and src[j + 1:j + 2] == "{":
                    depth += 1
                    j += 2
                    continue
                if ch == "}" and depth:
                    depth -= 1
                elif ch == "`" and depth == 0:
                    break
                elif ch == "\n":
                    line += 1
                j += 1
            i = j + 1
            continue
        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != pairs[c]:
                print(f"bracket mismatch '{c}' at line {line}; open stack top: {stack[-1] if stack else None}")
                return errors + 1
            stack.pop()
        i += 1
    if stack:
        print("unclosed brackets:", stack[:5])
        errors += 1
    return errors


def id_check(js: str, html: str) -> int:
    ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    used = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)', js)) | set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"', js))
    missing = sorted(used - ids)
    if missing:
        print("ids used in JS but missing from HTML:", missing)
    return len(missing)


def main() -> int:
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8").read()
    errors = bracket_check(js) + id_check(js, html)
    print("check_js:", "OK" if not errors else f"{errors} problem(s)", f"({js.count(chr(10))} lines)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
