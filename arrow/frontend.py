"""Frontend — parse arrow spec files into (func_name, examples).

Errors follow the compiler convention:
    arrow: error: <file>:<line>: <message>
"""
from __future__ import annotations


class ParseError(Exception):
    def __init__(self, file: str, line: int, msg: str):
        self.file = file
        self.line = line
        self.msg = msg
        super().__init__(f"arrow: error: {file}:{line}: {msg}")


def parse_spec(text: str, filename: str = "<stdin>"):
    """Parse spec text. Returns (name, examples) where examples = [(x, y), ...].

    Accepted lines:
        int f(int x)          — signature (optional; the name must match)
        f(0) = 1              — example
        # comment / blank     — ignored
    """
    name = None
    sig = None
    examples: list[tuple[int, int]] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # signature line:  int f(int x)
        if line.startswith("int "):
            rest = line[4:].strip()
            if "(" not in rest or not rest.endswith(")"):
                raise ParseError(filename, lineno,
                                 f"malformed signature: {raw!r}")
            fn, _args = rest.split("(", 1)
            fn = fn.strip()
            if not fn.isidentifier():
                raise ParseError(filename, lineno, f"bad function name: {fn!r}")
            if name is not None and fn != name:
                raise ParseError(filename, lineno,
                                 f"function {fn!r} does not match {name!r}")
            name = fn
            sig = line
            continue

        # example line:  f(0) = 1
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            try:
                y = int(rhs)
            except ValueError:
                raise ParseError(filename, lineno,
                                 f"expected integer output, got {rhs!r}")
            if not (lhs.endswith(")") and "(" in lhs):
                raise ParseError(filename, lineno,
                                 f"malformed example: {raw!r}")
            fn, args = lhs[:-1].split("(", 1)
            fn = fn.strip()
            try:
                x = int(args.strip())
            except ValueError:
                raise ParseError(filename, lineno,
                                 f"expected integer input, got {args.strip()!r}")
            if name is not None and fn != name:
                raise ParseError(filename, lineno,
                                 f"function {fn!r} does not match {name!r}")
            if name is None:
                name = fn
            if not fn.isidentifier():
                raise ParseError(filename, lineno, f"bad function name: {fn!r}")
            examples.append((x, y))
            continue

        raise ParseError(filename, lineno, f"unrecognized line: {raw!r}")

    if name is None:
        raise ParseError(filename, 1, "no function defined in spec")
    if not examples:
        raise ParseError(filename, 1, "no examples given — nothing to synthesize")
    return name, examples
