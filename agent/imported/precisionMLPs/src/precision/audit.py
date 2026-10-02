"""Static no-leakage audit of the p-bit emulator build.

Clang parses pbit_emul.c (which includes pbit_algo.h and lapack_gelss.h, macros expanded) and this
walks the syntax tree of every function defined in those files. Outside the rounding layer (the
functions that implement correctly rounded +, -, *, /, sqrt), it reports:

- floating-point +, -, *, / (and compound assignments),
- calls to functions that are neither defined in these files nor exact operations
  (fabs, copysign, signbit, nearbyint, ldexp),
- conversions integer -> floating and between floating types,
- floating literals not passed through FROMD (which checks they are format values).

Comparisons, unary minus, fabs, copysign, signbit, nearbyint, and ldexp (always followed by the
format rounding in LDEXP) are exact and allowed. The result lists every finding; the test
suite requires exactly the documented exceptions and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCES = {"pbit_emul.c", "pbit_algo.h", "lapack_gelss.h"}
ROUNDING_LAYER = {"watch", "rnd", "sgn", "e_add", "e_sub", "e_mul", "e_div", "e_sqrt", "e_fromd", "e_i2t",
                  "e_setfmt", "pbit_round", "pbit_op", "pbit_counters", "pbit_first_bad",
                  "pbit_first_bad_stage"}
EXACT_CALLS = {"fabs", "copysign", "nearbyint", "ldexp", "malloc", "free", "memcpy"}
# The one allowed exception: DGELSS's crossover MNTHR = ILAENV(6) = INT(REAL(MIN(M,N))*1.6E0), an integer
# computed in single precision exactly as reference LAPACK does; it only selects whether to QR-factor first.
ALLOWED = [{"function": "lp_gelss", "what": "floating '*'", "type": "float",
            "operands": ["ConditionalOperator", "FloatingLiteral:1.60000002"]},
           {"function": "lp_gelss", "what": "cast IntegralToFloating to float", "operands": ["ConditionalOperator"]},
           {"function": "lp_gelss", "what": "literal 1.60000002"}]
FLOAT_TYPES = {"double", "float", "long double", "_Float16", "__bf16"}
ARITH = {"+", "-", "*", "/"}


def _is_float(t: dict | None) -> bool:
    if not t:
        return False
    return t.get("desugaredQualType", t.get("qualType")) in FLOAT_TYPES or t.get("qualType") in FLOAT_TYPES


def _dump(src: Path) -> dict:
    out = subprocess.run(["clang", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-std=c11", "-I", str(src),
                          str(src / "pbit_emul.c")], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _callee(call: dict) -> str | None:
    stack = list(call.get("inner", [])[:1])
    while stack:
        n = stack.pop()
        if n.get("kind") == "DeclRefExpr":
            return n.get("referencedDecl", {}).get("name")
        stack.extend(n.get("inner", []))
    return None


def audit(src: Path = HERE) -> list[dict]:
    """Findings for the emulator build of the sources in ``src`` (default: this package)."""
    ast = _dump(Path(src))
    findings = []
    current = {"file": None}
    ours: set[str] = set()

    def track(node):
        for key in ("loc", "range"):
            locs = [node.get("loc", {})] if key == "loc" else [node.get("range", {}).get("begin", {}),
                                                                node.get("range", {}).get("end", {})]
            for loc in locs:
                for sub in (loc, loc.get("spellingLoc", {}), loc.get("expansionLoc", {})):
                    if "file" in sub:
                        current["file"] = Path(sub["file"]).name

    functions = []
    for decl in ast.get("inner", []):
        track(decl)
        stack = [decl]
        while stack:  # keep the file tracker in document order
            n = stack.pop()
            if n is not decl:
                track(n)
            stack.extend(reversed(n.get("inner", [])))
        if decl.get("kind") == "FunctionDecl" and any(c.get("kind") == "CompoundStmt" for c in decl.get("inner", [])):
            fdecl_file = current["file"]
            functions.append((decl, fdecl_file))
    # a definition's file is where its body ends; ours are the functions defined in SOURCES
    functions = [(d, f) for d, f in functions if f in SOURCES]
    ours.update(d["name"] for d, _ in functions)

    def walk(node, fn, parent_call=None):
        kind = node.get("kind")
        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            op = node.get("opcode", "").rstrip("=") if kind == "CompoundAssignOperator" else node.get("opcode")
            ftype = node.get("computationResultType") if kind == "CompoundAssignOperator" else node.get("type")
            if op in ARITH and _is_float(ftype):
                findings.append({"function": fn, "what": f"floating '{node['opcode']}'",
                                 "type": ftype.get("qualType"), "operands": [_describe(c) for c in node.get("inner", [])]})
        elif kind == "CallExpr":
            name = _callee(node)
            if name not in ours and name not in EXACT_CALLS:
                findings.append({"function": fn, "what": f"call {name}"})
            parent_call = name
        elif kind in ("ImplicitCastExpr", "CStyleCastExpr"):
            ck = node.get("castKind")
            if ck in ("IntegralToFloating", "FloatingCast"):
                findings.append({"function": fn, "what": f"cast {ck} to {node['type'].get('qualType')}",
                                 "operands": [_describe(c) for c in node.get("inner", [])]})
        elif kind == "FloatingLiteral" and parent_call != "e_fromd":
            findings.append({"function": fn, "what": f"literal {node.get('value')}"})
        for c in node.get("inner", []):
            walk(c, fn, parent_call if kind != "CallExpr" else parent_call)

    for decl, _ in functions:
        name = decl["name"]
        if name in ROUNDING_LAYER or name.startswith("__"):
            continue
        body = [c for c in decl.get("inner", []) if c.get("kind") == "CompoundStmt"]
        for b in body:
            walk(b, name)
    return findings


def _describe(node: dict) -> str:
    kind = node.get("kind")
    if kind in ("FloatingLiteral", "IntegerLiteral"):
        return f"{kind}:{node.get('value')}"
    for c in node.get("inner", []):
        return _describe(c) if kind in ("ImplicitCastExpr", "ParenExpr", "CStyleCastExpr") else kind
    return kind or "?"


if __name__ == "__main__":
    for f in audit():
        print(json.dumps(f))
