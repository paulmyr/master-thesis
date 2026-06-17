"""Reverse the simulate-injection migration: restore original loss signatures.

make_X_loss_fn(make_simulator(C, D, DT, T, init_state_fn=...), <rest>)
  -> make_X_loss_fn(cell=C, data_stimuli=D, t_max=T, dt_ms=DT, <rest minus dt_ms kwarg>)

Also removes the imports added by the forward migration. Run DRY=1 to preview.
"""
import ast
import glob
import json
import os
import re

DRY = os.environ.get("DRY", "1") == "1"
CALL_RE = re.compile(r"make_(mse|guarino|van_rossum)_loss_fn\s*\(")


def _match_close(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _line_is_comment(text, idx):
    bol = text.rfind("\n", 0, idx) + 1
    return text[bol:idx].lstrip().startswith("#")


def transform_calls(text):
    spans = []
    for m in CALL_RE.finditer(text):
        if _line_is_comment(text, m.start()):
            continue
        open_idx = m.end() - 1
        close = _match_close(text, open_idx)
        if close == -1:
            continue
        spans.append((m.start(), open_idx, close, m.group(1)))
    n = 0
    for name_start, open_idx, close, kind in sorted(spans, reverse=True):
        args_src = text[open_idx + 1 : close]
        try:
            call = ast.parse("f(" + args_src + ")", mode="eval").body
        except SyntaxError:
            continue
        if not call.args:
            continue
        sim = call.args[0]
        # first positional must be make_simulator(C, D, DT, T, ...)
        if not (isinstance(sim, ast.Call) and getattr(sim.func, "id", "") == "make_simulator"):
            continue
        if len(sim.args) < 4:
            continue
        C, D, DT, T = (ast.unparse(sim.args[i]) for i in range(4))
        rest = []
        for a in call.args[1:]:
            rest.append(ast.unparse(a))
        for k in call.keywords:
            if k.arg == "dt_ms":  # re-added from the simulator below
                continue
            if k.arg is None:
                rest.append("**" + ast.unparse(k.value))
            else:
                rest.append(f"{k.arg}={ast.unparse(k.value)}")
        head = f"make_{kind}_loss_fn(cell={C}, data_stimuli={D}, t_max={T}, dt_ms={DT}"
        new_call = head + (", " + ", ".join(rest) if rest else "") + ")"
        text = text[:name_start] + new_call + text[close + 1 :]
        n += 1
    return text, n


def strip_imports(text):
    out = []
    for ln in text.splitlines(keepends=True):
        s = ln.strip()
        if s == "from ADoptEX.core.init_state import make_steady_state_init":
            continue
        if s == "from ADoptEX.core.simulation import make_simulator":
            continue
        # merged simulation import: drop make_simulator from the name list
        m = re.match(r"^(\s*from ADoptEX\.core\.simulation import )(.+?)(\s*)$", ln)
        if m:
            names = [x.strip() for x in m.group(2).split(",") if x.strip() != "make_simulator"]
            if names:
                ln = f"{m.group(1)}{', '.join(names)}\n"
            else:
                continue
        out.append(ln)
    return "".join(out)


def process(text):
    new, n = transform_calls(text)
    if n:
        new = strip_imports(new)
    return new, n


def main():
    print("DRY-RUN" if DRY else "APPLYING", "\n")
    for p in sorted(glob.glob("notebooks/*.py")):
        if os.path.basename(p).startswith("_"):
            continue
        src = open(p).read()
        new, n = process(src)
        if n:
            print(f"[py ] {os.path.basename(p)}: {n} call(s) reverted")
            if not DRY:
                open(p, "w").write(new)
    for p in sorted(glob.glob("notebooks/*.ipynb")):
        if ".ipynb_checkpoints" in p:
            continue
        nb = json.load(open(p))
        total = 0
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            if not CALL_RE.search(src):
                continue
            new, n = process(src)
            total += n
            if n:
                c["source"] = new.splitlines(keepends=True)
        if total:
            print(f"[nb ] {os.path.basename(p)}: {total} call(s) reverted")
            if not DRY:
                with open(p, "w") as f:
                    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
