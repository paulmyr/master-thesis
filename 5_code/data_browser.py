#!/usr/bin/env python
"""
Interactive TUI data browser for experimental voltage/current traces.

Curses-based file tree with vim-style navigation. Hovering over a .dat
file automatically displays voltage and current in a matplotlib window.

Usage:
    python data_browser.py [data_dir]

Keys:
    j / Down        Move cursor down
    k / Up          Move cursor up
    l / Right / CR  Expand directory or select file
    h / Left        Collapse directory or go to parent
    g               Jump to top
    G               Jump to bottom
    Ctrl+D / Ctrl+U Half-page down / up
    PgDn / PgUp     Full page down / up
    J               Toggle junction potential correction (+/-35 mV)
    q / Esc         Quit
"""
from __future__ import annotations

import curses
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ADoptEX.core.data import detect_spikes, find_stim_window

# ── Configuration ────────────────────────────────────────────────────

JUNCTION_POTENTIAL_MV = 35.0
SPIKE_THRESHOLD_MV = -20.0
STIM_THRESHOLD_PA = 100.0
DEBOUNCE_S = 0.1

# Channel pairing: voltage tag -> current tag
_V_TO_I = {"_ch5_": "_ch4_", "_ch3_": "_ch2_"}
_I_TO_V = {v: k for k, v in _V_TO_I.items()}


# ── Data helpers ─────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _load_dat(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column .dat file (# comments skipped). Cached."""
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def _find_pair(path: Path) -> tuple[Path | None, Path | None]:
    """Return (voltage_path, current_path) for a .dat file."""
    name = path.name
    for vt, it in _V_TO_I.items():
        if vt in name:
            p = path.parent / name.replace(vt, it)
            return path, (p if p.exists() else None)
    for it, vt in _I_TO_V.items():
        if it in name:
            p = path.parent / name.replace(it, vt)
            return (p if p.exists() else None), path
    return path, None


def _is_current_file(path: Path) -> bool:
    return any(t in path.name for t in _I_TO_V)


# ── Tree model ───────────────────────────────────────────────────────


@dataclass
class Node:
    path: Path
    is_dir: bool
    depth: int = 0
    expanded: bool = False
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    _loaded: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_dat(self) -> bool:
        return not self.is_dir and self.path.suffix == ".dat"


def _ensure_children(node: Node) -> None:
    """Lazily load directory children (dirs + .dat files only)."""
    if node._loaded or not node.is_dir:
        return
    try:
        entries = sorted(
            node.path.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError:
        entries = []
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir() or e.suffix == ".dat":
            node.children.append(
                Node(path=e, is_dir=e.is_dir(), depth=node.depth + 1, parent=node)
            )
    node._loaded = True


def _flatten(node: Node, skip_self: bool = False) -> list[Node]:
    """Flatten expanded tree into a visible node list."""
    out: list[Node] = [] if skip_self else [node]
    if node.is_dir and node.expanded:
        _ensure_children(node)
        for c in node.children:
            out.extend(_flatten(c))
    return out


# ── Plot manager ─────────────────────────────────────────────────────


class _Plotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 7), num="Data Browser")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        self.ax_v = self.fig.add_subplot(gs[0])
        self.ax_i = self.fig.add_subplot(gs[1], sharex=self.ax_v)
        self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97)
        self._key: object = None
        self._info = ""

    def update(
        self, v_path: Path | None, i_path: Path | None, jp: bool
    ) -> str:
        """Redraw traces. Returns a short info string for the status bar."""
        key = (str(v_path), str(i_path), jp)
        if key == self._key:
            return self._info
        self._key = key

        self.ax_v.cla()
        self.ax_i.cla()
        titles: list[str] = []
        info_parts: list[str] = []
        stim_s = stim_e = None

        # Current trace
        if i_path and i_path.exists():
            try:
                ti, ir = _load_dat(str(i_path))
                self.ax_i.plot(ti, ir, color="tab:blue", lw=0.6)
                titles.append(i_path.name)
                si, ei = find_stim_window(ir, STIM_THRESHOLD_PA)
                if si < ei:
                    stim_s = ti[si]
                    stim_e = ti[min(ei, len(ti) - 1)]
                    info_parts.append(f"I={ir[si:ei].mean():.0f}pA")
            except Exception:
                pass

        # Voltage trace
        if v_path and v_path.exists():
            try:
                tv, vr = _load_dat(str(v_path))
                v = vr - JUNCTION_POTENTIAL_MV if jp else vr
                self.ax_v.plot(tv, v, color="k", lw=0.5)
                titles.insert(0, v_path.name)
                dt = float(np.mean(np.diff(tv)))
                spikes = detect_spikes(v, dt, SPIKE_THRESHOLD_MV)
                if len(spikes):
                    self.ax_v.plot(
                        spikes, np.interp(spikes, tv, v), "r|", ms=10, mew=1.5
                    )
                info_parts.insert(0, f"{len(spikes)} spikes")
                info_parts.insert(0, f"{tv[-1] - tv[0]:.0f}ms")
                if stim_s is not None and stim_e is not None:
                    for ax in (self.ax_v, self.ax_i):
                        ax.axvspan(stim_s, stim_e, alpha=0.06, color="green")
            except Exception:
                pass

        self.ax_v.set_ylabel("Voltage (mV)")
        self.ax_i.set_ylabel("Current (pA)")
        self.ax_i.set_xlabel("Time (ms)")
        for ax in (self.ax_v, self.ax_i):
            ax.grid(True, alpha=0.15)
        plt.setp(self.ax_v.get_xticklabels(), visible=False)
        jp_tag = " [JP corrected]" if jp else ""
        self.fig.suptitle(
            (" + ".join(titles) + jp_tag) if titles else "No data", fontsize=10
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._info = " | ".join(info_parts) if info_parts else ""
        return self._info

    def flush(self):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self):
        plt.close(self.fig)


# ── Browser TUI ──────────────────────────────────────────────────────


class Browser:
    def __init__(self, scr: curses.window, root_path: Path):
        self.scr = scr
        self.root = Node(path=root_path, is_dir=True, depth=0, expanded=True)
        _ensure_children(self.root)
        self.visible: list[Node] = _flatten(self.root, skip_self=True)
        self.cursor = 0
        self.scroll = 0
        self.jp = False
        self._trace_info = ""
        self._pending_plot = False
        self._last_move = 0.0

        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLUE, -1)  # directories
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # voltage .dat
        curses.init_pair(3, curses.COLOR_CYAN, -1)  # current .dat
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)  # cursor
        curses.init_pair(5, curses.COLOR_YELLOW, -1)  # status

        self.plot = _Plotter()

    # ── helpers ───────────────────────────────────────────────────

    def _refresh(self):
        self.visible = _flatten(self.root, skip_self=True)
        self.cursor = min(self.cursor, max(0, len(self.visible) - 1))

    def _nd(self) -> Node | None:
        if 0 <= self.cursor < len(self.visible):
            return self.visible[self.cursor]
        return None

    def _do_plot(self):
        nd = self._nd()
        if nd and nd.is_dat:
            vp, ip = _find_pair(nd.path)
            self._trace_info = self.plot.update(vp, ip, self.jp)
        self._pending_plot = False

    # ── navigation ───────────────────────────────────────────────

    def _move(self, delta: int):
        old = self.cursor
        self.cursor = max(0, min(len(self.visible) - 1, self.cursor + delta))
        if self.cursor != old:
            nd = self._nd()
            if nd and nd.is_dat:
                self._pending_plot = True
                self._last_move = time.monotonic()

    def _expand(self):
        nd = self._nd()
        if nd is None:
            return
        if nd.is_dir:
            if not nd.expanded:
                nd.expanded = True
                _ensure_children(nd)
                self._refresh()
            elif nd.children:
                self.cursor += 1
                nd2 = self._nd()
                if nd2 and nd2.is_dat:
                    self._pending_plot = True
                    self._last_move = time.monotonic()
        elif nd.is_dat:
            self._do_plot()

    def _collapse(self):
        nd = self._nd()
        if nd is None:
            return
        if nd.is_dir and nd.expanded:
            nd.expanded = False
            self._refresh()
        elif nd.parent and nd.parent is not self.root:
            for i, n in enumerate(self.visible):
                if n is nd.parent:
                    self.cursor = i
                    break

    # ── drawing ──────────────────────────────────────────────────

    def _draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        if h < 5 or w < 20:
            return
        hdr, ftr = 2, 3
        tree_h = h - hdr - ftr

        # Header
        title = f" Data Browser \u2014 {self.root.path}"
        self.scr.attron(curses.A_BOLD)
        self.scr.addnstr(0, 0, title[: w - 1], w - 1)
        self.scr.attroff(curses.A_BOLD)
        self.scr.addnstr(1, 0, "\u2500" * (w - 1), w - 1)

        # Scroll adjustment
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + tree_h:
            self.scroll = self.cursor - tree_h + 1

        # Tree
        if not self.visible:
            self.scr.addnstr(hdr, 2, "(empty \u2014 no .dat files)", w - 3, curses.A_DIM)

        for i in range(min(tree_h, len(self.visible) - self.scroll)):
            idx = self.scroll + i
            row = hdr + i
            nd = self.visible[idx]
            indent = "  " * nd.depth

            if nd.is_dir:
                icon = "\u25bc " if nd.expanded else "\u25b6 "
                clr = curses.color_pair(1)
            elif nd.is_dat and _is_current_file(nd.path):
                icon = "  "
                clr = curses.color_pair(3)
            else:
                icon = "  "
                clr = curses.color_pair(2)

            line = f"{indent}{icon}{nd.name}"
            if len(line) > w - 1:
                line = line[: w - 4] + "\u2026"

            if idx == self.cursor:
                attr = curses.color_pair(4) | curses.A_BOLD
                self.scr.addnstr(row, 0, " " * (w - 1), w - 1, attr)
                self.scr.addnstr(row, 0, line, w - 1, attr)
            else:
                self.scr.addnstr(row, 0, line, w - 1, clr)

        # Footer
        sep = h - ftr
        self.scr.addnstr(sep, 0, "\u2500" * (w - 1), w - 1)

        nd = self._nd()
        if nd:
            path_str = str(nd.path)
            extra = f"  ({self._trace_info})" if self._trace_info and nd.is_dat else ""
            info = path_str + extra
            if len(info) > w - 1:
                info = "\u2026" + info[-(w - 3) :]
            self.scr.addnstr(h - 2, 0, info[: w - 1], w - 1, curses.color_pair(5))

        jp_s = "ON" if self.jp else "OFF"
        hlp = f" \u2191k \u2193j \u2192l/\u23ce \u2190h  g/G Top/End  J JP:{jp_s}  q Quit"
        self.scr.addnstr(h - 1, 0, hlp[: w - 1], w - 1, curses.A_DIM)
        self.scr.refresh()

    # ── main loop ────────────────────────────────────────────────

    def run(self):
        self.scr.timeout(50)
        while True:
            self._draw()
            self.plot.flush()

            # Debounced auto-plot
            if self._pending_plot and (time.monotonic() - self._last_move) >= DEBOUNCE_S:
                self._do_plot()

            key = self.scr.getch()
            if key == -1:
                continue

            h, _ = self.scr.getmaxyx()
            half = max(1, (h - 5) // 2)

            if key in (ord("q"), 27):  # q / Esc
                break
            elif key in (ord("j"), curses.KEY_DOWN):
                self._move(1)
            elif key in (ord("k"), curses.KEY_UP):
                self._move(-1)
            elif key in (ord("l"), curses.KEY_RIGHT, 10, 13):  # l / Right / Enter
                self._expand()
            elif key in (ord("h"), curses.KEY_LEFT):
                self._collapse()
            elif key == ord("g"):
                self._move(-len(self.visible))
            elif key == ord("G"):
                self._move(len(self.visible))
            elif key == 4:  # Ctrl+D
                self._move(half)
            elif key == 21:  # Ctrl+U
                self._move(-half)
            elif key == curses.KEY_NPAGE:
                self._move(half * 2)
            elif key == curses.KEY_PPAGE:
                self._move(-half * 2)
            elif key == ord("J"):
                self.jp = not self.jp
                self.plot._key = None  # force redraw
                self._do_plot()
            elif key == curses.KEY_RESIZE:
                self.scr.clear()

        self.plot.close()


# ── Entry point ──────────────────────────────────────────────────────


def main(data_dir: str | None = None):
    if data_dir is None:
        default = (
            Path(__file__).resolve().parent.parent
            / "4_data"
            / "models"
            / "optimisations"
        )
        if default.is_dir():
            data_dir = str(default)
        else:
            print(f"Default data directory not found: {default}")
            print("Usage: python data_browser.py [data_dir]")
            sys.exit(1)

    path = Path(data_dir)
    if not path.is_dir():
        print(f"Not a directory: {path}")
        sys.exit(1)

    curses.wrapper(lambda scr: Browser(scr, path).run())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
