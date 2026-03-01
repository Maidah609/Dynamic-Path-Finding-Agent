"""
Dynamic Pathfinding Agent
AI2002 – Assignment #2
Algorithms: Greedy Best-First Search (GBFS) | A* Search
Heuristics : Manhattan Distance | Euclidean Distance
GUI        : Pygame  (settings dialog via Tkinter)
"""

import pygame
import tkinter as tk
from tkinter import ttk, messagebox
import math
import heapq
import random
import time
import sys

# ─────────────────────────── COLOURS ────────────────────────────
WHITE       = (255, 255, 255)
BLACK       = (15,  15,  15)
GRAY        = (180, 180, 180)
DARK_GRAY   = (80,  80,  80)
BG_COLOR    = (28,  28,  30)
PANEL_COLOR = (44,  44,  46)
PANEL_LIGHT = (58,  58,  62)
BORDER_COL  = (72,  72,  76)
TEXT_COLOR  = (230, 230, 230)
DIM_TEXT    = (140, 140, 150)
ACCENT      = (10,  132, 255)
SUCCESS     = (48,  209,  88)
WARNING     = (255, 159,  10)
DANGER      = (255,  69,  58)

# Cell palette
C_EMPTY    = (245, 245, 247)
C_WALL     = (30,  30,  32)
C_START    = (255, 149,  0)   # orange
C_GOAL     = (0,   199, 190)  # teal
C_FRONTIER = (255, 214,  10)  # yellow
C_VISITED  = (88,  86, 214)   # indigo / blue-purple
C_PATH     = (48,  209,  88)  # green
C_AGENT    = (255,  55,  95)  # pink-red
C_GRID     = (200, 200, 205)  # grid lines

# ─────────────────────────── CELL STATES ────────────────────────
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
FRONTIER = 4
VISITED  = 5
PATH     = 6
AGENT    = 7

STATE_COLORS = {
    EMPTY:    C_EMPTY,
    WALL:     C_WALL,
    START:    C_START,
    GOAL:     C_GOAL,
    FRONTIER: C_FRONTIER,
    VISITED:  C_VISITED,
    PATH:     C_PATH,
    AGENT:    C_AGENT,
}

PANEL_W = 290
MIN_CELL = 8
FPS      = 60


# ════════════════════════════════════════════════════════════════
#  NODE
# ════════════════════════════════════════════════════════════════
class Node:
    __slots__ = ("row", "col", "g", "h", "f", "parent")

    def __init__(self, row: int, col: int):
        self.row    = row
        self.col    = col
        self.g      = 0
        self.h      = 0
        self.f      = 0
        self.parent = None

    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.row, self.col))

    def pos(self):
        return (self.row, self.col)

# ════════════════════════════════════════════════════════════════
#  HEURISTICS
# ════════════════════════════════════════════════════════════════
def manhattan(r1, c1, r2, c2) -> float:
    return abs(r1 - r2) + abs(c1 - c2)

def euclidean(r1, c1, r2, c2) -> float:
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

# ════════════════════════════════════════════════════════════════
#  SEARCH ALGORITHMS  
# ════════════════════════════════════════════════════════════════
def get_neighbors(grid, rows, cols, node):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = node.row + dr, node.col + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            yield (nr, nc)

def gbfs(grid, rows, cols, start_pos, goal_pos, heuristic_fn):
    t0 = time.perf_counter()
    start_node = Node(*start_pos)
    goal_r, goal_c = goal_pos
    start_node.h = heuristic_fn(start_node.row, start_node.col, goal_r, goal_c)
    start_node.f = start_node.h

    open_heap   = [start_node]
    open_set    = {start_pos: start_node}
    closed_set  = set()
    nodes_visited = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        pos = current.pos()
        if pos in closed_set: continue
        closed_set.add(pos)
        nodes_visited += 1

        if pos != start_pos and pos != goal_pos: grid[current.row][current.col] = VISITED
        yield (set(open_set.keys()) - closed_set, closed_set.copy())

        if pos == goal_pos:
            path = []
            node = current
            while node: path.append(node.pos()); node = node.parent
            path.reverse()
            yield ("DONE", path, nodes_visited, len(path) - 1, (time.perf_counter() - t0) * 1000)
            return

        for nr, nc in get_neighbors(grid, rows, cols, current):
            npos = (nr, nc)
            if npos in closed_set: continue
            if npos not in open_set:
                child = Node(nr, nc)
                child.h = heuristic_fn(nr, nc, goal_r, goal_c)
                child.f = child.h
                child.parent = current
                open_set[npos] = child
                heapq.heappush(open_heap, child)
                if npos != goal_pos: grid[nr][nc] = FRONTIER

    yield ("NO_PATH", [], nodes_visited, 0, (time.perf_counter() - t0) * 1000)

def astar(grid, rows, cols, start_pos, goal_pos, heuristic_fn):
    t0 = time.perf_counter()
    start_node = Node(*start_pos)
    goal_r, goal_c = goal_pos
    start_node.h = heuristic_fn(start_node.row, start_node.col, goal_r, goal_c)
    start_node.f = start_node.h

    open_heap   = [start_node]
    open_set    = {start_pos: start_node}
    closed_set  = set()
    nodes_visited = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        pos = current.pos()
        if pos in closed_set: continue
        closed_set.add(pos)
        nodes_visited += 1

        if pos != start_pos and pos != goal_pos: grid[current.row][current.col] = VISITED
        yield (set(open_set.keys()) - closed_set, closed_set.copy())

        if pos == goal_pos:
            path = []
            node = current
            while node: path.append(node.pos()); node = node.parent
            path.reverse()
            yield ("DONE", path, nodes_visited, len(path) - 1, (time.perf_counter() - t0) * 1000)
            return

        for nr, nc in get_neighbors(grid, rows, cols, current):
            npos = (nr, nc)
            if npos in closed_set: continue
            tentative_g = current.g + 1
            if npos not in open_set or tentative_g < open_set[npos].g:
                child = Node(nr, nc)
                child.g = tentative_g
                child.h = heuristic_fn(nr, nc, goal_r, goal_c)
                child.f = child.g + child.h
                child.parent = current
                open_set[npos] = child
                heapq.heappush(open_heap, child)
                if npos != goal_pos: grid[nr][nc] = FRONTIER

    yield ("NO_PATH", [], nodes_visited, 0, (time.perf_counter() - t0) * 1000)

# ════════════════════════════════════════════════════════════════
#  SETTINGS DIALOG  (Tkinter)
# ════════════════════════════════════════════════════════════════
class SettingsDialog:
    def __init__(self):
        self.result = None
        self.root   = tk.Tk()
        self.root.title("Dynamic Pathfinding Agent – Settings")
        self.root.resizable(False, False)
        self.root.configure(bg="#1c1c1e")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel",       background="#1c1c1e", foreground="#e5e5ea", font=("Segoe UI", 10))
        style.configure("Header.TLabel",background="#1c1c1e", foreground="#ffffff", font=("Segoe UI", 13, "bold"))
        style.configure("TFrame",       background="#1c1c1e")
        style.configure("TButton",      font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TCombobox",    fieldbackground="#2c2c2e", background="#2c2c2e", foreground="#e5e5ea")
        style.configure("TSpinbox",     fieldbackground="#2c2c2e", foreground="#e5e5ea")
        style.map("TCombobox",          fieldbackground=[("readonly", "#2c2c2e")])

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self.root.eval("tk::PlaceWindow . center")
        self.root.mainloop()

    def _build(self):
        pad = {"padx": 14, "pady": 6}

        ttk.Label(self.root, text="Dynamic Pathfinding Agent", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(18, 4), **{k: v for k, v in pad.items() if k == "padx"})
        ttk.Label(self.root, text="AI2002 – Assignment #2", foreground="#8e8e93",
                  background="#1c1c1e", font=("Segoe UI", 9)).grid(
            row=1, column=0, columnspan=2, pady=(0, 14), **{k: v for k, v in pad.items() if k == "padx"})

        sep = tk.Frame(self.root, height=1, bg="#3a3a3c"); sep.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=4)

        # Grid size
        self._field(3, "Grid Rows:",    "rows_var",    tk.IntVar(value=25), 5, 60)
        self._field(4, "Grid Columns:", "cols_var",    tk.IntVar(value=40), 5, 80)
        self._field(5, "Obstacle Density (%):", "density_var", tk.IntVar(value=30), 0, 70)

        # Algorithm
        ttk.Label(self.root, text="Algorithm:").grid(row=6, column=0, sticky="e", **pad)
        self.algo_var = tk.StringVar(value="A*")
        cb_algo = ttk.Combobox(self.root, textvariable=self.algo_var,
                               values=["A*", "Greedy Best-First (GBFS)"],
                               state="readonly", width=26)
        cb_algo.grid(row=6, column=1, sticky="w", **pad)

        # Heuristic
        ttk.Label(self.root, text="Heuristic:").grid(row=7, column=0, sticky="e", **pad)
        self.heur_var = tk.StringVar(value="Manhattan")
        cb_heur = ttk.Combobox(self.root, textvariable=self.heur_var,
                                values=["Manhattan", "Euclidean"],
                                state="readonly", width=26)
        cb_heur.grid(row=7, column=1, sticky="w", **pad)

        sep2 = tk.Frame(self.root, height=1, bg="#3a3a3c"); sep2.grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=14, pady=6)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=(4, 16))
        tk.Button(btn_frame, text="Launch", width=12, bg="#0a84ff", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  activebackground="#0060cc", activeforeground="white",
                  command=self._launch).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Cancel", width=10, bg="#3a3a3c", fg="#e5e5ea",
                  font=("Segoe UI", 10), relief="flat",
                  activebackground="#4a4a4f", activeforeground="white",
                  command=self._cancel).pack(side="left", padx=4)

    def _field(self, row, label, attr, var, lo, hi):
        pad = {"padx": 14, "pady": 6}
        setattr(self, attr, var)
        ttk.Label(self.root, text=label).grid(row=row, column=0, sticky="e", **pad)
        sb = ttk.Spinbox(self.root, from_=lo, to=hi, textvariable=var, width=10)
        sb.grid(row=row, column=1, sticky="w", **pad)

    def _launch(self):
        try:
            rows    = int(self.rows_var.get())
            cols    = int(self.cols_var.get())
            density = int(self.density_var.get())
            if not (5 <= rows <= 60): raise ValueError("Rows must be 5–60")
            if not (5 <= cols <= 80): raise ValueError("Cols must be 5–80")
            if not (0 <= density <= 70): raise ValueError("Density 0–70 %")
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e)); return

        self.result = {
            "rows":      rows,
            "cols":      cols,
            "density":   density / 100,
            "algorithm": "astar" if self.algo_var.get() == "A*" else "gbfs",
            "heuristic": "manhattan" if self.heur_var.get() == "Manhattan" else "euclidean",
        }
        self.root.destroy()

    def _cancel(self):
        self.root.destroy()
