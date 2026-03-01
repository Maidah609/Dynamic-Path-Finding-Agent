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

    # Heap comparison – tie-break on h so both GBFS and A* work naturally
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
#  SEARCH ALGORITHMS  (generator – yield for step visualisation)
# ════════════════════════════════════════════════════════════════
def get_neighbors(grid, rows, cols, node):
    """Return 4-connected non-wall neighbours."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = node.row + dr, node.col + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            yield (nr, nc)


def gbfs(grid, rows, cols, start_pos, goal_pos, heuristic_fn):
    """
    Greedy Best-First Search  f(n) = h(n).
    Yields (frontier_set, visited_set) for visualisation, then returns
    (path_as_list_of_pos, nodes_visited, path_cost, elapsed_ms).
    """
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

        if pos in closed_set:
            continue
        closed_set.add(pos)
        nodes_visited += 1

        if pos != start_pos and pos != goal_pos:
            grid[current.row][current.col] = VISITED

        # Yield for step visualisation
        frontier_poses = set(open_set.keys()) - closed_set
        yield (frontier_poses, closed_set.copy())

        if pos == goal_pos:
            path = []
            node = current
            while node:
                path.append(node.pos())
                node = node.parent
            path.reverse()
            elapsed = (time.perf_counter() - t0) * 1000
            yield ("DONE", path, nodes_visited, len(path) - 1, elapsed)
            return

        for nr, nc in get_neighbors(grid, rows, cols, current):
            npos = (nr, nc)
            if npos in closed_set:
                continue
            if npos not in open_set:
                child = Node(nr, nc)
                child.h = heuristic_fn(nr, nc, goal_r, goal_c)
                child.f = child.h
                child.parent = current
                open_set[npos] = child
                heapq.heappush(open_heap, child)
                if npos != goal_pos:
                    grid[nr][nc] = FRONTIER
            # GBFS doesn't update g, so no re-insertion needed

    elapsed = (time.perf_counter() - t0) * 1000
    yield ("NO_PATH", [], nodes_visited, 0, elapsed)


def astar(grid, rows, cols, start_pos, goal_pos, heuristic_fn):
    """
    A* Search  f(n) = g(n) + h(n).
    Yields (frontier_set, visited_set) for visualisation, then returns
    (path_as_list_of_pos, nodes_visited, path_cost, elapsed_ms).
    """
    t0 = time.perf_counter()

    start_node = Node(*start_pos)
    goal_r, goal_c = goal_pos

    start_node.h = heuristic_fn(start_node.row, start_node.col, goal_r, goal_c)
    start_node.f = start_node.h

    open_heap   = [start_node]
    open_set    = {start_pos: start_node}   # pos -> best known node
    closed_set  = set()
    nodes_visited = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        pos = current.pos()

        if pos in closed_set:
            continue
        closed_set.add(pos)
        nodes_visited += 1

        if pos != start_pos and pos != goal_pos:
            grid[current.row][current.col] = VISITED

        frontier_poses = set(open_set.keys()) - closed_set
        yield (frontier_poses, closed_set.copy())

        if pos == goal_pos:
            path = []
            node = current
            while node:
                path.append(node.pos())
                node = node.parent
            path.reverse()
            elapsed = (time.perf_counter() - t0) * 1000
            yield ("DONE", path, nodes_visited, len(path) - 1, elapsed)
            return

        for nr, nc in get_neighbors(grid, rows, cols, current):
            npos = (nr, nc)
            if npos in closed_set:
                continue
            tentative_g = current.g + 1          # uniform cost grid
            if npos not in open_set or tentative_g < open_set[npos].g:
                child = Node(nr, nc)
                child.g = tentative_g
                child.h = heuristic_fn(nr, nc, goal_r, goal_c)
                child.f = child.g + child.h
                child.parent = current
                open_set[npos] = child
                heapq.heappush(open_heap, child)
                if npos != goal_pos:
                    grid[nr][nc] = FRONTIER

    elapsed = (time.perf_counter() - t0) * 1000
    yield ("NO_PATH", [], nodes_visited, 0, elapsed)


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


# ════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════
class PathfinderApp:

    # ── constants ──────────────────────────────────────────────
    SPAWN_PROB      = 0.04   # 4 % per step in dynamic mode
    AGENT_STEP_MS   = 120    # ms between agent moves
    SEARCH_STEP_MS  = 18     # ms between search-anim frames

    def __init__(self, cfg: dict):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent")

        self.rows      = cfg["rows"]
        self.cols      = cfg["cols"]
        self.density   = cfg["density"]
        self.algo_key  = cfg["algorithm"]
        self.heur_key  = cfg["heuristic"]

        # Determine cell pixel size
        screen_info = pygame.display.Info()
        avail_w = screen_info.current_w  - PANEL_W - 60
        avail_h = screen_info.current_h  - 80
        cell_w  = avail_w // self.cols
        cell_h  = avail_h // self.rows
        self.cs = max(MIN_CELL, min(cell_w, cell_h))   # cell size (square)

        self.grid_px_w = self.cols * self.cs
        self.grid_px_h = self.rows * self.cs
        self.win_w     = self.grid_px_w + PANEL_W
        self.win_h     = max(self.grid_px_h, 680)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_h1  = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_h2  = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.font_reg = pygame.font.SysFont("Segoe UI", 12)
        self.font_sm  = pygame.font.SysFont("Segoe UI", 11)
        self.font_ico = pygame.font.SysFont("Segoe UI", 20, bold=True)

        # State
        self.grid        = None
        self.start_pos   = (1, 1)
        self.goal_pos    = (self.rows - 2, self.cols - 2)
        self.path        = []
        self.agent_pos   = None
        self.agent_idx   = 0

        # Metrics
        self.nodes_visited  = 0
        self.path_cost      = 0
        self.exec_time_ms   = 0.0
        self.replan_count   = 0

        # Mode flags
        self.dynamic_mode   = False
        self.is_running     = False   # agent in motion
        self.is_searching   = False   # search animation active
        self.search_gen     = None
        self.status_msg     = "Configure the map, then press  ▶️ Start Search"
        self.status_color   = DIM_TEXT
        self.no_path_flag   = False
        self.placing_mode   = "wall"   # "wall" | "start" | "goal"

        # Timers
        self.last_agent_t  = 0
        self.last_search_t = 0

        # Buttons registry  { label: (rect, action) }
        self.buttons = {}

        # Init grid
        self._reset_grid()
        self._generate_random_map()

    # ─── GRID HELPERS ──────────────────────────────────────────
    def _reset_grid(self):
        self.grid = [[EMPTY] * self.cols for _ in range(self.rows)]
        self.grid[self.start_pos[0]][self.start_pos[1]] = START
        self.grid[self.goal_pos[0]][self.goal_pos[1]]   = GOAL

    def _clear_search_overlay(self):
        """Remove FRONTIER / VISITED / PATH / AGENT overlays."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (FRONTIER, VISITED, PATH, AGENT):
                    self.grid[r][c] = EMPTY
        self.grid[self.start_pos[0]][self.start_pos[1]] = START
        self.grid[self.goal_pos[0]][self.goal_pos[1]]   = GOAL

    def _generate_random_map(self):
        self._reset_grid()
        protected = {self.start_pos, self.goal_pos}
        # Protect start + goal neighbours so a path is likely reachable
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                protected.add((self.start_pos[0]+dr, self.start_pos[1]+dc))
                protected.add((self.goal_pos[0]+dr,  self.goal_pos[1]+dc))

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in protected and random.random() < self.density:
                    self.grid[r][c] = WALL

    def _paint_path(self):
        for pos in self.path:
            if pos != self.start_pos and pos != self.goal_pos:
                self.grid[pos[0]][pos[1]] = PATH

    def _grid_cell_from_px(self, px, py):
        col = (px) // self.cs
        row = (py) // self.cs
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None

    # ─── SEARCH INTEGRATION ────────────────────────────────────
    def _start_search(self, from_pos=None):
        self._clear_search_overlay()
        self.path         = []
        self.agent_pos    = None
        self.agent_idx    = 0
        self.is_running   = False
        self.no_path_flag = False

        src = from_pos or self.start_pos
        h_fn = manhattan if self.heur_key == "manhattan" else euclidean
        algo = gbfs if self.algo_key == "gbfs" else astar

        # Make a working copy of the grid for the search to annotate
        self.search_gen     = algo(self.grid, self.rows, self.cols,
                                   src, self.goal_pos, h_fn)
        self.is_searching   = True
        self.status_msg     = "Searching…"
        self.status_color   = WARNING
        self.last_search_t  = pygame.time.get_ticks()

    def _finish_search(self, path, nv, pc, et):
        self.nodes_visited  = nv
        self.path_cost      = pc
        self.exec_time_ms   = et
        self.path           = path
        self.is_searching   = False

        if path:
            self._paint_path()
            self.status_msg   = f"Path found!  Cost={pc}  Nodes={nv}  {et:.1f} ms"
            self.status_color = SUCCESS
        else:
            self.no_path_flag = True
            self.status_msg   = "No path found to goal."
            self.status_color = DANGER

    def _replan(self):
        """Re-run search from current agent position."""
        self.replan_count += 1
        self._clear_search_overlay()
        self._start_search(from_pos=self.agent_pos)
        self.status_msg = f"Re-planning… (replan #{self.replan_count})"

    # ─── DYNAMIC OBSTACLES ─────────────────────────────────────
    def _spawn_obstacle(self):
        if not self.dynamic_mode:
            return
        if random.random() > self.SPAWN_PROB:
            return
        # Pick a random non-special cell
        r = random.randint(0, self.rows - 1)
        c = random.randint(0, self.cols - 1)
        pos = (r, c)
        if (pos == self.start_pos or pos == self.goal_pos or
                pos == self.agent_pos or
                self.grid[r][c] in (WALL, START, GOAL, AGENT)):
            return
        self.grid[r][c] = WALL
        # Check if it blocks the remaining path
        if self.path and pos in self.path[self.agent_idx:]:
            self._replan()

    # ─── AGENT MOVEMENT ────────────────────────────────────────
    def _start_agent(self):
        if not self.path or len(self.path) < 2:
            return
        self.is_running   = True
        self.agent_idx    = 0
        self.agent_pos    = self.path[0]
        self.replan_count = 0
        self.last_agent_t = pygame.time.get_ticks()
        self._clear_search_overlay()
        self._paint_path()
        self.status_msg   = "Agent moving…"
        self.status_color = ACCENT

    def _step_agent(self):
        if not self.is_running or self.is_searching:
            return
        now = pygame.time.get_ticks()
        if now - self.last_agent_t < self.AGENT_STEP_MS:
            return
        self.last_agent_t = now

        self._spawn_obstacle()
        if self.is_searching:   # spawn triggered replan
            return

        if self.agent_idx >= len(self.path) - 1:
            # Reached goal
            self.is_running = False
            gr, gc = self.goal_pos
            self.grid[gr][gc] = GOAL
            self.status_msg   = (f"Goal reached!  Cost={self.path_cost}"
                                  f"  Re-plans={self.replan_count}")
            self.status_color = SUCCESS
            return

        # Un-mark previous cell
        pr, pc = self.path[self.agent_idx]
        if (pr, pc) != self.start_pos:
            self.grid[pr][pc] = VISITED

        self.agent_idx += 1
        self.agent_pos  = self.path[self.agent_idx]
        ar, ac = self.agent_pos
        if (ar, ac) != self.goal_pos:
            self.grid[ar][ac] = AGENT
        else:
            self.grid[ar][ac] = GOAL

    # ─── BUTTONS ───────────────────────────────────────────────
    def _build_buttons(self):
        self.buttons = {}
        bx = self.grid_px_w + 14
        bw = PANEL_W - 28
        bh = 34

        def reg(key, label, y, color=PANEL_LIGHT):
            rect = pygame.Rect(bx, y, bw, bh)
            self.buttons[key] = {"rect": rect, "label": label, "color": color}

        reg("search",   "▶️  Start Search",       198, (30, 120, 80))
        reg("run",      "⚡  Run Agent",          240, (20,  90, 170))
        reg("regen",    "⟳  Random Map",          294)
        reg("clear",    "✕  Clear Map",           336)
        reg("reset",    "↺  Full Reset",          378)
        reg("dynamic",  "◉  Dynamic Mode: OFF",   430, DARK_GRAY)
        reg("algo",     f"Algo: {self._algo_label()}",  494, PANEL_LIGHT)
        reg("heur",     f"Heur: {self._heur_label()}",  536, PANEL_LIGHT)
        reg("mode_wall","Place: Wall",             594, PANEL_LIGHT)
        reg("mode_start","Set Start",             634, PANEL_LIGHT)
        reg("mode_goal", "Set Goal",              674, PANEL_LIGHT)

    def _algo_label(self):
        return "A*" if self.algo_key == "astar" else "GBFS"

    def _heur_label(self):
        return "Manhattan" if self.heur_key == "manhattan" else "Euclidean"

    def _handle_button(self, key):
        if key == "search":
            if not self.is_running:
                self._start_search()

        elif key == "run":
            if self.path and not self.is_searching:
                if not self.is_running:
                    self._start_agent()
                else:
                    self.is_running = False
                    self.status_msg = "Agent paused."
                    self.status_color = DIM_TEXT

        elif key == "regen":
            self.is_running = self.is_searching = False
            self.path = []
            self._generate_random_map()
            self.status_msg   = "New random map generated."
            self.status_color = DIM_TEXT

        elif key == "clear":
            self.is_running = self.is_searching = False
            self.path = []
            self._reset_grid()
            self.status_msg   = "Map cleared."
            self.status_color = DIM_TEXT

        elif key == "reset":
            self.is_running = self.is_searching = False
            self.path = []; self.nodes_visited = 0
            self.path_cost = 0; self.exec_time_ms = 0
            self.replan_count = 0; self.no_path_flag = False
            self._reset_grid()
            self.status_msg   = "Reset complete."
            self.status_color = DIM_TEXT

        elif key == "dynamic":
            self.dynamic_mode = not self.dynamic_mode
            lbl = "ON" if self.dynamic_mode else "OFF"
            col = SUCCESS if self.dynamic_mode else DARK_GRAY
            self.buttons["dynamic"]["label"] = f"◉  Dynamic Mode: {lbl}"
            self.buttons["dynamic"]["color"] = col

        elif key == "algo":
            self.algo_key = "gbfs" if self.algo_key == "astar" else "astar"
            self.buttons["algo"]["label"] = f"Algo: {self._algo_label()}"

        elif key == "heur":
            self.heur_key = "euclidean" if self.heur_key == "manhattan" else "manhattan"
            self.buttons["heur"]["label"] = f"Heur: {self._heur_label()}"

        elif key == "mode_wall":
            self.placing_mode = "wall"
        elif key == "mode_start":
            self.placing_mode = "start"
        elif key == "mode_goal":
            self.placing_mode = "goal"

    # ─── MOUSE INTERACTION ─────────────────────────────────────
    _mouse_held = False
    _erase_mode = False   # True = erase walls when dragging

    def _handle_mouse_down(self, pos):
        # Panel buttons
        for key, btn in self.buttons.items():
            if btn["rect"].collidepoint(pos):
                self._handle_button(key)
                return

        # Grid interaction
        cell = self._grid_cell_from_px(pos[0], pos[1])
        if cell and not self.is_running:
            r, c = cell
            if self.placing_mode == "start":
                old_sr, old_sc = self.start_pos
                self.grid[old_sr][old_sc] = EMPTY
                self.start_pos = (r, c)
                self.grid[r][c] = START
            elif self.placing_mode == "goal":
                old_gr, old_gc = self.goal_pos
                self.grid[old_gr][old_gc] = EMPTY
                self.goal_pos = (r, c)
                self.grid[r][c] = GOAL
            else:
                # Toggle wall
                if (r, c) not in (self.start_pos, self.goal_pos):
                    if self.grid[r][c] == WALL:
                        self.grid[r][c] = EMPTY
                        PathfinderApp._erase_mode = True
                    else:
                        self.grid[r][c] = WALL
                        PathfinderApp._erase_mode = False
            PathfinderApp._mouse_held = True

    def _handle_mouse_drag(self, pos):
        if not PathfinderApp._mouse_held:
            return
        if self.placing_mode != "wall":
            return
        cell = self._grid_cell_from_px(pos[0], pos[1])
        if cell and not self.is_running:
            r, c = cell
            if (r, c) not in (self.start_pos, self.goal_pos):
                self.grid[r][c] = EMPTY if PathfinderApp._erase_mode else WALL

    # ─── DRAWING ───────────────────────────────────────────────
    def _draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                state = self.grid[r][c]
                color = STATE_COLORS.get(state, C_EMPTY)
                rect  = pygame.Rect(c * self.cs, r * self.cs, self.cs, self.cs)
                pygame.draw.rect(self.screen, color,  rect)
                if self.cs >= 6:
                    pygame.draw.rect(self.screen, C_GRID, rect, 1)

    def _draw_panel(self):
        px = self.grid_px_w
        # Background
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         pygame.Rect(px, 0, PANEL_W, self.win_h))
        pygame.draw.line(self.screen, BORDER_COL, (px, 0), (px, self.win_h), 2)

        x = px + 14
        y = 14

        # Title
        self._text(self.font_h1, "Pathfinding Agent", x, y, WHITE)
        y += 22
        self._text(self.font_sm, "AI2002 – Assignment #2", x, y, DIM_TEXT)
        y += 24

        # Divider
        pygame.draw.line(self.screen, BORDER_COL, (px+8, y), (px+PANEL_W-8, y))
        y += 10

        # Metrics
        self._text(self.font_h2, "METRICS", x, y, DIM_TEXT); y += 20
        self._metric_row(x, y, "Nodes Visited",  str(self.nodes_visited)); y += 22
        self._metric_row(x, y, "Path Cost",       str(self.path_cost));     y += 22
        self._metric_row(x, y, "Exec Time",
                         f"{self.exec_time_ms:.1f} ms");                    y += 22
        self._metric_row(x, y, "Re-plans",        str(self.replan_count));  y += 26

        pygame.draw.line(self.screen, BORDER_COL, (px+8, y), (px+PANEL_W-8, y))
        y += 10
        self._text(self.font_h2, "CONTROLS", x, y, DIM_TEXT); y += 20

        # Buttons (positions set by _build_buttons)
        for key, btn in self.buttons.items():
            is_active = (
                (key == "dynamic" and self.dynamic_mode) or
                (key == "mode_wall"  and self.placing_mode == "wall") or
                (key == "mode_start" and self.placing_mode == "start") or
                (key == "mode_goal"  and self.placing_mode == "goal")
            )
            col = btn["color"]
            if is_active:
                col = ACCENT
            pygame.draw.rect(self.screen, col, btn["rect"], border_radius=6)
            pygame.draw.rect(self.screen, BORDER_COL, btn["rect"], 1, border_radius=6)
            lbl_surf = self.font_reg.render(btn["label"], True, TEXT_COLOR)
            lx = btn["rect"].x + (btn["rect"].w - lbl_surf.get_width()) // 2
            ly = btn["rect"].y + (btn["rect"].h - lbl_surf.get_height()) // 2
            self.screen.blit(lbl_surf, (lx, ly))

        # Legend
        ly_start = self.win_h - 165
        pygame.draw.line(self.screen, BORDER_COL,
                         (px+8, ly_start-6), (px+PANEL_W-8, ly_start-6))
        self._text(self.font_h2, "LEGEND", x, ly_start, DIM_TEXT)
        ldata = [
            (C_START,    "Start Node"),
            (C_GOAL,     "Goal Node"),
            (C_AGENT,    "Agent"),
            (C_FRONTIER, "Frontier"),
            (C_VISITED,  "Visited"),
            (C_PATH,     "Path"),
            (C_WALL,     "Wall"),
        ]
        lx = x; ly = ly_start + 18
        for color, label in ldata:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(lx, ly, 14, 14), border_radius=3)
            pygame.draw.rect(self.screen, BORDER_COL,
                             pygame.Rect(lx, ly, 14, 14), 1, border_radius=3)
            self._text(self.font_sm, label, lx + 20, ly + 1, TEXT_COLOR)
            ly += 18

        # Status bar at bottom of panel
        status_rect = pygame.Rect(px+6, self.win_h - 34, PANEL_W-12, 28)
        pygame.draw.rect(self.screen, PANEL_LIGHT, status_rect, border_radius=5)
        self._text_wrap(self.font_sm, self.status_msg,
                        status_rect.x + 6, status_rect.y + 6,
                        status_rect.w - 12, self.status_color)

    def _metric_row(self, x, y, label, value):
        self._text(self.font_sm, label + ":", x, y, DIM_TEXT)
        surf = self.font_h2.render(value, True, WHITE)
        self.screen.blit(surf, (x + 130, y))

    def _text(self, font, txt, x, y, color):
        self.screen.blit(font.render(txt, True, color), (x, y))

    def _text_wrap(self, font, txt, x, y, max_w, color):
        words = txt.split()
        line  = ""
        for w in words:
            test = line + w + " "
            if font.size(test)[0] > max_w and line:
                self.screen.blit(font.render(line.strip(), True, color), (x, y))
                y += font.get_height()
                line = w + " "
            else:
                line = test
        if line:
            self.screen.blit(font.render(line.strip(), True, color), (x, y))

    # ─── SEARCH ANIMATION STEP ─────────────────────────────────
    def _advance_search(self):
        if not self.is_searching or self.search_gen is None:
            return
        now = pygame.time.get_ticks()
        if now - self.last_search_t < self.SEARCH_STEP_MS:
            return
        self.last_search_t = now

        try:
            result = next(self.search_gen)
        except StopIteration:
            self.is_searching = False
            return

        if isinstance(result, tuple) and isinstance(result[0], str):
            tag = result[0]
            if tag == "DONE":
                _, path, nv, pc, et = result
                self._finish_search(path, nv, pc, et)
            elif tag == "NO_PATH":
                _, path, nv, pc, et = result
                self._finish_search([], nv, pc, et)
        else:
            frontier_set, visited_set = result
            # Colour frontier cells (search algo already marks visited)
            for (fr, fc) in frontier_set:
                if (fr, fc) not in (self.start_pos, self.goal_pos):
                    self.grid[fr][fc] = FRONTIER

    # ─── MAIN LOOP ─────────────────────────────────────────────
    def run(self):
        self._build_buttons()

        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_SPACE:
                        self._handle_button("search")
                    elif event.key == pygame.K_RETURN:
                        self._handle_button("run")
                    elif event.key == pygame.K_r:
                        self._handle_button("regen")
                    elif event.key == pygame.K_c:
                        self._handle_button("clear")
                    elif event.key == pygame.K_d:
                        self._handle_button("dynamic")
                    elif event.key == pygame.K_TAB:
                        self._handle_button("algo")

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        PathfinderApp._mouse_held = False
                elif event.type == pygame.MOUSEMOTION:
                    if PathfinderApp._mouse_held:
                        self._handle_mouse_drag(event.pos)

            # Advance search animation
            self._advance_search()

            # After search animation finished – if agent was mid-run, resume
            if not self.is_searching and self.is_running:
                self._step_agent()

            # Draw
            self.screen.fill(BG_COLOR)
            self._draw_grid()
            self._draw_panel()
            pygame.display.flip()


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════
def main():
    dlg = SettingsDialog()
    if dlg.result is None:
        sys.exit(0)
    app = PathfinderApp(dlg.result)
    app.run()


if __name__ == "__main__":
    main()
