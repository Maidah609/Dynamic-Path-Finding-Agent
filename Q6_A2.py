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
    """Greedy Best-First Search"""
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
            if npos in closed_set: continue
            if npos not in open_set:
                child = Node(nr, nc)
                child.h = heuristic_fn(nr, nc, goal_r, goal_c)
                child.f = child.h
                child.parent = current
                open_set[npos] = child
                heapq.heappush(open_heap, child)
                if npos != goal_pos: grid[nr][nc] = FRONTIER

    elapsed = (time.perf_counter() - t0) * 1000
    yield ("NO_PATH", [], nodes_visited, 0, elapsed)

def astar(grid, rows, cols, start_pos, goal_pos, heuristic_fn):
    """A* Search"""
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

    elapsed = (time.perf_counter() - t0) * 1000
    yield ("NO_PATH", [], nodes_visited, 0, elapsed)
