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
