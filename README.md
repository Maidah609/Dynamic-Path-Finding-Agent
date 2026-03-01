# Dynamic Pathfinding Agent

This repository contains the source code for the **Dynamic Pathfinding Agent**, developed for AI2002 – Assignment #2. It is a Python-based visualizer built with Pygame and Tkinter that demonstrates how informed search algorithms navigate grid-based environments, specifically when faced with dynamic, real-time obstacles.

**Author:** Rayyan Asim  
**Roll Number:** [Insert your XXF-YYYY Roll Number here]  

## Features

* **Informed Search Algorithms:** Implements both **A* Search** ($f(n) = g(n) + h(n)$) and **Greedy Best-First Search (GBFS)** ($f(n) = h(n)$).
* **Heuristic Functions:** Supports toggling between **Manhattan Distance** and **Euclidean Distance** for cost estimation.
* **Dynamic Grid Environment:** * Configurable grid dimensions (Rows x Columns).
  * Random map generation with user-defined obstacle density (e.g., 30%).
  * Interactive map editor: Manually place or remove walls, and set Start/Goal nodes using mouse clicks.
* **Dynamic Obstacle Simulation:** Features a "Dynamic Mode" where new obstacles spawn probabilistically while the agent is in transit. The agent detects collisions and recalculates the path in real-time without resetting the entire grid.
* **Real-Time Visualization & Metrics:**
  * Color-coded node states: Frontier (Yellow), Visited (Indigo), Final Path (Green), Start (Orange), Goal (Teal).
  * Live dashboard tracking: Total Nodes Visited, Final Path Cost, Execution Time (ms), and Re-plan counts.

## Requirements

* Python 3.7+
* `pygame` (for the main simulation GUI)
* `tkinter` (Standard Python library, used for the initial settings dialog)
