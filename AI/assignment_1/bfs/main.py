import json
import os
import time
import tkinter as tk
import tracemalloc
from collections import deque

from assignment_1.performance import performanceStats
from assignment_1.puzzle import eightTilePuzzle
from assignment_1.ui import PuzzleUI

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
config_path = os.path.abspath(config_path) 

with open(config_path, "r") as f:
    config = json.load(f)

def BFS(puzzle: eightTilePuzzle) -> performanceStats:
    """
    Performs Breadth-First Search (BFS) to solve the 8-tile puzzle.

    Args:
        puzzle (eightTilePuzzle): The puzzle instance to solve.

    Returns:
        performanceStats: Object containing execution time, memory usage, number of moves, and the solution path.
    """
    if puzzle.isSolvable() is False:
        return performanceStats(0.0, 0.0, -1, "Unsolvable configuration") 
        
    start_time: float = time.time()
    tracemalloc.start()

    visited_states = set()
    queue = deque()
    nodes_expanded: int = 0

    start_board = puzzle.getBoard()
    queue.append((start_board, start_board.index(0), [])) # --> (current_state, blank_index, current_path)
    visited_states.add(tuple(start_board))

    moves = [
        ("up", puzzle.moveUp),
        ("down", puzzle.moveDown),
        ("left", puzzle.moveLeft),
        ("right", puzzle.moveRight)
    ]

    while queue:
        current_state, blank_index, path = queue.popleft()

        if current_state == puzzle.final_board:
            end_time: float = time.time()
            execution_time: float = end_time - start_time
            execution_memory, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return performanceStats(
                execution_time=execution_time,
                execution_memory=execution_memory,
                num_moves=len(path),
                path=path,
                nodes_expanded=nodes_expanded
            )

        nodes_expanded += 1

        for move_name, move_func in moves:
            result = move_func(current_state, blank_index)
            if result:
                new_state, new_blank_index = result
                state_tuple = tuple(new_state)
                if state_tuple not in visited_states:
                    visited_states.add(state_tuple)
                    queue.append((new_state, new_blank_index, path + [move_name]))

    tracemalloc.stop()
    return performanceStats(0.0, 0.0, -1, "Unsolvable configuration") 

if __name__ == "__main__":
    """
    Main entry point for the 8-tile puzzle BFS visualizer.
    Loads configuration, solves the puzzle using BFS, displays performance stats,
    and launches the Tkinter UI for visualization.
    """
    initial = config["puzzle"]["initial_board"]
    final = config["puzzle"]["goal_board"]

    puzzle = eightTilePuzzle(initial, final)
    stats = BFS(puzzle)
    stats.show_stats()

    root: tk.Tk = tk.Tk()
    root.title("8 Puzzle BFS Visualizer")

    ui: PuzzleUI = PuzzleUI(root, puzzle, stats)
    ui.animate_path(initial, stats.path)

    root.mainloop()