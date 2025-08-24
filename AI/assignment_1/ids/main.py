import json
import os
import time
import tkinter as tk
import tracemalloc
from typing import Any, Callable

from assignment_1.performance import performanceStats
from assignment_1.puzzle import eightTilePuzzle
from assignment_1.ui import PuzzleUI

config_path: str = os.path.join(os.path.dirname(__file__), "..", "config.json")
config_path = os.path.abspath(config_path)

with open(config_path, "r") as f:
    config: dict[str, Any] = json.load(f)


def ids_iterator(
    puzzle,
    curr_board: list[int],
    blank_index: int,
    curr_level: int,
    curr_path: list[str],
    max_level: int,
    moves,
    visited_states: set
) -> list[str] | None:
    """
    Recursive helper function for Iterative Deepening Search (IDS).
    Explores the puzzle state space up to a given depth limit.

    Args:
        puzzle: The eightTilePuzzle instance.
        curr_board (list[int]): Current board configuration.
        blank_index (int): Index of the blank tile (0).
        curr_level (int): Current depth in the search tree.
        curr_path (list[str]): List of moves taken so far.
        max_level (int): Maximum depth to search (depth limit).
        moves: List of possible moves (name, function).
        visited_states (set): Set of visited board states.

    Returns:
        list[str] | None: The solution path if found, else None.
    """
    if curr_level == max_level:
        if curr_board == config["puzzle"]["goal_board"]:
            return curr_path
        return None

    visited_states.add(tuple(curr_board))

    for move_name, move_func in moves:
        result: tuple[list[int], int] | None = move_func(curr_board, blank_index)
        if result:
            new_state, new_blank_index = result
            state_tuple: tuple[int, ...] = tuple(new_state)
            if state_tuple not in visited_states:
                solution = ids_iterator(
                    puzzle,
                    new_state,
                    new_blank_index,
                    curr_level + 1,
                    curr_path + [move_name],
                    max_level,
                    moves,
                    visited_states.copy() 
                )
                if solution is not None:
                    return solution
    return None


def ids(puzzle: eightTilePuzzle) -> performanceStats:
    """
    Solves the 8-tile puzzle using Iterative Deepening Search (IDS).

    Args:
        puzzle (eightTilePuzzle): The puzzle instance to solve.

    Returns:
        performanceStats: Object containing execution time, memory usage, number of moves, and the solution path.
    """
    if puzzle.isSolvable() is False:
        return performanceStats(0.0, 0.0, -1, "Unsolvable configuration")

    start_time: float = time.time()
    tracemalloc.start()

    moves: list[
        tuple[str, Callable[[list[int], int], tuple[list[int], int] | None]]
    ] = [
        ("up", puzzle.moveUp),
        ("down", puzzle.moveDown),
        ("left", puzzle.moveLeft),
        ("right", puzzle.moveRight),
    ]

    start_board: list[int] = puzzle.getBoard()
    blank_index: int = start_board.index(0)

    solution_path: list[str] | None = None
    depth_limit = 0

    while solution_path is None:
        print(f"[IDS] Trying depth limit = {depth_limit} ...")
        visited_states: set = set()
        solution_path = ids_iterator(
            puzzle,
            start_board,
            blank_index,
            0,
            [],
            depth_limit,
            moves,
            visited_states,
        )

        if solution_path is not None:
            print(f"[IDS] Solution found at depth {depth_limit}!")
            break

        depth_limit += 1

    end_time: float = time.time()
    execution_memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return performanceStats(
        execution_time=end_time - start_time,
        execution_memory=execution_memory,
        num_moves=len(solution_path),
        path=solution_path,
    )


if __name__ == "__main__":
    """
    Main entry point for the 8-tile puzzle IDS visualizer.
    Loads configuration, solves the puzzle using IDS, displays performance stats,
    and launches the Tkinter UI for visualization.
    """
    initial: list[int] = config["puzzle"]["initial_board"]
    final: list[int] = config["puzzle"]["goal_board"]

    puzzle: eightTilePuzzle = eightTilePuzzle(initial, final)
    stats: performanceStats = ids(puzzle)
    stats.show_stats()

    root: tk.Tk = tk.Tk()
    root.title("8 Puzzle IDS Visualizer")

    ui: PuzzleUI = PuzzleUI(root, puzzle, stats)
    ui.animate_path(initial, stats.path)

    root.mainloop()
