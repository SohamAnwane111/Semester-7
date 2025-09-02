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

GOAL_STATE = tuple(config["puzzle"]["goal_board"])
MAX_OPTIMAL_MOVES = 100

# Time complexity  : O( summation[l = 0 -> d - 1] (d - l) b^l )
# Space complexity : O(b * d)

def ids_iterator(
    puzzle,
    curr_board: tuple[int, ...],
    blank_index: int,
    depth: int,
    max_depth: int,
    path: list[str],
    moves,
    visited: set[tuple[int, ...]],
    nodes_expanded: list[int]
) -> list[str] | None:
    """
    Depth-Limited Search (DLS) with cycle pruning.
    """
    if curr_board == GOAL_STATE:
        return path.copy()

    if depth >= max_depth:
        return None

    nodes_expanded[0] += 1

    for move_name, move_func in moves:
        result = move_func(list(curr_board), blank_index)
        if result:
            new_state, new_blank = result
            new_state_tuple = tuple(new_state)

            if new_state_tuple in visited:
                continue

            visited.add(new_state_tuple)
            path.append(move_name)

            solution = ids_iterator(
                puzzle,
                new_state_tuple,
                new_blank,
                depth + 1,
                max_depth,
                path,
                moves,
                visited,
                nodes_expanded
            )
            if solution is not None:
                return solution

            path.pop() 
            # visited.remove(new_state_tuple)

    return None


def ids(puzzle: eightTilePuzzle) -> performanceStats:
    """
    Solves the 8-puzzle using optimized Iterative Deepening Search (IDS).
    """
    if not puzzle.isSolvable():
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
    start_tuple: tuple[int, ...] = tuple(start_board)
    nodes_expanded: list[int] = [0]

    solution_path: list[str] | None = None

    for depth_limit in range(MAX_OPTIMAL_MOVES + 1):
        print(f"[IDS] Trying depth limit = {depth_limit} ...")

        visited: set[tuple[int, ...]] = {start_tuple}
        solution_path = ids_iterator(
            puzzle,
            start_tuple,
            blank_index,
            0,
            depth_limit,
            [],
            moves,
            visited,
            nodes_expanded
        )

        if solution_path is not None:
            print(f"[IDS] Solution found at depth {depth_limit}!")
            break

    end_time: float = time.time()
    execution_memory, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return performanceStats(
        execution_time=end_time - start_time,
        execution_memory=execution_memory,
        num_moves=len(solution_path) if solution_path else -1,
        path=solution_path or ["No solution found"],
        nodes_expanded=nodes_expanded[0]
    )


if __name__ == "__main__":
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
