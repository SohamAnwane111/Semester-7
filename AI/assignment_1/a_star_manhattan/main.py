import json
import time
import tkinter as tk
import tracemalloc
import os
import heapq
from typing import Callable, Any
from assignment_1.performance import performanceStats
from assignment_1.puzzle import eightTilePuzzle
from assignment_1.ui import PuzzleUI

config_path: str = os.path.join(os.path.dirname(__file__), "..", "config.json")
config_path = os.path.abspath(config_path)

with open(config_path, "r") as f:
    config: dict[str, Any] = json.load(f)


def get_config_mapping() -> dict[int, int]:
    """
    Returns a mapping from tile value to its index in the goal configuration.

    Returns:
        dict[int, int]: Mapping from tile value to its goal index.
    """
    goal_config_mapping: dict[int, int] = {}
    goal_config: list[int] = config["puzzle"]["goal_board"]
    for i in range(len(goal_config)):
        goal_config_mapping[goal_config[i]] = i
    return goal_config_mapping


def compute_manhattan_difference(board: list[int]) -> int:
    """
    Computes the total Manhattan distance of the current board from the goal configuration.

    Args:
        board (list[int]): The current board configuration.

    Returns:
        int: The sum of Manhattan distances for all tiles.
    """
    cost: int = 0
    size: int = int(len(board) ** 0.5)
    goal_mapping: dict[int, int] = get_config_mapping()

    for idx, tile in enumerate(board):
        if tile == 0:
            continue

        row_curr: int
        col_curr: int
        row_curr, col_curr = divmod(idx, size)

        goal_idx: int = goal_mapping[tile]
        row_goal: int
        col_goal: int
        row_goal, col_goal = divmod(goal_idx, size)

        cost += abs(row_curr - row_goal) + abs(col_curr - col_goal)

    return cost


def compute_cost(level: int, board: list[int]) -> int:
    """
    Computes the total cost for A* (g(n) + h(n)).

    Args:
        level (int): The current depth (g(n)).
        board (list[int]): The current board configuration.

    Returns:
        int: The total cost.
    """
    return level + compute_manhattan_difference(board)


def a_star(puzzle: eightTilePuzzle) -> performanceStats:
    """
    Solves the 8-tile puzzle using the A* algorithm with Manhattan distance heuristic.

    Args:
        puzzle (eightTilePuzzle): The puzzle instance to solve.

    Returns:
        performanceStats: Object containing execution time, memory usage, number of moves, and the solution path.
    """
    if puzzle.isSolvable() is False:
        return performanceStats(0.0, 0.0, -1, "Unsolvable configuration")

    start_time: float = time.time()
    tracemalloc.start()

    start_board: list[int] = puzzle.getBoard()
    min_heap: list[tuple[int, int, list[int], int, list[str]]] = []
    visited_states: set[tuple[int, ...]] = set()

    initial_level: int = 0
    initial_cost: int = compute_cost(initial_level, start_board)

    heapq.heappush(
        min_heap, (initial_cost, initial_level, start_board, start_board.index(0), [])
    )
    visited_states.add(tuple(start_board))

    moves: list[tuple[str, Callable[[list[int], int], tuple[list[int], int] | None]]] = [
        ("up", puzzle.moveUp),
        ("down", puzzle.moveDown),
        ("left", puzzle.moveLeft),
        ("right", puzzle.moveRight),
    ]

    while min_heap:
        _: int
        current_level: int
        current_state: list[int]
        blank_index: int
        path: list[str]
        _, current_level, current_state, blank_index, path = heapq.heappop(
            min_heap
        )

        if current_state == puzzle.final_board:
            end_time: float = time.time()
            execution_time: float = end_time - start_time
            execution_memory: float
            execution_memory, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return performanceStats(
                execution_time=execution_time,
                execution_memory=execution_memory,
                num_moves=len(path),
                path=path,
            )

        for move_name, move_func in moves:
            result: tuple[list[int], int] | None = move_func(current_state, blank_index)
            if result:
                new_state: list[int]
                new_blank_index: int
                new_state, new_blank_index = result
                state_tuple: tuple[int, ...] = tuple(new_state)
                if state_tuple not in visited_states:
                    visited_states.add(state_tuple)
                    heapq.heappush(
                        min_heap,
                        (
                            compute_cost(current_level + 1, new_state),
                            current_level + 1,
                            new_state,
                            new_blank_index,
                            path + [move_name],
                        ),
                    )

    tracemalloc.stop()
    return performanceStats(0.0, 0.0, -1, "Unsolvable configuration")


if __name__ == "__main__":
    """
    Main entry point for the 8-tile puzzle A* visualizer.
    Loads configuration, solves the puzzle using A*, displays performance stats,
    and launches the Tkinter UI for visualization.
    """
    initial: list[int] = config["puzzle"]["initial_board"]
    final: list[int] = config["puzzle"]["goal_board"]

    puzzle: eightTilePuzzle = eightTilePuzzle(initial, final)
    stats: performanceStats = a_star(puzzle)
    stats.show_stats()

    root: tk.Tk = tk.Tk()
    root.title("8 Puzzle A* Visualizer")

    ui: PuzzleUI = PuzzleUI(root, puzzle, stats)
    ui.animate_path(initial, stats.path)

    root.mainloop()
