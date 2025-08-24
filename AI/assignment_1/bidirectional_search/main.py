import json
import time
import tkinter as tk
import tracemalloc
from collections import deque
import os

from assignment_1.performance import performanceStats
from assignment_1.puzzle import eightTilePuzzle
from assignment_1.ui import PuzzleUI

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
config_path = os.path.abspath(config_path)

with open(config_path, "r") as f:
    config = json.load(f)


def reconstruct_path(
    meet_state,
    forward_parents,
    backward_parents,
    start_state,
    goal_state,
):
    """
    Reconstructs the full path from the start state to the goal state via the meeting state.

    Args:
        meet_state (tuple): The state where the forward and backward searches meet.
        forward_parents (dict): Parent mapping for the forward search (state -> (parent, move)).
        backward_parents (dict): Parent mapping for the backward search (state -> (parent, move)).
        start_state (list): The initial board configuration.
        goal_state (list): The goal board configuration.

    Returns:
        list: The sequence of moves from start_state to goal_state.
    """
    inverse_move = {"up": "down", "down": "up", "left": "right", "right": "left"}

    # Forward path: meet -> start, then reversed
    path_fwd = []
    s = meet_state
    while s != tuple(start_state):
        parent, move = forward_parents[s]
        path_fwd.append(move)
        s = parent
    path_fwd.reverse()

    # Backward path: meet -> goal, invert moves
    path_bwd = []
    s = meet_state
    while s != tuple(goal_state):
        parent, move = backward_parents[s]
        path_bwd.append(inverse_move[move])
        s = parent

    return path_fwd + path_bwd


def BidirectionalBFS(puzzle: eightTilePuzzle) -> performanceStats:
    """
    Solves the 8-tile puzzle using Bidirectional Breadth-First Search (BFS).

    Args:
        puzzle (eightTilePuzzle): The puzzle instance to solve.

    Returns:
        performanceStats: An object containing execution time, memory usage, path length, and the solution path.
    """
    if puzzle.isSolvable() is False:
        return performanceStats(0.0, 0.0, -1, "Unsolvable configuration")

    start_time: float = time.time()
    tracemalloc.start()

    start_board = puzzle.getBoard()
    goal_board = puzzle.final_board

    start_state = tuple(start_board)
    goal_state = tuple(goal_board)

    if start_state == goal_state:
        end_time: float = time.time()
        execution_time: float = end_time - start_time
        execution_memory, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return performanceStats(execution_time, execution_memory, 0, [])

    forward_fringe_list = deque([(start_state, start_board.index(0))])
    backward_fringe_list = deque([(goal_state, goal_board.index(0))])

    forward_parents = {}
    backward_parents = {}

    visited_forwards = {start_state}
    visited_backwards = {goal_state}

    moves = [
        ("up", puzzle.moveUp),
        ("down", puzzle.moveDown),
        ("left", puzzle.moveLeft),
        ("right", puzzle.moveRight),
    ]

    while forward_fringe_list and backward_fringe_list:
        # Expand forward
        state, blank_idx = forward_fringe_list.popleft()
        board = list(state)

        for move_name, move_func in moves:
            result = move_func(board, blank_idx)
            if result:
                new_state, new_blank_idx = result
                state_tuple = tuple(new_state)

                if state_tuple not in visited_forwards:
                    forward_parents[state_tuple] = (state, move_name)
                    visited_forwards.add(state_tuple)
                    forward_fringe_list.append((state_tuple, new_blank_idx))

                    if state_tuple in visited_backwards:
                        # Found intersection
                        path = reconstruct_path(
                            state_tuple, forward_parents, backward_parents, start_board, goal_board
                        )
                        end_time: float = time.time()
                        execution_time: float = end_time - start_time
                        execution_memory, _ = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        return performanceStats(execution_time, execution_memory, len(path), path)

        # Expand backward
        state, blank_idx = backward_fringe_list.popleft()
        board = list(state)

        for move_name, move_func in moves:
            result = move_func(board, blank_idx)
            if result:
                new_state, new_blank_idx = result
                state_tuple = tuple(new_state)

                if state_tuple not in visited_backwards:
                    backward_parents[state_tuple] = (state, move_name)
                    visited_backwards.add(state_tuple)
                    backward_fringe_list.append((state_tuple, new_blank_idx))

                    if state_tuple in visited_forwards:
                        path = reconstruct_path(
                            state_tuple, forward_parents, backward_parents, start_board, goal_board
                        )
                        end_time: float = time.time()
                        execution_time: float = end_time - start_time
                        execution_memory, _ = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        return performanceStats(execution_time, execution_memory, len(path), path)

    tracemalloc.stop()
    return performanceStats(0.0, 0.0, -1, "No solution found (unexpected).")


if __name__ == "__main__":
    """
    Main entry point for the 8-tile puzzle Bidirectional BFS visualizer.

    Loads the puzzle configuration, solves the puzzle using Bidirectional BFS,
    displays performance statistics, and launches the Tkinter UI for visualization.
    """
    initial = config["puzzle"]["initial_board"]
    final = config["puzzle"]["goal_board"]

    puzzle = eightTilePuzzle(initial, final)
    stats = BidirectionalBFS(puzzle)
    stats.show_stats()

    root: tk.Tk = tk.Tk()
    root.title("8 Puzzle Bidirectional BFS Visualizer")

    ui = PuzzleUI(root, puzzle, stats)
    ui.animate_path(initial, stats.path)

    root.mainloop()
