# WARNING: This file is entirely copied from GPT (No one likes to create UIs hehe)

import time
import tkinter as tk
from typing import List

from performance import performanceStats
from puzzle import eightTilePuzzle


class PuzzleUI:
    """
    A class to handle the Tkinter-based UI for visualizing the 8-tile puzzle solution.
    Displays the puzzle board and animates the solution path.
    """
    def __init__(self, root: tk.Tk, puzzle: eightTilePuzzle, stats: performanceStats) -> None:
        """
        Initializes the PuzzleUI with the main window, puzzle instance, and performance stats.

        Args:
            root (tk.Tk): The main Tkinter window.
            puzzle (eightTilePuzzle): The puzzle instance.
            stats (performanceStats): The performance statistics of the solution.
        """
        self.root: tk.Tk = root
        self.puzzle: eightTilePuzzle = puzzle
        self.stats: performanceStats = stats
        self.labels: List[List[tk.Label]] = []

        for i in range(3):
            row: List[tk.Label] = []
            for j in range(3):
                lbl: tk.Label = tk.Label(
                    root,
                    text="",
                    font=("Helvetica", 32),
                    width=4,
                    height=2,
                    borderwidth=2,
                    relief="solid"
                )
                lbl.grid(row=i, column=j, padx=5, pady=5)
                row.append(lbl)
            self.labels.append(row)

        self.stats_label: tk.Label = tk.Label(root, text="", font=("Helvetica", 14))
        self.stats_label.grid(row=4, column=0, columnspan=3)

    def draw_board(self, board: List[int]) -> None:
        """
        Draws the current state of the puzzle board on the UI.

        Args:
            board (List[int]): The current board configuration.
        """
        for i in range(3):
            for j in range(3):
                val: int = board[i * 3 + j]
                self.labels[i][j].config(text=str(val) if val != 0 else "")

    def animate_path(self, initial_board: List[int], path: List[str]) -> None:
        """
        Animates the solution path on the puzzle board.

        Args:
            initial_board (List[int]): The starting board configuration.
            path (List[str]): The list of moves to solve the puzzle.
        """
        board: List[int] = initial_board[:]
        blank_index: int = board.index(0)

        self.draw_board(board)
        self.root.update()
        time.sleep(1)  

        for move in path:
            if move == "up":
                new_blank: int = blank_index - 3
            elif move == "down":
                new_blank = blank_index + 3
            elif move == "left":
                new_blank = blank_index - 1
            elif move == "right":
                new_blank = blank_index + 1
            else:
                continue  

            board[blank_index], board[new_blank] = board[new_blank], board[blank_index]
            blank_index = new_blank

            self.draw_board(board)
            self.root.update()
            time.sleep(0.5)

       
        self.stats_label.config(
            text=f"Moves: {self.stats.num_moves} | Time: {self.stats.execution_time:.4f}s | Memory: {self.stats.execution_memory:.4f} Bytes"
        )