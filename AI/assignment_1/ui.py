import tkinter as tk
from typing import List

from assignment_1.performance import performanceStats
from assignment_1.puzzle import eightTilePuzzle


class PuzzleUI:
    """
    Tkinter-based UI for visualizing the 8-tile puzzle solution with smooth animations.
    """
    def __init__(self, root: tk.Tk, puzzle: eightTilePuzzle, stats: performanceStats) -> None:
        self.root: tk.Tk = root
        self.puzzle: eightTilePuzzle = puzzle
        self.stats: performanceStats = stats
        self.labels: List[List[tk.Label]] = []

        tile_bg = "#7700C7"  
        empty_bg = "#2C2C2C" 
        text_color = "white"

        for i in range(3):
            row: List[tk.Label] = []
            for j in range(3):
                lbl: tk.Label = tk.Label(
                    root,
                    text="",
                    font=("Helvetica", 36, "bold"),
                    width=4,
                    height=2,
                    bg=tile_bg,
                    fg=text_color,
                    borderwidth=3,
                    relief="raised"
                )
                lbl.grid(row=i, column=j, padx=6, pady=6, sticky="nsew")
                row.append(lbl)
            self.labels.append(row)

        for i in range(3):
            root.grid_rowconfigure(i, weight=1)
            root.grid_columnconfigure(i, weight=1)

        self.stats_label: tk.Label = tk.Label(
            root,
            text="",
            font=("Helvetica", 14, "bold"),
            fg="white",
            bg="#1E1E1E",
            pady=10
        )
        self.stats_label.grid(row=4, column=0, columnspan=3, sticky="ew")

        self.tile_bg = tile_bg
        self.empty_bg = empty_bg

    def draw_board(self, board: List[int]) -> None:
        """
        Draws the current state of the puzzle board on the UI.
        """
        for i in range(3):
            for j in range(3):
                val: int = board[i * 3 + j]
                if val == 0:
                    self.labels[i][j].config(text="", bg=self.empty_bg, relief="sunken")
                else:
                    self.labels[i][j].config(text=str(val), bg=self.tile_bg, relief="raised")

    def animate_path(self, initial_board: List[int], path: List[str]) -> None:
        """
        Smooth animation of puzzle solution path.
        """
        self.board: List[int] = initial_board[:]
        self.blank_index: int = self.board.index(0)
        self.path: List[str] = path
        self.step_index: int = 0

        self.draw_board(self.board)
        self.root.after(700, self._animate_step)  

    def _animate_step(self):
        if self.step_index >= len(self.path):
            self.stats_label.config(
                text=f"Moves: {self.stats.num_moves} | "
                     f"Time: {self.stats.execution_time:.4f}s | "
                     f"Memory: {self.stats.execution_memory:.4f} Bytes"
            )
            return

        move = self.path[self.step_index]

        if move == "up":
            new_blank = self.blank_index - 3
        elif move == "down":
            new_blank = self.blank_index + 3
        elif move == "left":
            new_blank = self.blank_index - 1
        elif move == "right":
            new_blank = self.blank_index + 1
        else:
            self.step_index += 1
            self.root.after(250, self._animate_step)
            return

        self.board[self.blank_index], self.board[new_blank] = self.board[new_blank], self.board[self.blank_index]
        self.blank_index = new_blank
        self.draw_board(self.board)
        self.step_index += 1
        self.root.after(250, self._animate_step)
