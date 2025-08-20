from typing import List, Tuple
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

class eightTilePuzzle:
    """
    Class representing the 8-tile puzzle, including board state and valid moves.
    """
    def __init__(self, initial_board: List[int] = None, final_board: List[int] = None) -> None:
        """
        Initializes the puzzle with the given initial and final board configurations.

        Args:
            initial_board (List[int], optional): The starting board configuration.
            final_board (List[int], optional): The goal board configuration.
        """
        if initial_board is None:
            initial_board = config["default_board"]
        if final_board is None:
            final_board = config["default_board"]
        self.initial_board: List[int] = initial_board
        self.final_board: List[int] = final_board
        self.blank_index: int = self.initial_board.index(0)

    def isSolvable(self) -> bool:
        """
        Checks if the current puzzle configuration is solvable.

        Returns:
            bool: True if solvable, False otherwise.
        """
        inversions: int = 0
        for i in range(len(self.initial_board)):
            for j in range(i + 1, len(self.initial_board)):
                if self.initial_board[i] > self.initial_board[j] and self.initial_board[j] != 0:
                    inversions += 1
        return inversions % 2 == 0

    def moveUp(self, board: List[int], blank_index: int) -> Tuple[List[int], int] | None:
        """
        Moves the blank tile up if possible.

        Args:
            board (List[int]): The current board configuration.
            blank_index (int): The index of the blank tile.

        Returns:
            Tuple[List[int], int] | None: The new board and blank index, or None if move is invalid.
        """
        if blank_index >= 3:
            return self.__swap(board, blank_index, blank_index - 3)
        return None

    def moveDown(self, board: List[int], blank_index: int) -> Tuple[List[int], int] | None:
        """
        Moves the blank tile down if possible.

        Args:
            board (List[int]): The current board configuration.
            blank_index (int): The index of the blank tile.

        Returns:
            Tuple[List[int], int] | None: The new board and blank index, or None if move is invalid.
        """
        if blank_index <= 5:
            return self.__swap(board, blank_index, blank_index + 3)
        return None

    def moveLeft(self, board: List[int], blank_index: int) -> Tuple[List[int], int] | None:
        """
        Moves the blank tile left if possible.

        Args:
            board (List[int]): The current board configuration.
            blank_index (int): The index of the blank tile.

        Returns:
            Tuple[List[int], int] | None: The new board and blank index, or None if move is invalid.
        """
        if blank_index % 3 != 0:
            return self.__swap(board, blank_index, blank_index - 1)
        return None

    def moveRight(self, board: List[int], blank_index: int) -> Tuple[List[int], int] | None:
        """
        Moves the blank tile right if possible.

        Args:
            board (List[int]): The current board configuration.
            blank_index (int): The index of the blank tile.

        Returns:
            Tuple[List[int], int] | None: The new board and blank index, or None if move is invalid.
        """
        if blank_index % 3 != 2:
            return self.__swap(board, blank_index, blank_index + 1)
        return None

    def __swap(self, board: List[int], index1: int, index2: int) -> Tuple[List[int], int]:
        """
        Swaps two tiles on the board.

        Args:
            board (List[int]): The current board configuration.
            index1 (int): The first index to swap.
            index2 (int): The second index to swap.

        Returns:
            Tuple[List[int], int]: The new board and the new blank index.
        """
        new_board = board[:]
        new_board[index1], new_board[index2] = new_board[index2], new_board[index1]
        return new_board, index2

    def getBoard(self) -> List[int]:
        """
        Returns a copy of the initial board configuration.

        Returns:
            List[int]: The initial board.
        """
        return self.initial_board[:]