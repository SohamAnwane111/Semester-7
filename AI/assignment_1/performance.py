class performanceStats:
  """
  Class to store and display performance statistics for the puzzle solver.
  """
  def __init__(self, execution_time: float = 0.0, execution_memory: float = 0.0, num_moves: int = 0, path: list[str] = None) -> None:
    """
    Initializes the performance statistics.

    Args:
        execution_time (float, optional): Time taken to solve the puzzle.
        execution_memory (float, optional): Memory used during execution.
        num_moves (int, optional): Number of moves in the solution.
        path (list[str], optional): List of moves to solve the puzzle.
    """
    self.execution_time : float = execution_time
    self.execution_memory : float = execution_memory
    self.num_moves : int = num_moves
    self.path: list[str] = path if path else []

  def show_stats(self) -> None:
    """
    Prints the performance statistics to the console.
    """
    print("Number of moves needed : ", self.num_moves)
    print("Execution Time         : ", self.execution_time, "secs")
    print("Execution Memory       : ", self.execution_memory, "bytes")
    
    if self.path == "Unsolvable configuration":
      print("Unsolvable configuration")
    else:
      print("Path                   : ", " -> ".join(self.path) if self.path else "No path found")
