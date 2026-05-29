import random
from base_cube import BaseCube


class CubeEnvironment:
    """
    An environment wrapper for Rubik's Cube that provides:
    - Automatic scrambling on initialization
    - Solution checking
    - State retrieval
    """
    
    def __init__(self, size=3, scramble_moves=20):
        """
        Initialize the cube environment.
        
        Args:
            size (int): Size of the cube (2 for 2x2x2, 3 for 3x3x3)
            scramble_moves (int): Number of random moves to scramble the cube
        """
        self.size = size
        self.cube = BaseCube(size)
        self.solved_state = self.cube._cube  # Store the solved state
        self.scramble_moves = scramble_moves
        self.move_history = []
        
        # Scramble the cube on initialization
        self.scramble()
    
    def scramble(self, num_moves=None):
        """
        Scramble the cube with random moves.
        
        Args:
            num_moves (int, optional): Number of moves to scramble. 
                                       If None, uses self.scramble_moves
        """
        if num_moves is None:
            num_moves = self.scramble_moves
        
        # Define all possible move types
        move_types = ['right', 'left', 'upward', 'downward', 'clockwise', 'counterclockwise']
        degrees = [90, 180, 270]
        
        self.move_history = []
        
        for _ in range(num_moves):
            move_type = random.choice(move_types)
            degree = random.choice(degrees)
            index = random.randint(1, self.size)
            
            # Execute the move
            if move_type == 'right':
                self.cube.turn_right(index, degree)
            elif move_type == 'left':
                self.cube.turn_left(index, degree)
            elif move_type == 'upward':
                self.cube.turn_upward(index, degree)
            elif move_type == 'downward':
                self.cube.turn_downward(index, degree)
            elif move_type == 'clockwise':
                self.cube.turn_clockwise(index, degree)
            elif move_type == 'counterclockwise':
                self.cube.turn_counterclockwise(index, degree)
            
            # Record the move
            self.move_history.append({
                'type': move_type,
                'index': index,
                'degree': degree
            })
    
    def is_solved(self):
        """
        Check if the cube is in a solved state.
        
        Returns:
            bool: True if the cube is solved, False otherwise
        """
        return self.cube._cube == self.solved_state
    
    def get_state(self):
        """
        Get the current state of the cube.
        
        Returns:
            str: The current cube state as a string
        """
        return self.cube._cube
    
    def get_state_array(self):
        """
        Get the current state as a 2D representation (6 faces).
        
        Returns:
            dict: Dictionary with face names as keys and 2D lists as values
        """
        faces = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
        step = self.size ** 2
        state = self.cube._cube
        
        face_arrays = {}
        for face_name, face_idx in faces.items():
            start = face_idx * step
            face_str = state[start:start + step]
            
            # Convert to 2D array
            face_2d = []
            for row in range(self.size):
                row_start = row * self.size
                face_2d.append(list(face_str[row_start:row_start + self.size]))
            
            face_arrays[face_name] = face_2d
        
        return face_arrays
    
    def reset(self):
        """
        Reset the cube to solved state.
        """
        self.cube = BaseCube(self.size)
        self.move_history = []
    
    def reset_and_scramble(self, num_moves=None):
        """
        Reset the cube and scramble it again.
        
        Args:
            num_moves (int, optional): Number of moves to scramble
        """
        self.reset()
        self.scramble(num_moves)
    
    def get_move_history(self):
        """
        Get the history of moves made during scrambling.
        
        Returns:
            list: List of dictionaries containing move information
        """
        return self.move_history.copy()
    
    def print_cube(self):
        """
        Print the current cube state in a readable format.
        """
        print(self.cube)
    
    def print_internal(self):
        """
        Print the internal string representation of the cube.
        """
        self.cube.print_internal()
    
    def __str__(self):
        """
        String representation of the environment.
        """
        status = "SOLVED" if self.is_solved() else "SCRAMBLED"
        return f"CubeEnvironment({self.size}x{self.size}x{self.size}) - Status: {status}"