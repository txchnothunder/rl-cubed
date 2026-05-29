"""
cube3x3.py

Subclass of BaseCube for standard 3x3 Rubik's Cube.
"""

from src.base_cube import BaseCube

class Cube3x3(BaseCube):
    def __init__(self):
        super().__init__(3)

# Example usage
if __name__ == "__main__":
    cube = Cube3x3()
    print(cube)
