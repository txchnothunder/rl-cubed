"""
cube2x2.py

Subclass of BaseCube for 2x2 Rubik's Cube.
"""

from src.base_cube import BaseCube

class Cube2x2(BaseCube):
    def __init__(self):
        super().__init__(2)

# Example usage
if __name__ == "__main__":
    cube = Cube2x2()
    print(cube)
