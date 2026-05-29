"""
Demo — scramble a cube, print it, solve it, print again.
Good for visually confirming the solver works.
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from base_cube import BaseCube
from cube_solver import HybridSolver, _apply_move_to_obj, ALL_MOVES

N_SCRAMBLE = 8   # change this to test different depths

random.seed(None)  # random seed each run

cube = BaseCube(3)
scramble = []
for _ in range(N_SCRAMBLE):
    m = random.choice(ALL_MOVES)
    _apply_move_to_obj(cube, m)
    scramble.append(m)

print(f"Scramble ({N_SCRAMBLE} moves): {' '.join(scramble)}")
print("\nScrambled cube:")
print(cube)

solver = HybridSolver(cube)
solver.solution = []
solution = solver.solve(verbose=True, time_limit=30.0)

print("\nSolved cube:")
print(cube)

if solution:
    print(f"Solution ({len(solution)} moves): {' '.join(solution)}")
else:
    print("Could not solve within time limit.")
