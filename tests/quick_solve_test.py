"""
Quick solve test — 5 scrambles at different depths.
Run this first to verify everything is working.
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from base_cube import BaseCube
from cube_solver import HybridSolver, _apply_move_to_obj, ALL_MOVES

def run():
    random.seed(42)
    test_cases = [
        ("Very Easy", 3),
        ("Easy",      6),
        ("Medium",    9),
        ("Hard",     12),
        ("Deep",     15),
    ]

    print("=" * 50)
    print("QUICK SOLVE TEST — 5 scrambles")
    print("=" * 50)

    for label, n_moves in test_cases:
        cube = BaseCube(3)
        scramble = []
        for _ in range(n_moves):
            m = random.choice(ALL_MOVES)
            _apply_move_to_obj(cube, m)
            scramble.append(m)

        print(f"\n[{label}] {n_moves}-move scramble: {' '.join(scramble)}")
        print("Cube before solving:")
        print(cube)

        solver = HybridSolver(cube)
        solver.solution = []
        solution = solver.solve(verbose=True, time_limit=20.0)

        if solution:
            print(f"\nSolution ({len(solution)} moves): {' '.join(solution)}")
        else:
            print("\nNot solved within time limit.")
        print("-" * 50)

if __name__ == "__main__":
    run()
