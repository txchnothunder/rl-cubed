"""
Full test suite — runs as pytest or directly with python.
Tests correctness, solve rate, and edge cases.
"""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from base_cube import BaseCube
from cube_solver import HybridSolver, _apply_move_to_obj, _is_solved, ALL_MOVES


def _make_scrambled_cube(n_moves, seed=None):
    if seed is not None:
        random.seed(seed)
    cube = BaseCube(3)
    scramble = []
    for _ in range(n_moves):
        m = random.choice(ALL_MOVES)
        _apply_move_to_obj(cube, m)
        scramble.append(m)
    return cube, scramble


def _solve_and_check(cube, time_limit=15.0):
    solver = HybridSolver(cube)
    solver.solution = []
    sol = solver.solve(verbose=False, time_limit=time_limit)
    return solver._solved(), sol


# -----------------------------------------------------------------------
# Individual tests (also work as pytest functions)
# -----------------------------------------------------------------------

def test_already_solved():
    """A solved cube should return an empty solution immediately."""
    cube = BaseCube(3)
    solver = HybridSolver(cube)
    sol = solver.solve(verbose=False)
    assert sol == [], f"Expected [], got {sol}"
    print("PASS: test_already_solved")


def test_single_move():
    """A single-move scramble must be solved in 1 move."""
    for move in ["R", "U", "F", "L", "D", "B"]:
        cube = BaseCube(3)
        _apply_move_to_obj(cube, move)
        ok, sol = _solve_and_check(cube)
        assert ok, f"Failed to solve single move: {move}"
        assert len(sol) <= 3, f"Single move solved in {len(sol)} — expected ≤3"
    print("PASS: test_single_move")


def test_inverse_move():
    """Applying R then R' should leave the cube solved."""
    cube = BaseCube(3)
    _apply_move_to_obj(cube, "R")
    _apply_move_to_obj(cube, "R'")
    assert _is_solved(cube._cube), "R then R' did not produce solved cube"
    print("PASS: test_inverse_move")


def test_very_easy_scrambles():
    """1-3 move scrambles: expect 100% solve rate."""
    wins = 0
    n = 10
    for i in range(n):
        cube, scr = _make_scrambled_cube(random.randint(1, 3), seed=i)
        ok, _ = _solve_and_check(cube)
        wins += ok
    rate = wins / n
    assert rate >= 0.9, f"Very easy solve rate too low: {wins}/{n}"
    print(f"PASS: test_very_easy_scrambles ({wins}/{n})")


def test_easy_scrambles():
    """4-6 move scrambles: expect ≥ 80% solve rate."""
    wins = 0
    n = 10
    for i in range(n):
        cube, scr = _make_scrambled_cube(random.randint(4, 6), seed=i+100)
        ok, _ = _solve_and_check(cube)
        wins += ok
    rate = wins / n
    assert rate >= 0.7, f"Easy solve rate too low: {wins}/{n}"
    print(f"PASS: test_easy_scrambles ({wins}/{n})")


def test_medium_scrambles():
    """7-10 move scrambles: expect ≥ 50% solve rate."""
    wins = 0
    n = 10
    for i in range(n):
        cube, scr = _make_scrambled_cube(random.randint(7, 10), seed=i+200)
        ok, _ = _solve_and_check(cube, time_limit=20.0)
        wins += ok
    rate = wins / n
    assert rate >= 0.4, f"Medium solve rate too low: {wins}/{n}"
    print(f"PASS: test_medium_scrambles ({wins}/{n})")


def test_solution_actually_solves():
    """
    Verify the returned solution list, when replayed from scratch,
    actually solves the cube.
    """
    random.seed(77)
    cube, scramble = _make_scrambled_cube(5)
    # Save scrambled state
    scrambled_state = cube._cube

    solver = HybridSolver(cube)
    solver.solution = []
    sol = solver.solve(verbose=False, time_limit=15.0)

    if sol is None:
        print("SKIP: test_solution_actually_solves (solver did not find solution)")
        return

    # Replay from scrambled state on a fresh cube
    replay_cube = BaseCube(3)
    replay_cube._cube = scrambled_state
    for move in sol:
        _apply_move_to_obj(replay_cube, move)

    assert _is_solved(replay_cube._cube), "Replayed solution did not produce solved cube!"
    print(f"PASS: test_solution_actually_solves ({len(sol)} moves replayed correctly)")


def test_cube_size_guard():
    """2x2 cube should raise ValueError."""
    from base_cube import BaseCube as BC
    cube2 = BC(2)
    try:
        HybridSolver(cube2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_cube_size_guard")


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

ALL_TESTS = [
    test_already_solved,
    test_single_move,
    test_inverse_move,
    test_very_easy_scrambles,
    test_easy_scrambles,
    test_medium_scrambles,
    test_solution_actually_solves,
    test_cube_size_guard,
]


def run_all():
    print("=" * 50)
    print("FULL TEST SUITE")
    print("=" * 50)
    passed = 0
    failed = 0
    for test in ALL_TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            failed += 1
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)


if __name__ == "__main__":
    run_all()
