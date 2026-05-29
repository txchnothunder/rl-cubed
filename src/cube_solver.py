"""
cube_solver.py

Improved Hybrid Solver:
1. IDDFS for short scrambles (optimal, O(depth) memory)
2. Beam search for medium scrambles (explores top-k states per step)
3. Greedy restarts + macros for deeper scrambles
4. No hardcoded solve-count cap
"""

import random
import time
import sys
import os
import heapq

sys.path.insert(0, os.path.dirname(__file__))
from base_cube import BaseCube

ALL_MOVES = [
    "R", "R'", "R2",
    "L", "L'", "L2",
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
]

OPPOSITE_FACE = {"R": "L", "L": "R", "U": "D", "D": "U", "F": "B", "B": "F"}
SOLVED_STATE = BaseCube(3)._cube


def _apply_move_to_obj(cube_obj, move: str):
    face = move[0]
    modifier = move[1:] if len(move) > 1 else ""
    degrees = 180 if modifier == "2" else (270 if modifier == "'" else 90)
    if face == "R":   cube_obj.turn_right(3, degrees)
    elif face == "L": cube_obj.turn_right(1, 360 - degrees if degrees != 180 else 180)
    elif face == "U": cube_obj.turn_clockwise(1, degrees)
    elif face == "D": cube_obj.turn_clockwise(3, 360 - degrees if degrees != 180 else 180)
    elif face == "F": cube_obj.turn_upward(3, 360 - degrees if degrees != 180 else 180)
    elif face == "B": cube_obj.turn_upward(1, degrees)


def _apply_move(cube_str: str, move: str) -> str:
    tmp = BaseCube(3)
    tmp._cube = cube_str
    _apply_move_to_obj(tmp, move)
    return tmp._cube


def _is_solved(s: str) -> bool:
    for f in range(6):
        st = f * 9
        c = s[st]
        if any(s[st + i] != c for i in range(1, 9)):
            return False
    return True


def _count_correct(s: str) -> int:
    return sum(a == b for a, b in zip(s, SOLVED_STATE))


# ---------------------------------------------------------------------------
# IDDFS (depth ≤ 7 — reliably fast)
# ---------------------------------------------------------------------------

def _iddfs(start_str: str, max_depth: int, deadline: float):
    def dfs(state, depth, path, last_face, last2_face):
        if time.time() > deadline:
            return None
        if _is_solved(state):
            return path[:]
        if depth == 0:
            return None
        for move in ALL_MOVES:
            mf = move[0]
            if mf == last_face:
                continue
            if last2_face and mf == OPPOSITE_FACE.get(last_face) and last2_face == mf:
                continue
            path.append(move)
            r = dfs(_apply_move(state, move), depth - 1, path, mf, last_face)
            if r is not None:
                return r
            path.pop()
        return None

    for depth in range(max_depth + 1):
        if time.time() > deadline:
            return None
        r = dfs(start_str, depth, [], "", "")
        if r is not None:
            return r
    return None


# ---------------------------------------------------------------------------
# Beam search (width-first, keeps top-k states by score)
# ---------------------------------------------------------------------------

def _beam_search(start_str: str, beam_width: int, max_depth: int, deadline: float):
    """
    Keeps top `beam_width` states at each depth.  Memory = O(beam_width * depth).
    Returns list of moves or None.
    """
    # Each entry: (neg_score, state_str, path)
    beam = [(-_count_correct(start_str), start_str, [])]
    seen = {start_str}

    for depth in range(max_depth):
        if time.time() > deadline:
            return None
        candidates = []
        for _, state, path in beam:
            last_face = path[-1][0] if path else ""
            for move in ALL_MOVES:
                if move[0] == last_face:
                    continue
                new_state = _apply_move(state, move)
                if new_state in seen:
                    continue
                seen.add(new_state)
                new_path = path + [move]
                if _is_solved(new_state):
                    return new_path
                score = _count_correct(new_state)
                candidates.append((-score, new_state, new_path))

        if not candidates:
            return None
        # Keep top beam_width
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    return None


# ---------------------------------------------------------------------------
# HybridSolver
# ---------------------------------------------------------------------------

class HybridSolver:
    """
    Hybrid Rubik's Cube solver — no hardcoded solve-count cap.

    Phase 1: IDDFS depth ≤ 7  (optimal for very short scrambles)
    Phase 2: Beam search depth ≤ 20, width 200  (handles medium scrambles)
    Phase 3: Greedy restarts + macros until time_limit
    """

    MACROS = {
        "sexy":         "R U R' U'",
        "sledgehammer": "R' F R F'",
        "sune":         "R U R' U R U2 R'",
        "antisune":     "R U2 R' U' R U' R'",
        "double_sexy":  "R U R' U' R U R' U'",
        "Tperm":        "R U R' U' R' F R2 U' R' U' R U R' F'",
        "Jperm":        "R U R' F' R U R' U' R' F R2 U' R'",
        "fish":         "R U R' U R U2 R' U",
    }

    def __init__(self, cube):
        self.cube = cube
        self.size = cube.size
        self.solution = []
        if self.size != 3:
            raise ValueError("HybridSolver only supports 3x3 cubes")

    def _state(self): return self.cube._cube
    def _set_state(self, s): self.cube._cube = s

    def _apply(self, move):
        _apply_move_to_obj(self.cube, move)
        self.solution.append(move)

    def _apply_seq(self, seq):
        for m in seq.split():
            if m: self._apply(m)

    def _score(self): return _count_correct(self._state())
    def _solved(self): return _is_solved(self._state())

    def _solve_iddfs(self, max_depth=7, budget=12.0):
        r = _iddfs(self._state(), max_depth, time.time() + budget)
        if r is not None:
            for m in r: self._apply(m)
            return True
        return False

    def _solve_beam(self, width=200, max_depth=25, budget=30.0):
        r = _beam_search(self._state(), width, max_depth, time.time() + budget)
        if r is not None:
            for m in r: self._apply(m)
            return True
        return False

    def _greedy_pass(self, max_steps=500):
        stuck = 0
        for _ in range(max_steps):
            if self._solved(): return True
            saved = self._state()
            best_score = self._score()
            best_move = None
            for move in ALL_MOVES:
                s = _count_correct(_apply_move(saved, move))
                if s > best_score:
                    best_score = s
                    best_move = move
            if best_move:
                self._apply(best_move)
                stuck = 0
            else:
                self._apply(random.choice(ALL_MOVES))
                stuck += 1
                if stuck > 12:
                    return False
        return self._solved()

    def _macro_sweep(self, rotations=4):
        for _ in range(rotations):
            for seq in self.MACROS.values():
                ss, sl = self._state(), len(self.solution)
                old = self._score()
                self._apply_seq(seq)
                if self._solved(): return True
                if self._score() <= old + 2:
                    self._set_state(ss)
                    self.solution = self.solution[:sl]
            self._apply("U")
        return self._solved()

    def solve(self, verbose=True, time_limit=120.0):
        """
        Returns move list or None.  Runs until time_limit — no solve-count cap.
        """
        if verbose:
            print("=" * 55)
            print("HYBRID SOLVER  (IDDFS + Beam + Greedy restarts + Macros)")
            print("=" * 55)

        if self._solved():
            if verbose: print("Already solved!")
            return []

        start = time.time()
        deadline = start + time_limit

        if verbose: print(f"Start: {self._score()}/54 correct")

        # Phase 1 — IDDFS (fast for ≤7 move scrambles)
        budget1 = min(12.0, time_limit * 0.15)
        if verbose: print(f"Phase 1: IDDFS depth ≤ 7  ({budget1:.0f}s budget)")
        if self._solve_iddfs(max_depth=7, budget=budget1):
            if verbose:
                print(f"Solved by IDDFS in {len(self.solution)} moves "
                      f"({time.time()-start:.2f}s)")
            return self.solution

        # Phase 2 — Beam search (handles medium scrambles well)
        budget2 = min(40.0, (deadline - time.time()) * 0.5)
        if budget2 > 2:
            if verbose: print(f"Phase 2: Beam search width=200  ({budget2:.0f}s budget)")
            if self._solve_beam(width=200, max_depth=30, budget=budget2):
                if verbose:
                    print(f"Solved by beam in {len(self.solution)} moves "
                          f"({time.time()-start:.2f}s)")
                return self.solution

        if verbose: print("Phase 3: Greedy restarts + macros ...")

        # Phase 3 — greedy restarts with no cap
        best_score = self._score()
        best_state = self._state()
        best_sol = list(self.solution)
        restart = 0

        while time.time() < deadline:
            restart += 1
            if self._macro_sweep(rotations=4): break
            if self._greedy_pass(max_steps=600): break
            if self._macro_sweep(rotations=4): break

            curr = self._score()
            if curr > best_score:
                best_score = curr
                best_state = self._state()
                best_sol = list(self.solution)

            if verbose and restart % 10 == 0:
                print(f"  Restart {restart}: {curr}/54  ({time.time()-start:.1f}s)")

            self._set_state(best_state)
            self.solution = list(best_sol)
            for _ in range(random.randint(1, 5)):
                self._apply(random.choice(ALL_MOVES))

        elapsed = time.time() - start
        if self._solved():
            if verbose:
                print(f"Solved in {len(self.solution)} moves ({elapsed:.2f}s)")
            return self.solution
        else:
            if verbose:
                print(f"Not solved — best {self._score()}/54 ({elapsed:.2f}s)")
            return None


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _scramble_cube(cube_obj, n):
    moves = []
    for _ in range(n):
        m = random.choice(ALL_MOVES)
        _apply_move_to_obj(cube_obj, m)
        moves.append(m)
    return moves


def test_hybrid_solver(seed=42):
    random.seed(seed)
    print("=" * 55)
    print("HYBRID SOLVER TEST SUITE")
    print("=" * 55)

    buckets = [
        ("Very Easy (1-3)", 1, 3, 5),
        ("Easy     (4-6)", 4, 6, 5),
        ("Medium   (7-10)", 7, 10, 5),
        ("Hard     (11-15)", 11, 15, 5),
    ]

    total_ok = 0
    total_n = 0
    for label, lo, hi, n in buckets:
        wins = 0
        print(f"\n{label}")
        print("-" * 35)
        for t in range(n):
            cube = BaseCube(3)
            scr = _scramble_cube(cube, random.randint(lo, hi))
            solver = HybridSolver(cube)
            solver.solution = []
            sol = solver.solve(verbose=False, time_limit=20.0)
            ok = solver._solved()
            wins += ok
            mv = len(sol) if sol else "-"
            print(f"  Test {t+1}: scramble={len(scr)}  sol={mv}  "
                  f"{'OK' if ok else 'FAIL'}")
        print(f"  => {wins}/{n}  ({wins/n*100:.0f}%)")
        total_ok += wins
        total_n += n

    print(f"\n{'='*55}")
    print(f"OVERALL: {total_ok}/{total_n}  ({total_ok/total_n*100:.0f}%)")
    print("=" * 55)


if __name__ == "__main__":
    test_hybrid_solver()
