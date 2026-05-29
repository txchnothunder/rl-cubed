# Rubik's Cube RL Solver

Reinforcement learning and hybrid search approaches for solving the 3×3 Rubik's Cube across a state space of **4.3 × 10¹⁹** configurations.

---

## Original Course Project

This project began as a group assignment for **CS 273P — Machine Learning and Data Mining** at UC Irvine (Fall 2025).

**Team members and roles:**
- **PJ** — Environment Lead
- **Julia** — Data Lead
- **Ethan** — RL Algorithm Lead
- **Franz** — Evaluation & Integration Lead

The original submission included the cube environment (`environment.py`), Policy Iteration solver (`policy_iteration_solver.py`), and 3D visualization (`rubiks_visualizer.py`).

---

## Post-Course Improvements (Ethan, Dec 2025 – present)

After the course ended, I continued developing the solver independently. The main changes are in `src/cube_solver.py`:

### Algorithm Overhaul

The original `HybridSolver` had a hardcoded `iteration > 5` cap that limited it to 5 solve attempts before giving up — the primary reason it could only handle a small fraction of scrambles.

The rewritten solver uses three phases with no solve-count cap, running until a configurable time limit:

**Phase 1 — IDDFS (Iterative Deepening DFS)**
- Optimal for short scrambles (≤ 7 moves)
- O(depth) memory — no state explosion
- Prunes redundant move sequences (same-face repeats, inverse-face pairs)

**Phase 2 — Beam Search**
- Keeps the top-200 states per depth level ranked by sticker correctness
- Handles medium-depth scrambles (7–15 moves) that IDDFS can't reach in the time budget
- This is the biggest driver of the performance improvement

**Phase 3 — Greedy restarts + macros**
- Restarts from the best-known state with random perturbations
- Applies known macro-algorithms (sexy move, sune, T-perm, etc.) to escape local minima
- Runs until solved or time limit — no arbitrary restart cap

### Performance

| Scramble depth | Original success rate | Improved success rate |
|---|---|---|
| 1–3 moves | ~60% | ~100% |
| 4–6 moves | ~30% | ~95% |
| 7–10 moves | ~5% | ~70% |
| 11–15 moves | ~1% | ~40% |

*Tested with a 20-second time limit per solve.*

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Visualizer (Policy Iteration)

```bash
python rubiks_visualizer.py
```

### Run the Hybrid Solver

```python
import sys
sys.path.insert(0, 'src')
from base_cube import BaseCube
from cube_solver import HybridSolver, _apply_move_to_obj, ALL_MOVES
import random

# Create and scramble a cube
cube = BaseCube(3)
for _ in range(8):
    _apply_move_to_obj(cube, random.choice(ALL_MOVES))

# Solve
solver = HybridSolver(cube)
solution = solver.solve(verbose=True, time_limit=30.0)

if solution:
    print(f"Solved in {len(solution)} moves: {' '.join(solution)}")
else:
    print("Not solved within time limit")
```

### Run the Test Suite

```bash
cd src
python cube_solver.py
```

---

## Project Structure

```
rl-cubed/
├── rubiks_visualizer.py          # 3D visualization (original course deliverable)
├── src/
│   ├── base_cube.py              # Core cube mechanics — turns, rotations, state
│   ├── cube3x3.py                # 3×3 Rubik's cube
│   ├── cube2x2.py                # 2×2 pocket cube
│   ├── cube_solver.py            # Hybrid solver (IDDFS + Beam + Greedy) — improved post-course
│   ├── policy_iteration_solver.py  # RL Policy Iteration solver (original)
│   └── environment.py            # RL training environment (original)
├── data/                         # Training / testing data
├── agents/                       # Saved model checkpoints
├── results/                      # Output plots and logs
├── main.py                       # Basic cube operation tests
└── requirements.txt
```

---

## How Policy Iteration Works (original RL approach)

1. Generate random scrambled cube states as training set
2. Initialize a random policy (action per state)
3. Evaluate: compute value of each state under current policy
4. Improve: update policy to choose higher-value actions
5. Repeat until convergence
6. Follow learned policy at solve time

**Reward function:** +100 for a full solve · sticker-proportional reward (0–10) · −1 per move

---

## How the Hybrid Solver Works (improved approach)

See **Phase 1/2/3** description above. Key insight: the original greedy approach gets trapped in local maxima because sticker-count is a non-monotonic heuristic — improving one face can temporarily worsen another. Beam search and IDDFS avoid this by exploring multiple paths simultaneously or exhaustively.

---

## Troubleshooting

**Animation doesn't show:** Make sure `tkinter` is installed (`pip install tk` — usually bundled with Python). If the window appears but is blank, try changing the matplotlib backend on line 10 of `rubiks_visualizer.py`.

**Import errors:** Run `pip install -r requirements.txt` from the project root.

**Solver doesn't solve:** Increase `time_limit` in the `solve()` call. Very deep scrambles (20+ moves) require significantly more search time.

---

## Dependencies

- `numpy` — numerical operations
- `matplotlib` — 3D visualization and plotting
- Standard library: `random`, `time`, `collections`

---

## License

MIT License — feel free to use and modify.
