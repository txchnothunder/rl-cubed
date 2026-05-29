# VS Code Testing Setup — rl-cubed

## One-time setup

Open a terminal in VS Code (Terminal → New Terminal) and run:

```bash
python -m venv .venv
```

Then activate it:
- **Mac/Linux:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

Then install dependencies:

```bash
pip install -r requirements.txt
```

VS Code will ask "We noticed a new virtual environment — do you want to use it?" — click **Yes**.

---

## Running tests

### Option 1 — Run & Debug panel (easiest)

Press `F5` or go to **Run → Start Debugging**.  
Pick from the dropdown:

| Config | What it does |
|---|---|
| **Run: Quick Solve Test** | 5 scrambles at different depths, prints cubes and solutions |
| **Run: Full Test Suite** | All correctness and solve-rate tests |
| **Run: Demo** | Random scramble + solve + print (good for spot-checking) |
| **Run: Visualizer** | Opens the 3D Policy Iteration animation window |
| **Debug: cube_solver.py** | Runs the built-in test harness with breakpoints available |

### Option 2 — Terminal

```bash
# Quick test
python tests/quick_solve_test.py

# Full suite
python tests/test_solver.py

# Demo (different random scramble each run)
python tests/demo.py

# Pytest (if you have it installed)
pytest tests/test_solver.py -v
```

---

## Setting breakpoints

Open `src/cube_solver.py` in VS Code. Click in the left gutter (line numbers) to place a red dot. Then press `F5` with any debug config — execution will pause at your breakpoint and you can inspect variables in the left panel.

---

## Changing scramble depth

In `tests/demo.py`, edit this line:
```python
N_SCRAMBLE = 8   # change this
```

In `src/cube_solver.py`, the `solve()` method takes a `time_limit` parameter (seconds). Increase it for harder scrambles:
```python
solver.solve(verbose=True, time_limit=60.0)
```
