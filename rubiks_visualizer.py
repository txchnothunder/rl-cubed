"""
Rubik's Cube Policy Iteration Visualizer

A single-file visualization of policy iteration algorithm solving a Rubik's cube.
Shows scramble, solving animation, and real-time metrics.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Windows compatibility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import time
from collections import defaultdict

# Enable interactive mode for smooth animations
plt.ion()


# ============================================================================
# CUBE IMPLEMENTATION
# ============================================================================

class RubiksCube:
    """Simple 3x3 Rubik's Cube implementation"""

    def __init__(self):
        self.size = 3
        self._cube = self._init_cube()

    def _init_cube(self):
        """Initialize a solved cube"""
        colors = ['W', 'R', 'G', 'Y', 'O', 'B']
        return ''.join([c * 9 for c in colors])

    def copy_state(self):
        """Return a copy of the cube state"""
        return self._cube

    def set_state(self, state):
        """Set the cube to a specific state"""
        self._cube = state

    def _rotate_face(self, cube_list, face_idx, clockwise=True):
        """Rotate a face 90 degrees"""
        start = face_idx * 9
        face = cube_list[start:start + 9]
        new_face = face[:]

        mapping = [6, 3, 0, 7, 4, 1, 8, 5, 2] if clockwise else [2, 5, 8, 1, 4, 7, 0, 3, 6]
        for i in range(9):
            new_face[i] = face[mapping[i]]

        cube_list[start:start + 9] = new_face

    def apply_move(self, move):
        """Apply a move like 'R', "R'", 'R2'"""
        face = move[0]
        modifier = move[1:] if len(move) > 1 else ''

        times = 2 if modifier == '2' else (3 if modifier == "'" else 1)

        for _ in range(times):
            self._apply_single_move(face)

    def _apply_single_move(self, face):
        """Apply a single 90-degree move"""
        cube = list(self._cube)

        if face == 'R':
            self._rotate_face(cube, 1, True)
            temp = [cube[8], cube[17], cube[26]]
            cube[8], cube[17], cube[26] = cube[44], cube[43], cube[42]
            cube[44], cube[43], cube[42] = cube[35], cube[34], cube[33]
            cube[35], cube[34], cube[33] = cube[2], cube[11], cube[20]
            cube[2], cube[11], cube[20] = temp[0], temp[1], temp[2]

        elif face == 'L':
            self._rotate_face(cube, 4, True)
            temp = [cube[0], cube[9], cube[18]]
            cube[0], cube[9], cube[18] = cube[36], cube[37], cube[38]
            cube[36], cube[37], cube[38] = cube[47], cube[46], cube[45]
            cube[47], cube[46], cube[45] = cube[6], cube[15], cube[24]
            cube[6], cube[15], cube[24] = temp[0], temp[1], temp[2]

        elif face == 'U':
            self._rotate_face(cube, 0, True)
            temp = cube[18:21]
            cube[18:21] = cube[9:12]
            cube[9:12] = cube[45:48]
            cube[45:48] = cube[36:39]
            cube[36:39] = temp

        elif face == 'D':
            self._rotate_face(cube, 3, True)
            temp = cube[24:27]
            cube[24:27] = cube[42:45]
            cube[42:45] = cube[51:54]
            cube[51:54] = cube[15:18]
            cube[15:18] = temp

        elif face == 'F':
            self._rotate_face(cube, 2, True)
            temp = [cube[6], cube[7], cube[8]]
            cube[6], cube[7], cube[8] = cube[44], cube[41], cube[38]
            cube[44], cube[41], cube[38] = cube[20], cube[19], cube[18]
            cube[20], cube[19], cube[18] = cube[9], cube[12], cube[15]
            cube[9], cube[12], cube[15] = temp[0], temp[1], temp[2]

        elif face == 'B':
            self._rotate_face(cube, 5, True)
            temp = [cube[0], cube[1], cube[2]]
            cube[0], cube[1], cube[2] = cube[11], cube[14], cube[17]
            cube[11], cube[14], cube[17] = cube[26], cube[25], cube[24]
            cube[26], cube[25], cube[24] = cube[36], cube[39], cube[42]
            cube[36], cube[39], cube[42] = temp[0], temp[1], temp[2]

        self._cube = ''.join(cube)

    def is_solved(self):
        """Check if cube is solved"""
        for i in range(6):
            start = i * 9
            face_color = self._cube[start]
            if not all(self._cube[start + j] == face_color for j in range(9)):
                return False
        return True

    def count_correct_pieces(self):
        """Count correctly positioned pieces"""
        solved = self._init_cube()
        return sum(1 for i in range(54) if self._cube[i] == solved[i])


# ============================================================================
# POLICY ITERATION SOLVER
# ============================================================================

class PolicyIterationSolver:
    """Policy iteration solver for Rubik's cube"""

    def __init__(self, cube):
        self.cube = cube
        self.actions = [
            "R", "R'", "R2", "L", "L'", "L2",
            "U", "U'", "U2", "D", "D'", "D2",
            "F", "F'", "F2", "B", "B'", "B2"
        ]
        self.policy = {}
        self.value_function = defaultdict(float)
        self.discount = 0.99

    def get_state_key(self):
        """Get current state as string"""
        return self.cube._cube

    def get_reward(self, next_state):
        """Calculate reward for reaching next_state"""
        prev_state = self.cube.copy_state()
        self.cube.set_state(next_state)

        if self.cube.is_solved():
            reward = 100.0
        else:
            correct = self.cube.count_correct_pieces()
            reward = correct / 54.0 * 10 - 1

        self.cube.set_state(prev_state)
        return reward

    def get_next_state(self, action):
        """Get next state after applying action"""
        prev_state = self.cube.copy_state()
        self.cube.apply_move(action)
        next_state = self.get_state_key()
        self.cube.set_state(prev_state)
        return next_state

    def generate_training_states(self, num_states=500):
        """Generate random scrambled states for training"""
        states = set()

        for _ in range(num_states):
            self.cube.set_state(self.cube._init_cube())
            num_moves = random.randint(1, 12)
            for _ in range(num_moves):
                self.cube.apply_move(random.choice(self.actions))

            state = self.get_state_key()
            states.add(state)

            if state not in self.policy:
                self.policy[state] = random.choice(self.actions)
                self.value_function[state] = 0.0

        return states

    def policy_evaluation(self, max_iter=50):
        """Evaluate current policy"""
        for _ in range(max_iter):
            delta = 0
            for state in list(self.policy.keys()):
                self.cube.set_state(state)
                if self.cube.is_solved():
                    self.value_function[state] = 0
                    continue

                action = self.policy[state]
                next_state = self.get_next_state(action)
                reward = self.get_reward(next_state)

                new_value = reward + self.discount * self.value_function[next_state]
                delta = max(delta, abs(new_value - self.value_function[state]))
                self.value_function[state] = new_value

            if delta < 0.01:
                break

    def policy_improvement(self):
        """Improve policy"""
        stable = True
        for state in list(self.policy.keys()):
            self.cube.set_state(state)
            if self.cube.is_solved():
                continue

            old_action = self.policy[state]
            best_value = float('-inf')
            best_action = old_action

            for action in self.actions:
                next_state = self.get_next_state(action)
                reward = self.get_reward(next_state)
                value = reward + self.discount * self.value_function[next_state]

                if value > best_value:
                    best_value = value
                    best_action = action

            self.policy[state] = best_action
            if old_action != best_action:
                stable = False

        return stable

    def train(self, num_states=300, max_iterations=20):
        """Train the policy"""
        print(f"Generating {num_states} training states...")
        self.generate_training_states(num_states)

        print(f"Training policy iteration...")
        for i in range(max_iterations):
            self.policy_evaluation()
            stable = self.policy_improvement()
            print(f"  Iteration {i+1}/{max_iterations}")
            if stable:
                print(f"  Converged at iteration {i+1}")
                break

        print(f"Training complete! Policy size: {len(self.policy)} states")

    def solve(self, max_moves=100):
        """Solve the cube using trained policy"""
        solution = []

        for _ in range(max_moves):
            if self.cube.is_solved():
                break

            state = self.get_state_key()

            if state in self.policy:
                action = self.policy[state]
            else:
                best_action = None
                best_reward = float('-inf')
                for action in self.actions:
                    next_state = self.get_next_state(action)
                    reward = self.get_reward(next_state)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = action
                action = best_action

            self.cube.apply_move(action)
            solution.append(action)

        return solution if self.cube.is_solved() else None


# ============================================================================
# 3D VISUALIZER
# ============================================================================

class CubeVisualizer:
    """3D visualizer for Rubik's cube with real-time animations"""

    COLOR_MAP = {
        'W': '#FFFFFF', 'R': '#FF0000', 'G': '#00FF00',
        'Y': '#FFFF00', 'O': '#FF8800', 'B': '#0000FF'
    }

    def __init__(self, cube):
        self.cube = cube
        self.fig = plt.figure(figsize=(14, 7))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax_metrics = self.fig.add_subplot(122)
        self.move_history = []
        self.correct_history = []

        # Show the window
        plt.show(block=False)

    def get_sticker_color(self, face_idx, sticker_idx):
        """Get color of a sticker"""
        pos = face_idx * 9 + sticker_idx
        color_letter = self.cube._cube[pos]
        return self.COLOR_MAP.get(color_letter, '#888888')

    def create_sticker(self, center, normal, size=0.9):
        """Create a square sticker polygon"""
        nx, ny, nz = normal
        cx, cy, cz = center
        offset = size / 2

        if abs(nx) > 0.5:
            corners = [
                [cx, cy - offset, cz - offset],
                [cx, cy + offset, cz - offset],
                [cx, cy + offset, cz + offset],
                [cx, cy - offset, cz + offset]
            ]
        elif abs(ny) > 0.5:
            corners = [
                [cx - offset, cy, cz - offset],
                [cx + offset, cy, cz - offset],
                [cx + offset, cy, cz + offset],
                [cx - offset, cy, cz + offset]
            ]
        else:
            corners = [
                [cx - offset, cy - offset, cz],
                [cx + offset, cy - offset, cz],
                [cx + offset, cy + offset, cz],
                [cx - offset, cy + offset, cz]
            ]

        return corners

    def draw_face(self, face_idx, face_name, offset, normal):
        """Draw a cube face"""
        polygons = []
        colors = []

        for row in range(3):
            for col in range(3):
                sticker_idx = row * 3 + col

                if face_name in ['U', 'D']:
                    x, y, z = offset[0] + (col - 1), offset[1] + (row - 1), offset[2]
                elif face_name in ['F', 'B']:
                    x, y, z = offset[0] + (col - 1), offset[1], offset[2] + (1 - row)
                else:
                    x, y, z = offset[0], offset[1] + (col - 1), offset[2] + (1 - row)

                corners = self.create_sticker([x, y, z], normal)
                polygons.append(corners)
                colors.append(self.get_sticker_color(face_idx, sticker_idx))

        return polygons, colors

    def update_display(self, title="Rubik's Cube"):
        """Update the entire display"""
        # Clear both axes
        self.ax3d.clear()
        self.ax_metrics.clear()

        # Draw 3D cube
        faces = [
            (0, 'U', [0, 0, 1.5], [0, 0, 1]),
            (1, 'R', [1.5, 0, 0], [1, 0, 0]),
            (2, 'F', [0, 1.5, 0], [0, 1, 0]),
            (3, 'D', [0, 0, -1.5], [0, 0, -1]),
            (4, 'L', [-1.5, 0, 0], [-1, 0, 0]),
            (5, 'B', [0, -1.5, 0], [0, -1, 0])
        ]

        all_polygons = []
        all_colors = []

        for face_idx, face_name, offset, normal in faces:
            polygons, colors = self.draw_face(face_idx, face_name, offset, normal)
            all_polygons.extend(polygons)
            all_colors.extend(colors)

        poly_collection = Poly3DCollection(
            all_polygons,
            facecolors=all_colors,
            edgecolors='black',
            linewidths=2,
            alpha=0.95
        )

        self.ax3d.add_collection3d(poly_collection)

        limit = 2.5
        self.ax3d.set_xlim([-limit, limit])
        self.ax3d.set_ylim([-limit, limit])
        self.ax3d.set_zlim([-limit, limit])
        self.ax3d.set_xlabel('X', fontsize=10)
        self.ax3d.set_ylabel('Y', fontsize=10)
        self.ax3d.set_zlabel('Z', fontsize=10)
        self.ax3d.set_title(title, fontsize=14, fontweight='bold', pad=20)
        self.ax3d.view_init(elev=25, azim=45)
        self.ax3d.grid(True, alpha=0.2)

        # Draw metrics panel
        self.ax_metrics.axis('off')

        correct = self.cube.count_correct_pieces()
        is_solved = self.cube.is_solved()

        # Title
        title_text = "SOLVED!" if is_solved else "Solving..."
        title_color = 'green' if is_solved else 'blue'
        self.ax_metrics.text(0.5, 0.95, title_text,
                            ha='center', va='top', fontsize=24,
                            fontweight='bold', color=title_color,
                            transform=self.ax_metrics.transAxes)

        # Metrics box
        y_pos = 0.82
        metrics_text = [
            f"Correct Pieces: {correct}/54",
            f"Accuracy: {(correct/54)*100:.1f}%",
            f"Total Moves: {len(self.move_history)}"
        ]

        for text in metrics_text:
            self.ax_metrics.text(0.1, y_pos, text,
                                fontsize=13, family='monospace',
                                transform=self.ax_metrics.transAxes)
            y_pos -= 0.08

        # Recent moves
        if self.move_history:
            y_pos -= 0.05
            self.ax_metrics.text(0.1, y_pos, "Recent Moves:",
                                fontsize=12, fontweight='bold',
                                transform=self.ax_metrics.transAxes)

            recent_moves = self.move_history[-6:]
            moves_text = " ".join(recent_moves)
            y_pos -= 0.06
            self.ax_metrics.text(0.1, y_pos, moves_text,
                                fontsize=11, family='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                                transform=self.ax_metrics.transAxes)

        # Progress graph
        if len(self.correct_history) > 1:
            ax_graph = self.fig.add_axes([0.58, 0.12, 0.38, 0.28])
            ax_graph.clear()
            ax_graph.plot(self.correct_history, 'b-', linewidth=2.5, marker='o', markersize=4)
            ax_graph.axhline(y=54, color='g', linestyle='--', linewidth=2, alpha=0.7, label='Solved')
            ax_graph.fill_between(range(len(self.correct_history)), self.correct_history, alpha=0.3)
            ax_graph.set_xlabel('Move Number', fontsize=11, fontweight='bold')
            ax_graph.set_ylabel('Correct Pieces', fontsize=11, fontweight='bold')
            ax_graph.set_title('Solving Progress', fontsize=12, fontweight='bold')
            ax_graph.legend(loc='lower right')
            ax_graph.grid(True, alpha=0.3, linestyle='--')
            ax_graph.set_ylim([0, 56])
            ax_graph.set_xlim([0, max(len(self.correct_history)-1, 1)])

        # Force draw and flush events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def animate_solution(self, moves, interval=0.4):
        """Animate solving moves with smooth updates"""
        print(f"\nAnimating solution ({len(moves)} moves)...")

        for i, move in enumerate(moves):
            print(f"  Move {i+1}/{len(moves)}: {move}")

            # Apply move
            self.cube.apply_move(move)
            self.move_history.append(move)
            self.correct_history.append(self.cube.count_correct_pieces())

            # Update display
            self.update_display(f"Move {i+1}/{len(moves)}: {move}")

            # Pause for animation
            plt.pause(interval)

        # Final state
        if self.cube.is_solved():
            self.update_display("SOLVED!")
            print("\nCube SOLVED!")
        else:
            self.update_display(f"Final State ({self.cube.count_correct_pieces()}/54 correct)")

        plt.pause(1.5)


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def visualize_policy_iteration(scramble_moves=6, training_states=300, animation_speed=0.4):
    """
    Main visualization function

    Args:
        scramble_moves: Number of random moves to scramble (1-15 recommended)
        training_states: Number of states for training (200-500 recommended)
        animation_speed: Seconds between animation frames (0.3-1.0)
    """
    print("=" * 80)
    print("RUBIK'S CUBE - POLICY ITERATION VISUALIZATION")
    print("=" * 80)

    # Create cube and visualizer
    cube = RubiksCube()
    viz = CubeVisualizer(cube)

    # Show solved state
    print("\n[1] Initial Solved State")
    viz.update_display("Initial Solved State")
    plt.pause(1.5)

    # Scramble
    print(f"\n[2] Scrambling with {scramble_moves} random moves...")
    actions = ["R", "R'", "R2", "L", "L'", "L2",
               "U", "U'", "U2", "D", "D'", "D2",
               "F", "F'", "F2", "B", "B'", "B2"]

    scramble_sequence = []
    for i in range(scramble_moves):
        move = random.choice(actions)
        scramble_sequence.append(move)
        cube.apply_move(move)
        viz.move_history.append(move)

    initial_correct = cube.count_correct_pieces()
    viz.correct_history.append(initial_correct)
    print(f"  Scramble: {' '.join(scramble_sequence)}")
    print(f"  Initial correct pieces: {initial_correct}/54")

    viz.update_display(f"Scrambled ({initial_correct}/54 correct)")
    plt.pause(2.0)

    # Train solver
    print(f"\n[3] Training Policy Iteration Solver ({training_states} states)...")
    start_time = time.time()
    solver = PolicyIterationSolver(cube)
    solver.train(num_states=training_states, max_iterations=20)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")

    # Solve
    print("\n[4] Solving the cube...")
    start_time = time.time()
    solution = solver.solve(max_moves=100)
    solve_time = time.time() - start_time

    if solution:
        print(f"  Solution found: {len(solution)} moves")
        print(f"  Solution: {' '.join(solution)}")
        print(f"  Solve time: {solve_time:.2f}s")

        # Animate solution
        print("\n[5] Animating solution...")
        viz.animate_solution(solution, interval=animation_speed)

        print("\n" + "=" * 80)
        print("SUCCESS! Cube solved!")
        print("=" * 80)
    else:
        final_correct = cube.count_correct_pieces()
        print(f"  Could not fully solve")
        print(f"  Final state: {final_correct}/54 correct")
        viz.update_display(f"Final State ({final_correct}/54 correct)")
        plt.pause(2.0)

    print("\nClose the window to exit...")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # Run visualization with customizable parameters
    # Adjust these values to your preference:

    visualize_policy_iteration(
        scramble_moves=6,        # Difficulty: 3-6 easy, 7-10 medium, 11-15 hard
        training_states=300,     # More states = better learning (200-500 recommended)
        animation_speed=0.4      # Seconds between moves (0.3=fast, 0.8=slow)
    )
