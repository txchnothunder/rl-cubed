"""
policy_iteration_solver.py

Policy Iteration approach for solving Rubik's cube.
Uses value function approximation and policy improvement.
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict, deque
from src.base_cube import BaseCube


class PolicyIterationSolver:
    """
    Policy Iteration solver for Rubik's cube using reinforcement learning.
    """
    
    def __init__(self, cube, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.cube = cube
        self.size = cube.size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        
        if self.size != 3:
            raise ValueError("Only works for 3x3 cubes")
        
        # Action space - basic moves
        self.actions = [
            "R", "R'", "R2",
            "L", "L'", "L2", 
            "U", "U'", "U2",
            "D", "D'", "D2",
            "F", "F'", "F2",
            "B", "B'", "B2"
        ]
        
        # Initialize policy and value function
        self.policy = {}  # state -> action
        self.value_function = defaultdict(float)  # state -> value
        
        # Metrics tracking
        self.metrics = {
            'iterations': 0,
            'policy_changes': 0,
            'solve_time': 0,
            'final_moves': 0,
            'convergence_data': []
        }
    
    def _get_state_key(self):
        """Convert cube state to hashable string."""
        return ''.join(self.cube._cube)
    
    def _apply_move_string(self, move_str):
        """Apply a move like 'R', "R'", 'R2', etc."""
        face = move_str[0]
        modifier = move_str[1:] if len(move_str) > 1 else ''
        
        if modifier == '2':
            degrees = 180
        elif modifier == "'":
            degrees = 270
        else:
            degrees = 90
        
        if face == 'R':
            self.cube.turn_right(3, degrees)
        elif face == 'L':
            self.cube.turn_right(1, 360 - degrees if degrees != 180 else 180)
        elif face == 'U':
            self.cube.turn_clockwise(1, degrees)
        elif face == 'D':
            self.cube.turn_clockwise(3, 360 - degrees if degrees != 180 else 180)
        elif face == 'F':
            self.cube.turn_upward(3, 360 - degrees if degrees != 180 else 180)
        elif face == 'B':
            self.cube.turn_upward(1, degrees)
    
    def _is_solved(self):
        """Check if cube is solved."""
        step = 9
        for face_idx in range(6):
            start = face_idx * step
            face_color = self.cube._cube[start]
            for i in range(step):
                if self.cube._cube[start + i] != face_color:
                    return False
        return True
    
    def _count_solved_pieces(self):
        """Count how many pieces are in correct position."""
        solved = BaseCube(3)
        correct = 0
        for i in range(len(self.cube._cube)):
            if self.cube._cube[i] == solved._cube[i]:
                correct += 1
        return correct
    
    def _get_reward(self, state_key, action, next_state_key):
        """Calculate reward for state-action-next_state transition."""
        # Save current state
        current_cube_state = self.cube._cube
        
        # Temporarily set to next state to calculate reward
        self.cube._cube = list(next_state_key)
        
        if self._is_solved():
            reward = 100.0  # Large positive reward for solving
        else:
            solved_pieces = self._count_solved_pieces()
            # Reward based on number of solved pieces
            reward = solved_pieces / 54.0 * 10 - 1  # -1 for each move, up to +10 for progress
        
        # Restore current state
        self.cube._cube = current_cube_state
        return reward
    
    def _get_next_state(self, action):
        """Get next state after applying action."""
        # Save current state
        cube_backup = self.cube._cube
        
        # Apply action
        self._apply_move_string(action)
        next_state = self._get_state_key()
        
        # Restore state
        self.cube._cube = cube_backup
        
        return next_state
    
    def _policy_evaluation(self, max_iterations=100, threshold=1e-6):
        """Evaluate current policy using iterative policy evaluation."""
        for iteration in range(max_iterations):
            delta = 0
            old_values = dict(self.value_function)
            
            for state_key in list(self.policy.keys()):
                if state_key not in self.value_function:
                    continue
                    
                # Set cube to this state
                self.cube._cube = list(state_key)
                
                if self._is_solved():
                    self.value_function[state_key] = 0  # Terminal state
                    continue
                
                action = self.policy[state_key]
                next_state = self._get_next_state(action)
                reward = self._get_reward(state_key, action, next_state)
                
                new_value = reward + self.discount * self.value_function[next_state]
                old_value = self.value_function[state_key]
                self.value_function[state_key] = new_value
                
                delta = max(delta, abs(new_value - old_value))
            
            if delta < threshold:
                break
        
        return iteration + 1
    
    def _policy_improvement(self):
        """Improve policy by being greedy with respect to value function."""
        policy_stable = True
        
        for state_key in list(self.policy.keys()):
            # Set cube to this state
            self.cube._cube = list(state_key)
            
            if self._is_solved():
                continue
            
            old_action = self.policy[state_key]
            
            # Find best action
            best_value = float('-inf')
            best_action = old_action
            
            for action in self.actions:
                next_state = self._get_next_state(action)
                reward = self._get_reward(state_key, action, next_state)
                value = reward + self.discount * self.value_function[next_state]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state_key] = best_action
            
            if old_action != best_action:
                policy_stable = False
                self.metrics['policy_changes'] += 1
        
        return policy_stable
    
    def _generate_training_states(self, num_states=1000):
        """Generate states for training by random scrambling."""
        states = set()
        
        for _ in range(num_states):
            # Start with solved cube
            self.cube = BaseCube(3)
            
            # Apply random moves
            num_moves = random.randint(1, 15)
            for _ in range(num_moves):
                action = random.choice(self.actions)
                self._apply_move_string(action)
            
            state_key = self._get_state_key()
            states.add(state_key)
            
            # Initialize policy and value for this state
            if state_key not in self.policy:
                self.policy[state_key] = random.choice(self.actions)
                self.value_function[state_key] = 0.0
        
        return states
    
    def train(self, num_training_states=500, max_iterations=50, verbose=True):
        """Train the policy using policy iteration."""
        if verbose:
            print("=" * 70)
            print("POLICY ITERATION SOLVER")
            print("=" * 70)
            print(f"Training on {num_training_states} states with {len(self.actions)} actions")
            print("-" * 70)
        
        start_time = time.time()
        
        # Generate training states
        if verbose:
            print("Generating training states...")
        training_states = self._generate_training_states(num_training_states)
        
        if verbose:
            print(f"Generated {len(training_states)} unique states")
            print("Starting policy iteration...")
        
        # Policy iteration
        for iteration in range(max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}:")
            
            # Policy evaluation
            eval_iterations = self._policy_evaluation()
            if verbose:
                print(f"  Policy evaluation converged in {eval_iterations} iterations")
            
            # Policy improvement
            policy_stable = self._policy_improvement()
            if verbose:
                print(f"  Policy changes: {self.metrics['policy_changes']}")
            
            # Track convergence
            avg_value = np.mean(list(self.value_function.values()))
            self.metrics['convergence_data'].append({
                'iteration': iteration + 1,
                'avg_value': avg_value,
                'policy_changes': self.metrics['policy_changes'],
                'num_states': len(self.policy)
            })
            
            if policy_stable:
                if verbose:
                    print(f"  Policy converged after {iteration + 1} iterations!")
                break
        
        self.metrics['iterations'] = iteration + 1
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f}s")
            print(f"Final policy has {len(self.policy)} states")
            print("-" * 70)
    
    def solve(self, max_moves=100, verbose=True):
        """Solve the cube using the trained policy."""
        if verbose:
            print("Solving cube using trained policy...")
        
        start_time = time.time()
        solution = []
        
        for move_count in range(max_moves):
            if self._is_solved():
                break
            
            state_key = self._get_state_key()
            
            # Use policy if we have it, otherwise random action (exploration)
            if state_key in self.policy:
                action = self.policy[state_key]
            else:
                # Epsilon-greedy exploration for unseen states
                if random.random() < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    # Choose action that maximizes immediate reward
                    best_action = None
                    best_reward = float('-inf')
                    
                    for a in self.actions:
                        next_state = self._get_next_state(a)
                        reward = self._get_reward(state_key, a, next_state)
                        if reward > best_reward:
                            best_reward = reward
                            best_action = a
                    
                    action = best_action if best_action else random.choice(self.actions)
            
            # Apply the action
            self._apply_move_string(action)
            solution.append(action)
            
            if verbose and move_count % 10 == 9:
                solved_pieces = self._count_solved_pieces()
                print(f"  Move {move_count + 1}: {solved_pieces}/54 pieces correct")
        
        solve_time = time.time() - start_time
        self.metrics['solve_time'] = solve_time
        self.metrics['final_moves'] = len(solution)
        
        if self._is_solved():
            if verbose:
                print(f"SOLVED in {len(solution)} moves! ({solve_time:.2f}s)")
            return solution
        else:
            if verbose:
                final_score = self._count_solved_pieces()
                print(f"Not fully solved ({final_score}/54 correct) after {len(solution)} moves")
            return None


def run_policy_iteration_experiments(num_experiments):
    """Run policy iteration experiments and collect performance metrics."""
    print("=" * 80)
    print("POLICY ITERATION EXPERIMENTS")
    print("=" * 80)
    
    results = []
    
    for experiment in range(num_experiments):
        print(f"\n{'='*20} EXPERIMENT {experiment + 1} {'='*20}")
        
        # Create scrambled cube
        cube = BaseCube(3)
        solver = PolicyIterationSolver(cube)
        
        # Scramble the cube
        num_scramble_moves = random.randint(5, 12)
        scramble_moves = []
        for _ in range(num_scramble_moves):
            move = random.choice(solver.actions)
            solver._apply_move_string(move)
            scramble_moves.append(move)
        
        print(f"Scramble ({num_scramble_moves} moves): {' '.join(scramble_moves)}")
        initial_score = solver._count_solved_pieces()
        print(f"Initial score: {initial_score}/54 pieces correct")
        
        # Train the policy
        solver.train(num_training_states=300, max_iterations=20, verbose=False)
        
        # Try to solve
        solution = solver.solve(max_moves=50, verbose=False)
        
        # Record results
        result = {
            'experiment': experiment + 1,
            'scramble_moves': num_scramble_moves,
            'initial_score': initial_score,
            'solved': solver._is_solved(),
            'final_score': solver._count_solved_pieces(),
            'solution_moves': len(solution) if solution else 0,
            'solve_time': solver.metrics['solve_time'],
            'training_iterations': solver.metrics['iterations'],
            'policy_changes': solver.metrics['policy_changes'],
            'convergence_data': solver.metrics['convergence_data']
        }
        
        results.append(result)
        
        # Print experiment results
        if result['solved']:
            print(f"SOLVED in {result['solution_moves']} moves")
        else:
            print(f"Not solved ({result['final_score']}/54 correct)")
        
        print(f"Training: {result['training_iterations']} iterations, {result['policy_changes']} policy changes")
        print(f"Solve time: {result['solve_time']:.3f}s")
    
    return results


def create_performance_visualizations(results):
    """Create visualizations of the policy iteration performance."""
    print("\n" + "="*80)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Policy Iteration Solver Performance', fontsize=16)
    
    # Extract data
    experiments = [r['experiment'] for r in results]
    solved = [r['solved'] for r in results]
    solution_moves = [r['solution_moves'] if r['solved'] else 0 for r in results]
    solve_times = [r['solve_time'] for r in results]
    training_iterations = [r['training_iterations'] for r in results]
    initial_scores = [r['initial_score'] for r in results]
    final_scores = [r['final_score'] for r in results]
    
    # 1. Success Rate
    success_rate = sum(solved) / len(solved) * 100
    axes[0, 0].bar(['Failed', 'Solved'], [len(solved) - sum(solved), sum(solved)], 
                   color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_title(f'Success Rate: {success_rate:.1f}%')
    axes[0, 0].set_ylabel('Number of Experiments')
    
    # 2. Solution Length Distribution
    if any(solved):
        solved_moves = [m for m, s in zip(solution_moves, solved) if s]
        axes[0, 1].hist(solved_moves, bins=5, alpha=0.7, color='blue')
        axes[0, 1].set_title('Solution Length Distribution')
        axes[0, 1].set_xlabel('Number of Moves')
        axes[0, 1].set_ylabel('Frequency')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Solutions Found', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Solution Length Distribution')
    
    # 3. Solve Time vs Problem Difficulty
    axes[0, 2].scatter(initial_scores, solve_times, c=['green' if s else 'red' for s in solved], alpha=0.7)
    axes[0, 2].set_xlabel('Initial Score (pieces correct)')
    axes[0, 2].set_ylabel('Solve Time (seconds)')
    axes[0, 2].set_title('Solve Time vs Initial Difficulty')
    
    # 4. Training Convergence (first experiment)
    if results[0]['convergence_data']:
        conv_data = results[0]['convergence_data']
        iterations = [d['iteration'] for d in conv_data]
        avg_values = [d['avg_value'] for d in conv_data]
        axes[1, 0].plot(iterations, avg_values, marker='o')
        axes[1, 0].set_xlabel('Training Iteration')
        axes[1, 0].set_ylabel('Average Value Function')
        axes[1, 0].set_title('Training Convergence (Exp 1)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Initial vs Final Scores
    axes[1, 1].scatter(initial_scores, final_scores, c=['green' if s else 'red' for s in solved], alpha=0.7)
    axes[1, 1].plot([0, 54], [0, 54], 'k--', alpha=0.5)  # diagonal line
    axes[1, 1].set_xlabel('Initial Score')
    axes[1, 1].set_ylabel('Final Score')
    axes[1, 1].set_title('Score Improvement')
    
    # 6. Training Efficiency
    axes[1, 2].scatter(training_iterations, solve_times, c=['green' if s else 'red' for s in solved], alpha=0.7)
    axes[1, 2].set_xlabel('Training Iterations')
    axes[1, 2].set_ylabel('Solve Time (seconds)')
    axes[1, 2].set_title('Training vs Solve Time')
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\Owner\\Downloads\\policy_iteration_performance.png',
            dpi=300, bbox_inches='tight')
    print("Performance visualization saved as 'policy_iteration_performance.png'")
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 50)
    print(f"Experiments run: {len(results)}")
    print(f"Success rate: {success_rate:.1f}%")
    if any(solved):
        avg_moves = np.mean([m for m, s in zip(solution_moves, solved) if s])
        print(f"Average solution length: {avg_moves:.1f} moves")
    print(f"Average solve time: {np.mean(solve_times):.3f}s")
    print(f"Average training iterations: {np.mean(training_iterations):.1f}")
    
    return fig


if __name__ == "__main__":
    # Run experiments
    results = run_policy_iteration_experiments(250)
    
    # Create visualizations
    create_performance_visualizations(results)
    
    print("\n" + "="*80)
    print("POLICY ITERATION EXPERIMENT COMPLETE!")
    print("="*80)
