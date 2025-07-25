import numpy as np
from typing import Tuple, List

class ClassicHopfield:
    def __init__(self) -> None:
        self.W = None
        self.d = 0
        self.patterns = None
    
    def store_patterns(self, patterns: np.ndarray, zero_diag=False) -> None:
        if patterns.ndim != 2:
            raise ValueError
        
        if not np.all(np.isin(patterns, [1, -1])):
            raise ValueError

        self.d = patterns.shape[1]
        self.W = np.zeros((self.d, self.d))

        self.patterns = patterns

        for pattern in patterns:
            if pattern.shape[0] != self.d:
                raise ValueError
            else:
                self.W += np.outer(pattern, pattern)
        
        if zero_diag:
            np.fill_diagonal(self.W, 0)
            print("Zeroed diagonal of W")

    @staticmethod
    def sign(x: np.array, zero_to_positive=True) -> np.array:
        signs = np.atleast_1d(x)
        signs = np.sign(signs)
        
        if zero_to_positive:
            for i in range(len(signs)):
                if signs[i] == 0:
                    signs[i] = 1
        else:
            for i in range(len(signs)):
                if signs[i] == 0:
                    signs[i] = -1
            
        return np.atleast_1d(signs)
    
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def energy(self, state: np.array) -> float:
        if state.shape[0] != self.d:
            raise ValueError
        
        if not np.all(np.isin(state, [1, -1])):
            raise ValueError
        
        energy = -0.5 * state.T @ self.W @ state
        return energy


    def retrieve(self, initial_state: np.ndarray,
                 mode: str = 'async',
                 max_iter: int = 1000,
                 ) -> np.ndarray:

        if initial_state.ndim != 1 or initial_state.shape[0] != self.d:
            raise ValueError(f"Initial state must be a 1D array of shape ({self.d},), but got {initial_state.shape}")
        
        if mode not in ['sync', 'async']:
            raise ValueError("Mode must be 'sync' or 'async'.")
        
        if not np.all(np.isin(initial_state, [-1, 1])):
            raise ValueError("Initial state must contain only -1 and 1 values.")
        
        state = self.sign(initial_state.astype(np.float64)).astype(np.int8)

        state_history = [state.copy()]
        energy_history = [self.energy(state)]

        converged = False
        for iteration in range(max_iter):
            iter_num = iteration + 1
            prev_state = state.copy()

            if mode == 'sync':
                x = self.W @ state
         
                state = self.sign(x)
                state_history.append(state.copy())

                current_energy = self.energy(state)
                energy_history.append(current_energy)

                converged = np.array_equal(state, prev_state)


            elif mode == 'async':
                indices = np.random.permutation(self.d)
                state_changed_in_sweep = False

                for i in indices:
                    x_i = self.W[i] @ state
                    new_val_i = self.sign(x_i)[0]

                    if new_val_i != state[i]:
                        state[i] = new_val_i
                        state_changed_in_sweep = True

                state_history.append(state.copy())
                current_energy = self.energy(state)
                energy_history.append(current_energy)

                if not state_changed_in_sweep:
                    converged = np.isclose(current_energy, energy_history[-2], atol=1e-9)


            delta_e = energy_history[-1] - energy_history[-2]
            print(f"Iteration #{iter_num}: Energy={energy_history[-1]:.4f} (Delta E = {delta_e:.4f}), State={state}")

            if converged:
                break

        return state