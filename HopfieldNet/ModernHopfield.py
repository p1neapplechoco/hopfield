import numpy as np
from typing import Optional
from scipy.special import softmax

def lse(beta: float, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return 0.0
    if beta <= 0:
         raise ValueError("beta must be positive for LSE.")

    scaled_x = beta * x
    max_val = np.max(scaled_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_x - max_val)))
    
    return (1.0 / beta) * log_sum_exp

def sign(value: float) -> int:
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 1

class DiscreteHopfield:
    def __init__(self, beta: float=1.0) -> None:
        self.patterns = None
        self.N = 0
        self.d = 0
        self.beta = beta                       
 
    def store_patterns(self, patterns: np.ndarray) -> None:
        if patterns.ndim != 2:
            raise ValueError("Input 'patterns' must be a 2D NumPy array (N, d).")
        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Pattern values must be only -1 or 1.")

        self.N, self.d = patterns.shape
        self.patterns = patterns.astype(np.int8)

    def energy(self, state: np.ndarray) -> float:
        if self.patterns is None:
            raise RuntimeError("No patterns stored yet. Call store_patterns() first.")
        if state.shape != (self.d,):
             raise ValueError(f"State dimension {state.shape} doesn't match pattern dimension {self.d}.")

        dot_products = self.patterns @ state.astype(np.float64) # can be proved

        log_sum_exp_val = lse(1, dot_products)
        energy = -np.exp(log_sum_exp_val)

        return energy

    def update_component(self, current_state: np.ndarray, l: int) -> int:
        state_float = current_state.astype(np.float64)
        patterns_float = self.patterns.astype(np.float64)

        dot_products_current = patterns_float @ state_float
        x_col_l = patterns_float[:, l]

        term_common = dot_products_current - x_col_l * state_float[l]
        dot_products_l_plus = term_common + x_col_l
        dot_products_l_minus = term_common - x_col_l

        scaled_dots_plus = self.beta * dot_products_l_plus
        scaled_dots_minus = self.beta * dot_products_l_minus

        log_sum_exp_plus = lse(1, scaled_dots_plus)
        log_sum_exp_minus = lse(1, scaled_dots_minus)

        tolerance = 1e-9
        diff = log_sum_exp_plus - log_sum_exp_minus

        if diff > tolerance:
            new_value_l = 1
        elif diff < -tolerance:
            new_value_l = -1
        else:
            new_value_l = int(current_state[l])

        return new_value_l


    def retrieve(self, initial_state: np.ndarray,
                 max_iter: int = 1,
                 update_order: str = 'random',
                 max_unchanged: int = 10) -> np.ndarray:
        
        if self.patterns is None:
            raise RuntimeError("No patterns stored yet. Call store_patterns() first.")
        if initial_state.ndim != 1 or initial_state.shape[0] != self.d:
             raise ValueError(f"Initial state must be 1D with dimension {self.d}.")
        if not np.all(np.isin(initial_state, [-1, 1])):
             raise ValueError("Warning: Initial state contains values other than {-1, 1}.")

        state = initial_state.copy()

        state_history = [state.copy()]
        energy_history = [self.energy(state)]

        consecutive_unchanged = 0
        converged = False

        for it in range(max_iter):
            state_changed_this_sweep = False
            order = np.random.permutation(self.d) if update_order == 'random' else np.arange(self.d)

            for l in order:
                new_val_l = self.update_component(state, l)

                if new_val_l != state[l]:
                    state[l] = new_val_l
                    state_changed_this_sweep = True

            state_history.append(state.copy())
            current_energy = self.energy(state)
            energy_history.append(current_energy)

            if not state_changed_this_sweep:
                consecutive_unchanged += 1
            else:
                consecutive_unchanged = 0

            if consecutive_unchanged >= max_unchanged:
                converged = True
            
            if converged:
                break
            
            print(f'Iteration {it + 1}: State={state}')

        return state

class ContinuousHopfield:
    def __init__(self, beta: float = 1.0) -> None:
        self.patterns = None
        self.d = 0
        self.N = 0
        self.beta = beta             
        self.M = 0.0                 

    def store_patterns(self, patterns: np.ndarray) -> None:
        if not isinstance(patterns, np.ndarray) or patterns.ndim != 2:
            raise ValueError("patterns must be a 2D numpy array with shape (N, d).")

        patterns = patterns.astype(np.float64)

        self.N, self.d = patterns.shape
        self.patterns = patterns

        pattern_norms = np.linalg.norm(self.patterns, axis=1)
        self.M = np.max(pattern_norms) if pattern_norms.size > 0 else 0.0

    def energy(self, state: np.ndarray) -> float:
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if state.shape != (self.d,):
             raise ValueError(f"Input state must be flattened to shape ({self.d},), but got {state.shape}")

        e1 = -lse(self.beta, self.patterns @ state)
        e2 = 0.5 * state.T @ state

        return e1 + e2 + np.log(self.N) / self.beta + self.M ** 2 / 2


    def update_state(self, state: np.ndarray) -> np.ndarray:
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if state.shape != (self.d,):
             raise ValueError(f"Input state must be flattened to shape ({self.d},), but got {state.shape}")

        if self.N == 0:
            return np.zeros_like(state)

        similarity = self.patterns @ state

        activation = softmax(self.beta * similarity, axis=0)

        state_next = self.patterns.T @ activation

        return state_next

    def retrieve(self,
                 initial_state: np.ndarray,
                 max_iter: int = 1,
                 tolerance: Optional[float] = 1e-5,
                ) -> np.ndarray:
        
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if initial_state.shape != (self.d,):
             raise ValueError(f"Initial state must be flattened to shape ({self.d},), but got {initial_state.shape}. Use initial_state.flatten() if needed.")
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative.")

        current_state = initial_state.copy()

        for _ in range(max_iter):
            next_state = self.update_state(current_state)

            if tolerance is not None:
                change_norm = np.linalg.norm(next_state - current_state)
                if change_norm < tolerance:
                    current_state = next_state
                    break

            current_state = next_state

        return current_state