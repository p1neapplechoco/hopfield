import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
import os
import matplotlib.pyplot as plt

def stable_softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Computes the softmax function numerically stably.

    Args:
        x (np.ndarray): Input array.
        axis (Optional[int]): Axis along which softmax is computed. Default is None (flattened).

    Returns:
        np.ndarray: Softmax output array, shape like x.
    """
    # Subtract max for numerical stability (prevents exp overflow)
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)

    # Handle potential division by zero if sum_exp_x is zero
    # (e.g., if all inputs were -inf after shift)
    softmax_output = np.divide(exp_x, sum_exp_x, out=np.zeros_like(exp_x), where=sum_exp_x != 0)
    return softmax_output

def stable_lse(beta: float, x: np.ndarray) -> float:
    """
    Computes the Log-Sum-Exp function numerically stably:
    beta^{-1} * log(sum(exp(beta * x))).

    Args:
        beta (float): Inverse temperature.
        x (np.ndarray): Input vector (typically similarities, shape (N,)).

    Returns:
        float: The Log-Sum-Exp scalar value. Returns 0 if x is empty.
    """
    if x.size == 0:
        return 0.0
    if beta <= 0:
         raise ValueError("beta must be positive for LSE.")

    # Add max trick for numerical stability inside exp
    scaled_x = beta * x
    max_val = np.max(scaled_x)
    # Use np.longdouble temporarily for intermediate sum for better precision if needed
    # Although np.exp already uses float64 which is often sufficient
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_x - max_val)))
    return (1.0 / beta) * log_sum_exp

class ContinuousHopfield:
    """
    Implements a Continuous Hopfield Network (Modern/Dense Associative Memory).

    Stores patterns (as rows, N x d) and retrieves them from noisy query states (flattened, d,).
    Designed with numerical stability for potentially high-dimensional data like images.
    Based on Ramsauer et al. arXiv:2008.02217v3.
    """
    def __init__(self, beta: float = 1.0) -> None:
        """
        Initializes the Hopfield network parameters. Patterns are stored later.

        Args:
            beta (float): The inverse temperature parameter (scaling factor).
                          Controls the sharpness of the retrieval. Must be positive.
                          Defaults to 1.0. May need tuning based on data scale/dimension.
        """
        if not isinstance(beta, (int, float)) or beta <= 0:
             raise ValueError("beta must be a positive number.")

        self.patterns: Optional[np.ndarray] = None # Stored patterns, shape (N, d)
        self.d: int = 0                     # Dimension of the patterns
        self.N: int = 0                     # Number of patterns
        self.beta: float = beta             # Scaling factor

        # Attributes calculated after storing patterns
        self.M: float = 0.0                 # Max L2 norm of stored patterns
        self._log_N_beta: float = 0.0       # Precalculated term for energy
        self._M_squared_half: float = 0.0   # Precalculated term for energy

    def store_patterns(self, patterns: np.ndarray) -> None:
        """
        Stores the patterns in the network's memory.

        Args:
            patterns (np.ndarray): The data matrix containing patterns as rows.
                                   Shape must be (N, d), where N is the number
                                   of patterns and d is the dimension.
                                   Data type should be float (e.g., float32 or float64).
                                   Consider normalizing patterns (e.g., to [-1, 1] or unit norm)
                                   before storing for potentially better performance.
        """
        if not isinstance(patterns, np.ndarray) or patterns.ndim != 2:
            raise ValueError("patterns must be a 2D numpy array with shape (N, d).")
        if not np.issubdtype(patterns.dtype, np.floating):
             # Warn if not float, as calculations assume float
             print("Warning: patterns dtype is not float. Converting to float64.")
             patterns = patterns.astype(np.float64)

        self.N, self.d = patterns.shape
        self.patterns = patterns # Store as (N, d)

        if self.N > 0:
            # Calculate M = max(||x_i||) using the (N, d) representation
            pattern_norms = np.linalg.norm(self.patterns, axis=1) # Norm of each row
            self.M = np.max(pattern_norms) if pattern_norms.size > 0 else 0.0

            # Pre-calculate energy terms
            self._log_N_beta = (1.0 / self.beta) * np.log(self.N)
            self._M_squared_half = 0.5 * (self.M ** 2)
        else:
            self.M = 0.0
            self._log_N_beta = 0.0
            self._M_squared_half = 0.0

        print(f"Stored {self.N} patterns of dimension {self.d}. Max norm M = {self.M:.4f}")


    def energy(self, state: np.ndarray) -> float:
        """
        Computes the energy function E for a given state pattern.
        Requires state to be a flattened vector of shape (d,).

        E = -lse(beta, patterns @ state) + 0.5 * state.T @ state + beta^{-1} * log(N) + 0.5 * M^2

        Args:
            state (np.ndarray): The current state pattern vector, shape (d,).

        Returns:
            float: The scalar energy value.

        Raises:
            ValueError: If patterns have not been stored or state shape is incorrect.
        """
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if state.shape != (self.d,):
             raise ValueError(f"Input state must be flattened to shape ({self.d},), but got {state.shape}")

        if self.N == 0:
            # Energy is just self-interaction term if no patterns stored
            return 0.5 * state.T @ state

        # Similarity: Dot product of each pattern (row) with state
        similarity = self.patterns @ state # Shape (N, d) @ (d,) -> (N,)

        e1 = -stable_lse(self.beta, similarity)
        e2 = 0.5 * state.T @ state # Use np.dot(state, state) for potential minor speedup?

        return e1 + e2 + self._log_N_beta + self._M_squared_half

    # total_energy seems redundant/incomplete based on the full energy function. Removed.
    # def total_energy(self, state: np.array):
    #     E1 = (1 / 2) * (state.T @ state)
    #     E2 =  lse(self.beta, self.patterns @ state) # Sign is opposite of energy function


    def update_state(self, state: np.ndarray) -> np.ndarray:
        """
        Applies one step of the Hopfield network update rule (single state update).
        Requires state to be a flattened vector of shape (d,).

        Update rule derived from original paper for patterns as columns (d, N):
          xi^{t+1} = X @ softmax(beta * X.T @ xi)
        Adapted for patterns stored as rows (N, d) self.patterns:
          similarity = self.patterns @ state             (N,)
          activation = softmax(beta * similarity)       (N,)
          state_next = self.patterns.T @ activation     (d,)

        Args:
            state (np.ndarray): The current state pattern vector, shape (d,).

        Returns:
            np.ndarray: The next state pattern vector state^{t+1}, shape (d,).

        Raises:
            ValueError: If patterns have not been stored or state shape is incorrect.
        """
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if state.shape != (self.d,):
             raise ValueError(f"Input state must be flattened to shape ({self.d},), but got {state.shape}")

        if self.N == 0:
            return np.zeros_like(state) # Return zero vector if no patterns

        # 1. Calculate similarity between state and each stored pattern
        similarity = self.patterns @ state # Shape (N, d) @ (d,) -> (N,)

        # 2. Apply scaled softmax to get activation/weights for each pattern
        activation = stable_softmax(self.beta * similarity, axis=0) # Softmax over N patterns

        # 3. Calculate next state as weighted sum of stored patterns (transposed)
        state_next = self.patterns.T @ activation # Shape (d, N) @ (N,) -> (d,)

        return state_next

    def retrieve(self,
                 initial_state: np.ndarray,
                 max_iter: int = 100,
                 tolerance: Optional[float] = 1e-5,
                 store_history: bool = False
                ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Retrieves a stored pattern by iterating the update rule from an initial state.
        Requires the initial_state to be a flattened vector of shape (d,).

        Args:
            initial_state (np.ndarray): The starting state vector, shape (d,).
                                        Flatten image data before passing.
            max_iter (int): Maximum number of update iterations. Defaults to 100.
            tolerance (Optional[float]): If set, stop iteration if the L2 norm of the
                                         change in state is less than this value.
                                         Set to None to disable early stopping. Defaults to 1e-5.
            store_history (bool): If True, return the history of states. Defaults to False.

        Returns:
            Tuple[np.ndarray, Optional[List[np.ndarray]]]:
                - The final retrieved state vector, shape (d,).
                - A list containing the state history [state0, state1, ..., state_final]
                  if store_history is True, otherwise None.

        Raises:
            ValueError: If patterns have not been stored or initial_state shape is incorrect.
        """
        if self.patterns is None:
            raise ValueError("Patterns have not been stored yet. Call store_patterns() first.")
        if initial_state.shape != (self.d,):
             raise ValueError(f"Initial state must be flattened to shape ({self.d},), but got {initial_state.shape}. Use initial_state.flatten() if needed.")
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative.")

        current_state = initial_state.copy()
        history = [current_state.copy()] if store_history else None

        for _ in range(max_iter):
            next_state = self.update_state(current_state)

            # Check for convergence
            if tolerance is not None:
                change_norm = np.linalg.norm(next_state - current_state)
                if change_norm < tolerance:
                    # print(f"Converged after {it + 1} iterations.")
                    current_state = next_state # Ensure the last state is stored
                    if store_history:
                        history.append(current_state.copy())
                    # Stop iterating
                    break

            current_state = next_state
            if store_history:
                history.append(current_state.copy())

        return current_state, history

    def __repr__(self) -> str:
        status = "Initialized" if self.patterns is None else f"Stored N={self.N}, d={self.d}"
        return f"ContinuousHopfield({status}, beta={self.beta:.2f})"
    

def load_images(d: str, shape=(28, 28)) -> np.ndarray:
    # load and convert to grayscale
    images = []
    for filename in os.listdir(d):
        img = Image.open(os.path.join(d, filename)).convert("L")
        img = img.resize(shape)  # Resize to shape
        img_array = np.array(img).flatten()  # Flatten the image to a vector
        images.append(img_array)

    images = np.array(images, dtype=np.float32)  # Convert to numpy array
    images = (images - 127.5) / 127.5  # Normalize to [-1, 1]
    return images
     
def save_processed_images(images, path):
    for i, img in enumerate(images):
        img = (img + 1) * 127.5  # Rescale to [0, 255]
        img = img.astype(np.uint8)  # Convert to uint8
        img = Image.fromarray(img.reshape(140, 90))  # Reshape to original dimensions
        img.save(os.path.join(path, f"processed_{i}.png"))
    

def plot_images(images, n_cols=6, shape=(28, 28)):
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(shape), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_image(image, shape=(28, 28)):
    plt.imshow(image.reshape(shape), cmap='gray')
    plt.axis('off')
    plt.show()

def noise_image(image, noise_factor=0.5):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, -1.0, 1.0)  # Ensure values are in [-1, 1]
    return noisy_image

def set_element_random(image, fraction=0.5):
    # set from portion of the image to random from the first element
    num_elements = image.size
    num_to_set = int(num_elements * fraction)
    random_values = np.random.choice(image, num_to_set, replace=False)
    image[:num_to_set] = random_values
    return image

def grayen_half_image(image):
    half = image.shape[0] // 2
    image[half:] = 0
    return image.copy()

def main():
    path = "D:\.projects\hopfield\imgs\continuous"
    shape = (128, 128)

    images = load_images(path, shape)  # Load images and flatten

    hfnet = ContinuousHopfield(beta=0.000001)  # Initialize the network with beta=0.5
    hfnet.store_patterns(images)  # Store patterns in the network

    img_to_retrieve = images[np.random.randint(0, len(images))].copy()  # Randomly select an image
    plot_image(img_to_retrieve, shape=(shape[1], shape[0]))

    img_to_retrieve = grayen_half_image(img_to_retrieve)  # Add noise to the first image
    plot_image(img_to_retrieve, shape=(shape[1], shape[0]))

    retrieved = hfnet.retrieve(img_to_retrieve, max_iter=1)[0]
    
    plot_image(retrieved, shape=(shape[1], shape[0]))

main()