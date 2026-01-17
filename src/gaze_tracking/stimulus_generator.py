"""
Stimulus generator for red-dot tracking experiments.

Generates various stimulus trajectories for gaze tracking evaluation.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class StimulusGenerator:
    """
    Generates stimulus trajectories (red dot positions).

    Supports:
    - Linear horizontal movement
    - Circular motion
    - Semi-random Brownian motion
    - Stationary position
    """

    def __init__(
        self,
        screen_width: int = 1280,
        screen_height: int = 720,
        speed: float = 100,  # pixels per second
    ):
        """
        Initialize stimulus generator.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            speed: Movement speed in pixels per second
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.speed = speed
        self.time = 0.0

        # Random state for Brownian motion
        self.random_position = np.array(
            [screen_width / 2.0, screen_height / 2.0], dtype=np.float32
        )
        self.random_velocity = np.zeros(2, dtype=np.float32)

    def get_linear_position(self, t: float) -> Tuple[float, float]:
        """
        Get stimulus position for horizontal linear motion.

        Args:
            t: Time in seconds

        Returns:
            (x, y) position tuple
        """
        # Horizontal oscillation
        x = self.screen_width / 2 + (self.screen_width / 3) * np.sin(2 * np.pi * t / 4)
        y = self.screen_height / 2

        # Clamp to screen
        x = np.clip(x, 0, self.screen_width)
        y = np.clip(y, 0, self.screen_height)

        return (float(x), float(y))

    def get_circular_position(self, t: float) -> Tuple[float, float]:
        """
        Get stimulus position for circular motion.

        Args:
            t: Time in seconds

        Returns:
            (x, y) position tuple
        """
        # Circular trajectory
        radius = min(self.screen_width, self.screen_height) / 4
        cx, cy = self.screen_width / 2, self.screen_height / 2

        angle = 2 * np.pi * t / 5  # Complete circle every 5 seconds

        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)

        # Clamp to screen
        x = np.clip(x, 0, self.screen_width)
        y = np.clip(y, 0, self.screen_height)

        return (float(x), float(y))

    def get_random_position(self, dt: float) -> Tuple[float, float]:
        """
        Get stimulus position for Brownian motion.

        Args:
            dt: Time delta since last call (seconds)

        Returns:
            (x, y) position tuple
        """
        # Brownian motion (random walk)
        acceleration = np.random.normal(0, 50, 2)  # Random acceleration
        self.random_velocity += acceleration * dt
        self.random_velocity *= 0.95  # Damping

        # Update position
        self.random_position += self.random_velocity * dt

        # Bounce off walls
        if self.random_position[0] < 0 or self.random_position[0] > self.screen_width:
            self.random_velocity[0] *= -1

        if self.random_position[1] < 0 or self.random_position[1] > self.screen_height:
            self.random_velocity[1] *= -1

        # Clamp to screen
        x = np.clip(self.random_position[0], 0, self.screen_width)
        y = np.clip(self.random_position[1], 0, self.screen_height)

        return (float(x), float(y))

    def get_static_position(self) -> Tuple[float, float]:
        """
        Get stimulus position (center, stationary).

        Returns:
            (x, y) position tuple (screen center)
        """
        return (self.screen_width / 2, self.screen_height / 2)

    def get_position(
        self,
        mode: str = "linear",
        t: float = 0.0,
        dt: float = 0.033,  # ~30 FPS
    ) -> Tuple[float, float]:
        """
        Get stimulus position for the given mode.

        Args:
            mode: Trajectory mode ("linear", "circular", "random", "static")
            t: Current time in seconds
            dt: Time delta for random motion

        Returns:
            (x, y) position tuple
        """
        if mode == "linear":
            return self.get_linear_position(t)
        elif mode == "circular":
            return self.get_circular_position(t)
        elif mode == "random":
            return self.get_random_position(dt)
        elif mode == "static":
            return self.get_static_position()
        else:
            raise ValueError(f"Unknown stimulus mode: {mode}")

    def reset(self):
        """Reset stimulus generator state."""
        self.time = 0.0
        self.random_position = np.array(
            [self.screen_width / 2.0, self.screen_height / 2.0], dtype=np.float32
        )
        self.random_velocity = np.zeros(2, dtype=np.float32)

    def get_trajectory_info(self, mode: str) -> Dict[str, str]:
        """
        Get description of trajectory mode.

        Args:
            mode: Trajectory mode

        Returns:
            Dictionary with mode description
        """
        descriptions = {
            "linear": "Horizontal linear oscillation",
            "circular": "Smooth circular motion",
            "random": "Semi-random Brownian motion",
            "static": "Stationary center position",
        }

        return {"mode": mode, "description": descriptions.get(mode, "Unknown")}


if __name__ == "__main__":
    # Example usage
    gen = StimulusGenerator(screen_width=1280, screen_height=720)

    print("Testing stimulus trajectories:")
    for mode in ["linear", "circular", "random", "static"]:
        gen.reset()
        print(f"\n{mode.upper()}:")

        for t in [0, 1, 2, 3, 4, 5]:
            pos = gen.get_position(mode=mode, t=t)
            print(f"  t={t}s: {pos}")
