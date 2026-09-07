"""Slab geometry definitions and domain validation for 1D reactor models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SlabGeometry:
    """One-dimensional slab geometry representation.

    Parameters
    ----------
    length : float
        Slab thickness/length $L$ along the $x$-axis in centimeters (cm).
        Must be strictly positive.

    Attributes
    ----------
    x_min : float
        Left boundary coordinate (0.0 cm).
    x_max : float
        Right boundary coordinate ($L$ cm).
    """

    length: float

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(f"Slab length must be strictly positive, got {self.length} cm.")

    @property
    def x_min(self) -> float:
        """Left boundary coordinate in centimeters."""
        return 0.0

    @property
    def x_max(self) -> float:
        """Right boundary coordinate in centimeters."""
        return self.length

    def contains(self, x: float) -> bool:
        """Check whether coordinate $x$ lies within $[0, L]$.

        Parameters
        ----------
        x : float
            Coordinate in cm.

        Returns
        -------
        bool
            True if $0 \\le x \\le L$, False otherwise.
        """
        return 0.0 <= x <= self.length
