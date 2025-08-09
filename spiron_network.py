"""
Spiron Network – conceptual extension of Spincore
=================================================

This module introduces a simple implementation of a **Spiron** network.
In contrast to a single Spincore neuron, a Spiron network models a web
of two‑dimensional spinning points (nodes) anchored around a central
point called the *spironite*.  Each node alternates between two states
(0 and 1) at a fixed time interval and contributes to an aggregated
activation vector.  The network can be used as a source of control
signals for downstream models such as large language models (LLMs).

Key concepts
------------

* **SpironNode** – Represents a point in 2D space that toggles between
  two discrete spin states.  The node maintains its position, a
  current spin value (0 or 1), and a timestamp marking its last
  state transition.  The state flips every ``interval`` seconds.

* **SpironNetwork** – Manages a collection of ``SpironNode`` objects
  arranged around the centre (spironite).  The network exposes
  ``update`` to refresh the spins based on elapsed time and
  ``get_output`` to assemble a vector of node states.  Optionally,
  each node's contribution can be weighted by its orientation with
  respect to the centre.

The design emphasises clarity over biological fidelity.  It is a
lightweight simulation intended to serve as a source of deterministic
yet dynamic features when interfacing with other models.

Example usage:

>>> from spiron_network import SpironNetwork
>>> network = SpironNetwork(num_nodes=4, radius=1.0, interval=0.05)
>>> for _ in range(10):
...     network.update()
...     print(network.get_output())

Author: OpenAI Assistant
Date: 2025‑08‑09 (Asia/Kolkata timezone)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SpironNode:
    """A node in the Spiron network.

    Each node occupies a fixed position in 2D space relative to the
    network's centre.  It toggles between a binary spin state (0 or 1)
    at a fixed interval.  The state is updated lazily when ``update`` is
    called; the caller is responsible for invoking ``update`` at
    reasonable intervals.

    Attributes:
        position: 2D coordinates of the node relative to the centre.
        interval: Time in seconds between state transitions.
        spin_state: Current binary spin (0 or 1).  The state flips
            between calls to ``update`` when the elapsed time exceeds
            ``interval``.
        last_flip: Timestamp (in seconds since epoch) when the spin
            state last toggled.
    """

    position: np.ndarray
    interval: float
    spin_state: int = 0
    last_flip: float = 0.0

    def __post_init__(self) -> None:
        # ensure position is a 2D vector
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape != (2,):
            raise ValueError("position must be a 2D vector")
        # initialise last_flip to current time
        self.last_flip = time.perf_counter()

    def update(self) -> None:
        """Update the spin state based on elapsed time.

        If more than ``interval`` seconds have passed since the last
        transition, the spin state is toggled (0 becomes 1 and vice
        versa), and ``last_flip`` is updated to the current time.
        """
        now = time.perf_counter()
        elapsed = now - self.last_flip
        # handle multiple intervals: flip once per interval elapsed
        if elapsed >= self.interval:
            # number of flips is floor(elapsed / interval) but parity is
            # what matters.  Toggle the spin state accordingly.
            flips = int(elapsed // self.interval)
            if flips % 2 == 1:
                self.spin_state = 1 - self.spin_state
            # update last_flip to the time of the most recent boundary
            self.last_flip += flips * self.interval


class SpironNetwork:
    """A network of spinning nodes anchored around a central point.

    The network initialises a specified number of nodes evenly spaced on
    a circle of given ``radius`` around the origin.  Each node's spin
    state is updated via ``update``.  The network can produce an
    aggregated vector of spins via ``get_output``.  Optionally the
    contributions of each node may be weighted by the orientation of its
    position relative to the centre.

    Args:
        num_nodes: Number of nodes in the network; also the length of
            the output vector.
        radius: Radius of the circle on which nodes are placed.
        interval: Time in seconds between spin state flips for each node.
        weighted: Whether to weight each node's spin by the sign of its
            x‑coordinate (a simple orientation proxy).  If True,
            contributions in the left half‑plane are negated.
    """

    def __init__(self, num_nodes: int, radius: float = 1.0, interval: float = 0.025, weighted: bool = False) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer")
        self.num_nodes = int(num_nodes)
        self.radius = float(radius)
        self.interval = float(interval)
        self.weighted = bool(weighted)
        # compute equally spaced angles
        angles = np.linspace(0.0, 2.0 * math.pi, num=self.num_nodes, endpoint=False)
        self.nodes: List[SpironNode] = []
        for angle in angles:
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            node = SpironNode(position=np.array([x, y]), interval=self.interval, spin_state=0)
            self.nodes.append(node)

    def update(self) -> None:
        """Update all nodes' spin states based on elapsed time."""
        for node in self.nodes:
            node.update()

    def get_output(self, as_float: bool = False) -> np.ndarray:
        """Return the current network activation as a 2D array.

        The returned array has shape ``(1, num_nodes)``.  Each entry is
        the current spin state of the corresponding node.  If
        ``weighted`` was set at construction, spins in the left half
        (nodes with negative x‑coordinate) are multiplied by −1.  If
        ``as_float`` is True, the values are cast to float64; otherwise
        integers are retained.
        """
        values = []
        for node in self.nodes:
            val = node.spin_state
            if self.weighted:
                # simple orientation weighting: sign of x coordinate
                if node.position[0] < 0:
                    val = -val
            values.append(val)
        arr = np.array(values, dtype=np.float64 if as_float else np.int64)
        return arr.reshape(1, -1)

    def __len__(self) -> int:
        return self.num_nodes


__all__ = ["SpironNode", "SpironNetwork"]