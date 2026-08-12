"""
obstacles.py

Static obstacle geometry for each scene, extracted directly from the values
used in scene_warehouse.xml / scene_courtyard.xml / scene_urban.xml.

This is deliberately NOT parsed from the XML at runtime -- we authored those
files ourselves, so the source of truth is duplicated here explicitly. If you
ever edit a scene's clutter layout, this file needs a matching update (a
comment to that effect is left in each scene file's obstacle block below).
Runtime XML parsing would be more DRY but adds a fragile dependency on
MJCF's internal representation for a problem that's simpler solved with a
flat data file.

All obstacles are approximated as circles in the XY plane (bounding circle
around each box/cylinder footprint) for cheap 2D collision checking --
sufficient for planning a GROUND target's path where the target and all
obstacles both sit near z=0. Height is not checked here; see the trajectory
generator's docstring for the ground-only vs. drone-altitude caveat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    radius: float  # bounding circle radius around the actual footprint


@dataclass(frozen=True)
class SceneBounds:
    """Usable flight-area bounds for target trajectory sampling. Kept
    somewhat inside the actual wall/building extents so the target doesn't
    graze scene boundaries.
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def _box_bounding_radius(half_x: float, half_y: float) -> float:
    """Bounding circle radius for a box footprint, from its XY half-extents."""
    return math.hypot(half_x, half_y)


# -- Warehouse -----------------------------------------------------------

WAREHOUSE_BOUNDS = SceneBounds(x_min=-17, x_max=17, y_min=-17, y_max=17)

WAREHOUSE_OBSTACLES: list[CircleObstacle] = [
    # crates (box half-extents from scene_warehouse.xml geom size, XY only)
    CircleObstacle(3.0, 2.0, _box_bounding_radius(0.5, 0.5)),      # crate_01
    CircleObstacle(-2.5, 4.0, _box_bounding_radius(0.4, 0.4)),     # crate_02
    CircleObstacle(5.0, -3.0, _box_bounding_radius(0.6, 0.6)),     # crate_03
    CircleObstacle(-4.0, -2.0, _box_bounding_radius(0.4, 0.4)),    # crate_04
    CircleObstacle(1.0, 6.0, _box_bounding_radius(0.5, 0.9)),      # crate_05
    CircleObstacle(3.2, 2.0, _box_bounding_radius(0.45, 0.45)),    # crate_stack_a
    # pillars (cylinder radius directly)
    CircleObstacle(8.0, 6.0, 0.3),                                  # pillar_01
    CircleObstacle(-7.0, -6.0, 0.3),                                # pillar_02
    # walls: not included as circle obstacles (too large/far to approximate
    # usefully this way) -- SceneBounds already keeps trajectories well
    # inside the 20x20 wall boundary instead.
]


# -- Courtyard -------------------------------------------------------------

COURTYARD_BOUNDS = SceneBounds(x_min=-13, x_max=13, y_min=-13, y_max=13)

COURTYARD_OBSTACLES: list[CircleObstacle] = [
    CircleObstacle(4.0, 3.0, _box_bounding_radius(0.4, 0.4)),      # planter_01
    CircleObstacle(-5.0, -2.0, _box_bounding_radius(0.4, 0.4)),    # planter_02
    CircleObstacle(2.0, -6.0, _box_bounding_radius(0.4, 0.4)),     # planter_03
    CircleObstacle(-3.0, 4.0, _box_bounding_radius(0.9, 0.3)),     # bench_01
    CircleObstacle(15.0, 10.0, 0.5),                                # landmark_tower_01
    CircleObstacle(-14.0, -12.0, 0.5),                              # landmark_tower_02
]


# -- Urban -------------------------------------------------------------------

URBAN_BOUNDS = SceneBounds(x_min=-4.5, x_max=4.5, y_min=-4.5, y_max=4.5)

URBAN_OBSTACLES: list[CircleObstacle] = [
    # buildings (large -- these dominate the usable space, hence the
    # tighter SceneBounds above)
    CircleObstacle(6.0, 5.0, _box_bounding_radius(1.5, 4.0)),       # building_01
    CircleObstacle(6.0, -3.0, _box_bounding_radius(1.5, 3.0)),      # building_02
    CircleObstacle(-6.0, 4.0, _box_bounding_radius(2.0, 3.5)),      # building_03
    CircleObstacle(-6.0, -4.0, _box_bounding_radius(1.2, 2.5)),     # building_04
    # crates
    CircleObstacle(2.0, 2.0, _box_bounding_radius(0.4, 0.4)),       # crate_01
    CircleObstacle(2.8, 1.2, _box_bounding_radius(0.4, 0.4)),       # crate_02
    CircleObstacle(-2.0, 1.5, _box_bounding_radius(0.35, 0.35)),    # crate_03
    CircleObstacle(-1.5, -2.0, _box_bounding_radius(0.5, 0.5)),     # crate_04
    CircleObstacle(1.0, -2.5, _box_bounding_radius(0.4, 0.4)),      # crate_05
    CircleObstacle(3.0, -1.0, _box_bounding_radius(0.45, 0.45)),    # crate_06
    # barriers
    CircleObstacle(0.0, 3.0, _box_bounding_radius(1.2, 0.15)),      # barrier_01
    CircleObstacle(0.0, -3.0, _box_bounding_radius(1.2, 0.15)),     # barrier_02
]


SCENE_OBSTACLES: dict[str, list[CircleObstacle]] = {
    "warehouse": WAREHOUSE_OBSTACLES,
    "courtyard": COURTYARD_OBSTACLES,
    "urban": URBAN_OBSTACLES,
}

SCENE_BOUNDS: dict[str, SceneBounds] = {
    "warehouse": WAREHOUSE_BOUNDS,
    "courtyard": COURTYARD_BOUNDS,
    "urban": URBAN_BOUNDS,
}