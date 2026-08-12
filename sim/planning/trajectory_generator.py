"""
trajectory_generator.py
 
Generates a collision-free 2D ground-target trajectory for one episode, using
random waypoint sampling + rejection against the scene's known obstacles.
 
SCOPE NOTE (important): this checks collisions for the GROUND TARGET's path
only, treating it as a point/circle moving at z~0 against the obstacles in
obstacles.py. It does NOT check the drone's flight path at altitude. Per the
Phase 1 design (trajectory-replay following: the drone's horizontal position
at time t = target's position at time t-delta, altitude-shifted), the drone
inherits the ground path's clearance guarantee UNDER THE ASSUMPTION that
obstacles don't get narrower with height -- true for the boxy buildings/
crates/pillars in these 3 scenes, since they're modeled as vertical prisms/
cylinders with constant XY footprint. If a scene ever gets obstacles that
narrow or widen with height (e.g. the "overhang" elements in scene_urban.xml
-- see its comments), this assumption breaks and a separate altitude-aware
check would be needed. Flagging this now rather than silently assuming it
away.
 
Algorithm, first-cut (deliberately simple, per "aim small, miss small"):
  1. Sample random waypoints inside the scene's usable bounds, rejecting any
     point that falls within (obstacle.radius + target_radius + margin) of
     an obstacle center.
  2. For each consecutive waypoint pair, check the straight-line segment
     between them against every obstacle (point-to-segment distance vs.
     obstacle radius + target_radius + margin). If a segment collides,
     resample that waypoint (bounded retry count) rather than the whole path.
  3. Time-parameterize the resulting piecewise-linear path at a constant
     target speed (arc-length based), producing evenly time-spaced (x, y)
     samples -- directly usable as the target-velocity feedforward signal
     for the PD following teacher designed earlier.
 
Piecewise-linear waypoint connection is collision-checked directly (straight
segments are exactly what got checked). SMOOTHING (added after the initial
version): a Catmull-Rom spline is fit through the same waypoints for a
visually smooth, natural-looking path -- but a spline can bulge outward
between waypoints and re-enter an obstacle's margin even when the waypoints
and straight segments were clear. To handle this honestly rather than
ignore it: waypoints are sampled with an EXTRA smoothing buffer beyond the
normal clearance (so there's room for the curve to bulge without touching an
obstacle), and the resulting dense spline path is then independently
re-checked against the original (non-buffered) clearance. If that check
ever fails, generation retries with fresh waypoints (bounded attempts)
rather than silently shipping a path that clips an obstacle.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim.planning.obstacles import CircleObstacle, SceneBounds, SCENE_BOUNDS, SCENE_OBSTACLES

# Target plate is 0.15 x 0.15 (half-extents) per target_body.xml -> bounding
# radius ~0.21. Add a safety margin on top for the drone's own footprint
# following above/near it.
TARGET_BOUNDING_RADIUS = 0.21
DEFAULT_SAFETY_MARGIN = 0.8  # extra clearance beyond obstacle + target radius
DEFAULT_SMOOTHING_BUFFER = 0.25  # extra margin during waypoint sampling, to
                                   # leave room for spline bulge
 
 
@dataclass
class TrajectoryConfig:
    num_waypoints: int = 6
    target_speed: float = 0.8          # m/s, constant-speed assumption
    dt: float = 0.1                     # time-parameterization sample interval
    safety_margin: float = DEFAULT_SAFETY_MARGIN
    max_resample_attempts: int = 200
    seed: int | None = None
    smooth: bool = True
    smoothing_buffer: float = DEFAULT_SMOOTHING_BUFFER
    samples_per_segment: int = 20       # spline density between waypoints
    max_smoothing_retries: int = 10     # full-path regeneration attempts if
                                          # the smoothed path fails recheck
 
 
def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
 
 
def _point_in_bounds(p: tuple[float, float], bounds: SceneBounds) -> bool:
    return bounds.x_min <= p[0] <= bounds.x_max and bounds.y_min <= p[1] <= bounds.y_max
 
 
def _point_collides(
    p: tuple[float, float], obstacles: list[CircleObstacle], clearance: float
) -> bool:
    for obs in obstacles:
        if _dist(p, (obs.x, obs.y)) < obs.radius + clearance:
            return True
    return False
 
 
def _point_to_segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Shortest distance from point p to the segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p
 
    abx, aby = bx - ax, by - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-9:
        return _dist(p, a)
 
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * abx, ay + t * aby)
    return _dist(p, closest)
 
 
def _segment_collides(
    a: tuple[float, float],
    b: tuple[float, float],
    obstacles: list[CircleObstacle],
    clearance: float,
) -> bool:
    for obs in obstacles:
        if _point_to_segment_distance((obs.x, obs.y), a, b) < obs.radius + clearance:
            return True
    return False
 
 
def _catmull_rom_spline(
    waypoints: list[tuple[float, float]], samples_per_segment: int
) -> list[tuple[float, float]]:
    """Fits a uniform Catmull-Rom spline through the waypoints (passes
    through every waypoint exactly, C1-continuous), returning a dense list
    of (x, y) points for smooth playback/rendering.
 
    Endpoint tangents are handled by duplicating the first/last waypoint
    (standard "clamped" Catmull-Rom trick) so the curve doesn't require
    points outside the given waypoint list.
    """
    if len(waypoints) < 2:
        return list(waypoints)
 
    pts = [waypoints[0]] + list(waypoints) + [waypoints[-1]]
    dense: list[tuple[float, float]] = []
 
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            dense.append((x, y))
 
    dense.append(waypoints[-1])
    return dense
 
 
def _path_collides(
    path: list[tuple[float, float]], obstacles: list[CircleObstacle], clearance: float
) -> bool:
    for i in range(len(path) - 1):
        if _segment_collides(path[i], path[i + 1], obstacles, clearance):
            return True
    return False
 
 
 
def generate_waypoints(
    scene_name: str, config: TrajectoryConfig | None = None
) -> list[tuple[float, float]]:
    """Samples a collision-free sequence of waypoints for the given scene.
 
    Raises RuntimeError if a valid waypoint can't be found within
    max_resample_attempts -- this is a real signal the scene is too
    cluttered for the requested num_waypoints/margin combination, not
    something to silently paper over.
    """
    config = config or TrajectoryConfig()
    rng = random.Random(config.seed)
 
    if scene_name not in SCENE_BOUNDS:
        raise ValueError(f"Unknown scene '{scene_name}'. Expected one of {list(SCENE_BOUNDS)}")
 
    bounds = SCENE_BOUNDS[scene_name]
    obstacles = SCENE_OBSTACLES[scene_name]
    base_clearance = TARGET_BOUNDING_RADIUS + config.safety_margin
    # When smoothing, sample waypoints with extra buffer so the spline has
    # room to bulge outward without touching an obstacle's real clearance.
    clearance = base_clearance + (config.smoothing_buffer if config.smooth else 0.0)
 
    waypoints: list[tuple[float, float]] = []
 
    def _sample_valid_point() -> tuple[float, float]:
        for _ in range(config.max_resample_attempts):
            x = rng.uniform(bounds.x_min, bounds.x_max)
            y = rng.uniform(bounds.y_min, bounds.y_max)
            if not _point_collides((x, y), obstacles, clearance):
                return (x, y)
        raise RuntimeError(
            f"Could not sample a collision-free point in scene '{scene_name}' "
            f"after {config.max_resample_attempts} attempts. Scene may be too "
            f"cluttered for the current safety_margin={config.safety_margin}."
        )
 
    # First waypoint: no segment to check yet.
    waypoints.append(_sample_valid_point())
 
    while len(waypoints) < config.num_waypoints:
        prev = waypoints[-1]
        found = False
        for _ in range(config.max_resample_attempts):
            candidate = _sample_valid_point()
            if not _segment_collides(prev, candidate, obstacles, clearance):
                waypoints.append(candidate)
                found = True
                break
        if not found:
            raise RuntimeError(
                f"Could not find a collision-free segment from waypoint "
                f"{prev} in scene '{scene_name}' after "
                f"{config.max_resample_attempts} attempts. Consider reducing "
                f"num_waypoints, safety_margin, or increasing max_resample_attempts."
            )
 
    return waypoints
 
 
def time_parameterize(
    path: list[tuple[float, float]], config: TrajectoryConfig | None = None
) -> list[tuple[float, float, float, float, float]]:
    """Converts a piecewise-linear path (dense spline samples or raw
    waypoints) into evenly time-spaced (x, y, t, vx, vy) samples at constant
    speed magnitude -- (x, y, t) is the target's scripted position/time
    signal for episode playback; (vx, vy) is the ground-truth velocity
    VECTOR at that instant, which is the actual feedforward term the PD
    following teacher needs (speed alone isn't enough -- direction changes
    continuously along a curved/smoothed path, only the magnitude is held
    constant by construction here).
 
    vx, vy are computed by finite difference between consecutive position
    samples (which are evenly spaced in time by config.dt), not analytically
    from the spline -- simpler, and accurate enough at the dt used here.
    The last sample reuses the second-to-last sample's velocity since there's
    no forward difference available at the final point.
    """
    config = config or TrajectoryConfig()
 
    cum_dist = [0.0]
    for i in range(1, len(path)):
        cum_dist.append(cum_dist[-1] + _dist(path[i - 1], path[i]))
    total_dist = cum_dist[-1]
 
    if total_dist < 1e-9:
        return [(path[0][0], path[0][1], 0.0, 0.0, 0.0)]
 
    total_time = total_dist / config.target_speed
    num_samples = max(2, int(total_time / config.dt) + 1)
 
    positions: list[tuple[float, float, float]] = []
    seg_idx = 0
    for i in range(num_samples):
        t = i * config.dt
        t = min(t, total_time)
        target_dist = t * config.target_speed
 
        while seg_idx < len(cum_dist) - 2 and cum_dist[seg_idx + 1] < target_dist:
            seg_idx += 1
 
        seg_start_dist = cum_dist[seg_idx]
        seg_end_dist = cum_dist[seg_idx + 1]
        seg_len = seg_end_dist - seg_start_dist
        frac = 0.0 if seg_len < 1e-9 else (target_dist - seg_start_dist) / seg_len
        frac = max(0.0, min(1.0, frac))
 
        p0 = path[seg_idx]
        p1 = path[seg_idx + 1]
        x = p0[0] + frac * (p1[0] - p0[0])
        y = p0[1] + frac * (p1[1] - p0[1])
        positions.append((x, y, t))
 
    samples: list[tuple[float, float, float, float, float]] = []
    for i in range(len(positions)):
        x, y, t = positions[i]
        if i < len(positions) - 1:
            nx, ny, nt = positions[i + 1]
            dt_actual = nt - t
            vx = (nx - x) / dt_actual if dt_actual > 1e-9 else 0.0
            vy = (ny - y) / dt_actual if dt_actual > 1e-9 else 0.0
        else:
            # reuse previous velocity at the final sample (no forward diff available)
            vx, vy = (samples[-1][3], samples[-1][4]) if samples else (0.0, 0.0)
        samples.append((x, y, t, vx, vy))
 
    return samples
 
 
def generate_episode_trajectory(
    scene_name: str, config: TrajectoryConfig | None = None
) -> list[tuple[float, float, float, float, float]]:
    """Convenience entry point: waypoints -> (optionally smoothed +
    recollision-checked) dense path -> time-parameterized (x, y, t) samples.
    This is what episode collection scripts should actually call.
 
    If smoothing is enabled and the smoothed path fails the post-smoothing
    collision recheck, regenerates fresh waypoints and retries (bounded by
    max_smoothing_retries) rather than silently returning an unsafe path.
    Falls back to the safe piecewise-linear path (never smoothed-but-unsafe)
    if all retries are exhausted, with a printed warning.
    """
    config = config or TrajectoryConfig()
    base_clearance = TARGET_BOUNDING_RADIUS + config.safety_margin
    obstacles = SCENE_OBSTACLES[scene_name]
 
    for attempt in range(config.max_smoothing_retries if config.smooth else 1):
        attempt_config = config
        if attempt > 0:
            # Vary the seed on retry so we actually get different waypoints,
            # not the same rejected path again.
            retry_seed = None if config.seed is None else config.seed + attempt
            attempt_config = TrajectoryConfig(**{**config.__dict__, "seed": retry_seed})
 
        waypoints = generate_waypoints(scene_name, attempt_config)
 
        if not config.smooth:
            return time_parameterize(waypoints, config)
 
        dense_path = _catmull_rom_spline(waypoints, config.samples_per_segment)
 
        if not _path_collides(dense_path, obstacles, base_clearance):
            return time_parameterize(dense_path, config)
 
        print(
            f"[trajectory_generator] smoothed path attempt {attempt + 1} clipped "
            f"an obstacle margin in scene '{scene_name}' -- retrying with fresh waypoints."
        )
 
    print(
        f"[trajectory_generator] all {config.max_smoothing_retries} smoothing "
        f"attempts failed collision recheck for scene '{scene_name}' -- "
        f"falling back to the safe piecewise-linear (unsmoothed) path."
    )
    waypoints = generate_waypoints(scene_name, config)
    return time_parameterize(waypoints, config)
 
 
if __name__ == "__main__":
    # Smoke test across all 3 scenes.
    for scene in SCENE_BOUNDS:
        cfg = TrajectoryConfig(seed=42)
        traj = generate_episode_trajectory(scene, cfg)
        print(f"{scene}: {len(traj)} samples, "
              f"start={traj[0][:2]}, end={traj[-1][:2]}, "
              f"duration={traj[-1][2]:.2f}s")