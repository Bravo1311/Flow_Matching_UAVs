"""
scene_loader.py

Per-episode scene selection and domain randomization for the Phase 1
following-policy MuJoCo environment.

Responsibilities:
  1. Pick one of the 3 scene XMLs (warehouse / courtyard / urban) at episode
     reset, either randomly or via a caller-specified distribution.
  2. Load the composed MJCF (scene + robot + target includes) into a fresh
     mujoco.MjModel / MjData pair.
  3. Apply per-episode domain randomization on top of the loaded model:
     light position/direction/intensity jitter, and material color/reflectance
     jitter -- without needing to rewrite the XML on disk each episode.

Design note: steps 1-2 pick *which* scene; step 3 randomizes *within* that
scene's already-defined lights/materials at the compiled-model level (via
model.light_pos, model.mat_rgba, etc.), which is far cheaper than re-parsing
XML per episode and is the standard MuJoCo pattern for this.

This intentionally does NOT yet implement variable follow-distance
(Phase 1.1) or mode-specific reset logic -- scope is scene selection +
visual randomization only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SCENES_DIR = Path(__file__).parent / "scenes"

SCENE_FILES = {
    "warehouse": SCENES_DIR / "scene_warehouse.xml",
    "courtyard": SCENES_DIR / "scene_courtyard.xml",
    "urban": SCENES_DIR / "scene_urban.xml",
}

# Default sampling distribution across scenes. Urban is down-weighted since
# it's the hardest scene and meant to be introduced once modes are validated
# on warehouse/courtyard (per the Phase 1 plan) -- override via
# SceneLoaderConfig.scene_weights once you're ready to include it fully.
DEFAULT_SCENE_WEIGHTS = {
    "warehouse": 0.5,
    "courtyard": 0.5,
    "urban": 0.0,
}


@dataclass
class RandomizationRanges:
    """Jitter ranges applied per episode. Tune these once you see real
    render output -- these are reasonable starting points, not tuned values.
    """

    light_pos_jitter: float = 1.5          # meters, uniform +/- per axis
    light_diffuse_jitter: float = 0.15     # fraction, multiplicative jitter
    light_dir_jitter_deg: float = 15.0     # degrees, jitter on light direction

    mat_rgba_jitter: float = 0.08          # fraction, per-channel additive jitter
    mat_reflectance_jitter: float = 0.03   # absolute jitter on reflectance


@dataclass
class SceneLoaderConfig:
    scene_weights: dict = field(default_factory=lambda: dict(DEFAULT_SCENE_WEIGHTS))
    randomize: bool = True
    randomization: RandomizationRanges = field(default_factory=RandomizationRanges)
    seed: int | None = None


class SceneLoader:
    """Picks a scene per episode, composes the full MJCF (scene + robot +
    target), compiles it, and applies domain randomization on the compiled
    model. Caches compiled models per scene name so repeated resets on the
    same scene don't re-parse XML from scratch.
    """

    def __init__(
        self,
        robot_xml: str | Path,
        target_xml: str | Path,
        config: SceneLoaderConfig | None = None,
    ):
        self.robot_xml = Path(robot_xml)
        self.target_xml = Path(target_xml)
        self.config = config or SceneLoaderConfig()
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.default_rng(self.config.seed)

        self._validate_paths()
        self._model_cache: dict[str, mujoco.MjModel] = {}

    def _validate_paths(self) -> None:
        missing = [name for name, p in SCENE_FILES.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing scene XML(s): {missing}. Expected under {SCENES_DIR}. "
                "Did you drop scene_warehouse.xml / scene_courtyard.xml / "
                "scene_urban.xml into sim/scenes/?"
            )
        for label, p in [("robot_xml", self.robot_xml), ("target_xml", self.target_xml)]:
            if not p.exists():
                raise FileNotFoundError(f"{label} not found at {p}")

    def _compose_mjcf(self, scene_name: str) -> str:
        """Builds the top-level MJCF string that includes scene + robot +
        target, matching the include pattern documented in each scene file.
        """
        scene_path = SCENE_FILES[scene_name]
        return f"""<mujoco model="following_env_{scene_name}">
  <include file="{scene_path.as_posix()}"/>
  <include file="{self.robot_xml.as_posix()}"/>
  <include file="{self.target_xml.as_posix()}"/>
</mujoco>
"""

    def _get_or_compile(self, scene_name: str) -> mujoco.MjModel:
        if scene_name not in self._model_cache:
            mjcf_str = self._compose_mjcf(scene_name)
            # x2.xml has <compiler assetdir="assets"/>, resolved relative to
            # wherever the final merged file lives (not x2.xml's own
            # directory, since <include> splices text before path resolution
            # happens). from_xml_string has no "location" at all, so this
            # would always fail to find x2.xml's mesh/texture assets.
            # Writing to a temp file inside mujoco_menagerie/skydio_x2/ makes
            # that relative path resolve correctly. Our own scene/robot/
            # target <include> paths are absolute, so this doesn't affect them.
            project_root = SCENES_DIR.parent.parent
            menagerie_x2_dir = project_root / "mujoco_menagerie" / "skydio_x2"
            if not menagerie_x2_dir.exists():
                raise FileNotFoundError(
                    f"Expected mujoco_menagerie/skydio_x2/ at {menagerie_x2_dir}"
                )

            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", dir=menagerie_x2_dir, delete=False
            ) as f:
                f.write(mjcf_str)
                temp_path = f.name
            try:
                self._model_cache[scene_name] = mujoco.MjModel.from_xml_path(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        # Return a fresh copy so per-episode randomization never mutates the
        # cached base model.
        return mujoco.MjModel(self._model_cache[scene_name])

    def sample_scene_name(self) -> str:
        names = list(self.config.scene_weights.keys())
        weights = list(self.config.scene_weights.values())
        return self._rng.choices(names, weights=weights, k=1)[0]

    def load_episode(self, scene_name: str | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
        """Main entry point: call at every episode reset.

        Returns (model, data, scene_name) so the caller can log which scene
        was used for this episode (useful for later analyzing per-scene
        policy performance).
        """
        scene_name = scene_name or self.sample_scene_name()
        model = self._get_or_compile(scene_name)

        if self.config.randomize:
            self._randomize_lights(model)
            self._randomize_materials(model)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, scene_name

    # -- randomization internals ------------------------------------------------

    def _randomize_lights(self, model: mujoco.MjModel) -> None:
        r = self.config.randomization
        for i in range(model.nlight):
            jitter = self._np_rng.uniform(-r.light_pos_jitter, r.light_pos_jitter, size=3)
            model.light_pos[i] = model.light_pos[i] + jitter

            diffuse_scale = 1.0 + self._np_rng.uniform(
                -r.light_diffuse_jitter, r.light_diffuse_jitter
            )
            model.light_diffuse[i] = np.clip(model.light_diffuse[i] * diffuse_scale, 0.0, 1.0)

            dir_jitter_rad = np.deg2rad(
                self._np_rng.uniform(-r.light_dir_jitter_deg, r.light_dir_jitter_deg, size=3)
            )
            d = model.light_dir[i].copy()
            # small-angle rotation approximation per-axis, then renormalize --
            # fine for the modest jitter ranges here; swap for a proper
            # axis-angle rotation if larger jitter is ever needed.
            d = d + np.cross(d, dir_jitter_rad)
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                model.light_dir[i] = d / norm

    def _randomize_materials(self, model: mujoco.MjModel) -> None:
        r = self.config.randomization
        for i in range(model.nmat):
            rgba = model.mat_rgba[i].copy()
            channel_jitter = self._np_rng.uniform(-r.mat_rgba_jitter, r.mat_rgba_jitter, size=3)
            rgba[:3] = np.clip(rgba[:3] + channel_jitter, 0.0, 1.0)
            model.mat_rgba[i] = rgba

            model.mat_reflectance[i] = float(
                np.clip(
                    model.mat_reflectance[i]
                    + self._np_rng.uniform(-r.mat_reflectance_jitter, r.mat_reflectance_jitter),
                    0.0,
                    1.0,
                )
            )


if __name__ == "__main__":
    # Smoke test: run directly once robot_xml / target_xml exist, e.g.:
    #   python scene_loader.py
    # SCENES_DIR = flow_matching_following/sim/scenes.
    # skydio_x2.xml wrapper lives at project root; target_body.xml now
    # lives in sim/assets/ (moved there by user).
    robot_xml = SCENES_DIR.parent / "assets" / "skydio_x2.xml"
    target_xml = SCENES_DIR.parent / "assets" / "target_body.xml"

    loader = SceneLoader(robot_xml, target_xml)
    for _ in range(5):
        model, data, scene_name = loader.load_episode()
        print(f"Loaded episode scene={scene_name}  nlight={model.nlight}  nmat={model.nmat}")