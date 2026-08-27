# Franka Emika Panda assets

These MJCF files and meshes back `FrankaZoneEnv`, the MJX-based Panda reaching
environment. `mjx_scene.xml` is the entry point: it includes `mjx_panda.xml`
and adds the floor. Load it through
`jaxolotl.environments.assets.asset_path(PANDA_SCENE)` so MuJoCo can resolve
the mesh directory reliably.

## Provenance and licence

`mjx_panda.xml`, `mjx_scene.xml`, and `assets/` come from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
revision `a03e87b`, under `franka_emika_panda/`. They are available under the
Apache 2.0 licence; see [LICENSE](LICENSE).

## Project-specific model choice

The Menagerie MJX model uses a deliberately small collision set. Its visual arm
meshes do not collide, while the hand capsule and fingertip pads retain the
contacts needed by the source scene. This keeps MJX stepping inexpensive. Zone
targets in `FrankaZoneEnv` are virtual spheres, so the task does not require
object or self-contact simulation.

The model retains the two visible finger joints, held open at reset, but this
project removes the eighth (gripper) actuator from `mjx_panda.xml`. The
environment therefore exposes only the seven Panda arm actuators and has a
plain six-dimensional Cartesian action space.

Keep upstream meshes and licence information intact when updating these files.
The analytical Panda kinematics used by the environment live in
`jaxolotl.environments.franka_zone_env.kinematics`, rather than in this asset
directory.
