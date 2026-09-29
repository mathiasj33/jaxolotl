# Third-party notices

The root [MIT license](LICENSE) applies to original jaxolotl material. The
following material retains its original license and attribution. The shared
[Apache 2.0 text](LICENSES/Apache-2.0.txt) covers the Apache-licensed code
adaptations listed below. The distribution therefore declares both MIT and
Apache 2.0 in its package metadata. The Panda asset directory also contains its own
[license copy](src/jaxolotl/environments/assets/franka_emika_panda/LICENSE).

| Material in this repository | Source and changes | License |
| --- | --- | --- |
| `src/jaxolotl/environments/environment.py`, `spaces.py`, `wrappers/wrapper.py` | Adapted from [Gymnax](https://github.com/RobertTLange/gymnax). The environment, spaces, and wrapper interfaces were changed for jaxolotl. Upstream license notice: Copyright 2020 Rémi Louf. | [Apache 2.0](https://github.com/RobertTLange/gymnax/blob/main/LICENSE) |
| `src/jaxolotl/rl/ppo.py` | Adapted from [PureJaxRL](https://github.com/luchris429/purejaxrl). The PPO implementation was changed for jaxolotl. Upstream license notice: Copyright 2023 Chris Lu. | [Apache 2.0](https://github.com/luchris429/purejaxrl/blob/main/LICENSE) |
| `src/jaxolotl/networks/gru_cell.py` | Adapted from [Equinox's GRU cell](https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_rnn.py), with separate biases for the hidden gates. | [Apache 2.0](https://github.com/patrick-kidger/equinox/blob/main/LICENSE) |
| `src/jaxolotl/environments/franka_zone_env/kinematics.py` | Adapted from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) Panda kinematics, with changes for the grasp-centre frame and jaxolotl's environment. | [Apache 2.0](https://github.com/google-deepmind/mujoco_playground/blob/main/LICENSE) |
| `src/jaxolotl/environments/assets/franka_emika_panda/` | Panda MJCF and meshes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/a03e87b/franka_emika_panda). The gripper actuator was removed from `mjx_panda.xml`; details and provenance are in the [asset README](src/jaxolotl/environments/assets/franka_emika_panda/README.md). | [Apache 2.0](src/jaxolotl/environments/assets/franka_emika_panda/LICENSE) |

The warehouse replay uses [Three.js](https://github.com/mrdoob/three.js)
version 0.182.0 via jsDelivr at runtime. Three.js is not copied into this
repository. It is [MIT licensed](https://github.com/mrdoob/three.js/blob/r182/LICENSE).

The `semml/` directory is a separately maintained [Git submodule](.gitmodules),
currently pinned to commit `18b2d2ef005cfaa8f94f48cd4009904eb4e8b966`.
Its code is under [GPLv3](semml/LICENSE). It also carries third-party binaries,
Java code, and benchmark data; some have their own notices under `semml/`. The root
MIT license and the package metadata do not relicense the submodule.

The pinned submodule does not include adjacent license or build-provenance files
for `thirdparty/syfco/meyerphi-syfco` and `jbdd-0.7.0.jar`. The apparent upstream
projects are [SyFCo](https://github.com/reactive-systems/syfco) (MIT) and
[JBDD](https://github.com/incaseoftrouble/jbdd) (GPLv3), but the exact bundled
binaries have not been matched to upstream releases. Confirm their provenance
in `semml` before distributing an initialized checkout or those binaries.
