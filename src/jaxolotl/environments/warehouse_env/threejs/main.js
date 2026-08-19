import { initScene } from "./scene.js";
import { buildWarehouse } from "./warehouse.js";
import {
  getInitialState,
  normalizeEnvConfig,
  normalizeReplayConfig,
  parseEmbeddedData,
} from "./utils.js";
import { setupUI } from "./ui.js";
import { startReplay } from "./playback.js";

const data = parseEmbeddedData() ?? {};
const envConfig = normalizeEnvConfig(data);
const replayConfig = normalizeReplayConfig(data);
const trajectories = Array.isArray(data.trajectories) ? data.trajectories : [];
const initialState = getInitialState(trajectories);

const { scene, camera, renderer, controls } = initScene(envConfig);
const warehouse = buildWarehouse(scene, envConfig, initialState);
let statusUpdater = null;
const playback = startReplay({
  scene,
  camera,
  renderer,
  controls,
  warehouse,
  trajectories,
  envConfig,
  replayConfig,
  setStatus: (...args) => statusUpdater?.(...args),
});
const ui = setupUI(renderer, scene, camera, playback);
statusUpdater = ui.setStatus;
