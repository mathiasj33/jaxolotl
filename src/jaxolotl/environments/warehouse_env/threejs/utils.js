import { DEFAULT_ENV } from "./config.js";

function toRegion(input, fallback) {
  if (!input) {
    return fallback;
  }
  if (Array.isArray(input) && input.length === 4) {
    return {
      xMin: input[0],
      xMax: input[1],
      yMin: input[2],
      yMax: input[3],
    };
  }
  return {
    xMin: input.xMin ?? fallback.xMin,
    xMax: input.xMax ?? fallback.xMax,
    yMin: input.yMin ?? fallback.yMin,
    yMax: input.yMax ?? fallback.yMax,
  };
}

export function parseEmbeddedData() {
  const dataEl = document.getElementById("warehouse-data");
  if (!dataEl) {
    return null;
  }
  const raw = dataEl.textContent?.trim();
  if (!raw || raw === "__WAREHOUSE_DATA__") {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    console.error("Failed to parse embedded replay data", error);
    return null;
  }
}

export function normalizeEnvConfig(data) {
  const env = data?.env ?? {};
  const worldSize = env.worldSize ?? env.world_size ?? DEFAULT_ENV.worldSize;
  const pickupRadius =
    env.pickupRadius ?? env.pickup_radius ?? DEFAULT_ENV.pickupRadius;
  const numVases = env.numVases ?? env.num_vases ?? DEFAULT_ENV.numVases;
  const numCrates = env.numCrates ?? env.num_crates ?? DEFAULT_ENV.numCrates;

  return {
    worldSize,
    pickupRadius,
    numVases,
    numCrates,
    regionA: toRegion(env.regionA ?? env.region_a, DEFAULT_ENV.regionA),
    regionB: toRegion(env.regionB ?? env.region_b, DEFAULT_ENV.regionB),
    doorRegion: toRegion(
      env.doorRegion ?? env.door_region,
      DEFAULT_ENV.doorRegion,
    ),
  };
}

export function normalizeReplayConfig(data) {
  const replay = data?.replay ?? {};
  return {
    framesPerStep: replay.frames_per_step ?? replay.framesPerStep ?? 60,
    pauseBetweenEpisodes:
      replay.pause_between_episodes ?? replay.pauseBetweenEpisodes ?? 0,
  };
}

export function getInitialState(trajectories) {
  if (!trajectories || trajectories.length === 0) {
    return null;
  }
  const states = trajectories[0].states ?? [];
  if (states.length === 0) {
    return null;
  }
  return states[0];
}

export function clampIndex(value, max) {
  if (value < 0) {
    return -1;
  }
  return Math.min(value, max);
}
