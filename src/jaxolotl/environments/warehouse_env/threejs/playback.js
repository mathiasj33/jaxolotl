import { updateWarehouse } from "./warehouse.js";

export function startReplay({
  scene,
  camera,
  renderer,
  controls,
  warehouse,
  trajectories,
  envConfig,
  replayConfig,
  setStatus,
}) {
  const hasTrajectories = trajectories && trajectories.length > 0;
  let episodeIndex = 0;
  let stepIndex = 0;
  let frame = 0;
  let pauseElapsed = 0;
  let lastTimestamp = 0;
  let paused = false;

  if (hasTrajectories) {
    const initialState = trajectories[0].states?.[0];
    updateWarehouse(warehouse, envConfig, initialState);
    if (setStatus) {
      setStatus(episodeIndex, stepIndex, trajectories[0].states?.length ?? 0);
    }
  }

  function animate(timestamp) {
    requestAnimationFrame(animate);
    controls.update();

    if (!lastTimestamp) {
      lastTimestamp = timestamp;
    }

    const delta = (timestamp - lastTimestamp) / 1000;
    lastTimestamp = timestamp;

    if (paused) {
      renderer.render(scene, camera);
      return;
    }

    if (hasTrajectories) {
      const currentTrajectory = trajectories[episodeIndex];
      const states = currentTrajectory.states ?? [];

      if (stepIndex >= states.length) {
        pauseElapsed += delta;
        if (pauseElapsed >= replayConfig.pauseBetweenEpisodes) {
          episodeIndex = (episodeIndex + 1) % trajectories.length;
          stepIndex = 0;
          frame = 0;
          pauseElapsed = 0;
          updateWarehouse(
            warehouse,
            envConfig,
            trajectories[episodeIndex].states?.[0],
          );
          if (setStatus) {
            setStatus(
              episodeIndex,
              stepIndex,
              trajectories[episodeIndex].states?.length ?? 0,
            );
          }
        }
      } else {
        frame += 1;
        if (frame >= replayConfig.framesPerStep) {
          frame = 0;
          stepIndex += 1;
          updateWarehouse(warehouse, envConfig, states[stepIndex]);
          if (setStatus) {
            setStatus(episodeIndex, stepIndex, states.length);
          }
        }
      }
    }

    renderer.render(scene, camera);
  }

  requestAnimationFrame(animate);

  function gotoEpisode(index) {
    if (!hasTrajectories) {
      return;
    }
    const total = trajectories.length;
    episodeIndex = ((index % total) + total) % total;
    stepIndex = 0;
    frame = 0;
    pauseElapsed = 0;
    updateWarehouse(
      warehouse,
      envConfig,
      trajectories[episodeIndex].states?.[0],
    );
    if (setStatus) {
      setStatus(
        episodeIndex,
        stepIndex,
        trajectories[episodeIndex].states?.length ?? 0,
      );
    }
  }

  function togglePause() {
    paused = !paused;
    lastTimestamp = 0;
    return paused;
  }

  function setPaused(nextPaused) {
    paused = Boolean(nextPaused);
    lastTimestamp = 0;
  }

  function nextEpisode() {
    gotoEpisode(episodeIndex + 1);
  }

  function prevEpisode() {
    gotoEpisode(episodeIndex - 1);
  }

  return {
    togglePause,
    setPaused,
    nextEpisode,
    prevEpisode,
    getPaused: () => paused,
  };
}
