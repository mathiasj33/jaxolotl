export function setupUI(renderer, scene, camera, playback) {
  const infoEl = document.getElementById("info-status");
  const screenshotBtn = document.getElementById("screenshot-btn");
  const prevBtn = document.getElementById("prev-episode-btn");
  const toggleBtn = document.getElementById("toggle-play-btn");
  const nextBtn = document.getElementById("next-episode-btn");

  if (screenshotBtn) {
    screenshotBtn.addEventListener("click", () =>
      saveScreenshot(renderer, scene, camera),
    );
  }

  if (prevBtn && playback) {
    prevBtn.addEventListener("click", () => playback.prevEpisode());
  }

  if (nextBtn && playback) {
    nextBtn.addEventListener("click", () => playback.nextEpisode());
  }

  if (toggleBtn && playback) {
    toggleBtn.addEventListener("click", () =>
      updateToggleLabel(toggleBtn, playback.togglePause()),
    );
    updateToggleLabel(toggleBtn, playback.getPaused());
  }

  if (playback) {
    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (toggleBtn) {
          updateToggleLabel(toggleBtn, playback.togglePause());
        } else {
          playback.togglePause();
        }
        return;
      }
      if (event.code === "ArrowRight") {
        playback.nextEpisode();
      }
      if (event.code === "ArrowLeft") {
        playback.prevEpisode();
      }
    });
  }

  return {
    setStatus: (episodeIndex, stepIndex, totalSteps) => {
      if (!infoEl) {
        return;
      }
      infoEl.textContent = `Episode ${episodeIndex + 1} | Step ${stepIndex + 1} / ${totalSteps}`;
    },
  };
}

function updateToggleLabel(button, isPaused) {
  const paused = Boolean(isPaused);
  button.textContent = paused ? "▶︎" : "⏸︎";
}

function saveScreenshot(renderer, scene, camera) {
  renderer.render(scene, camera);
  const link = document.createElement("a");
  link.download = "warehouse_environment.png";
  link.href = renderer.domElement.toDataURL("image/png");
  link.click();
}
