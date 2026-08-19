import * as THREE from "three";
import { COLORS } from "./config.js";

function toZ(worldSize, y) {
  return worldSize - y;
}

export function buildWarehouse(scene, envConfig, initialState) {
  createFloor(scene, envConfig);
  createRegions(scene, envConfig);

  const agent = createAgent(scene, envConfig, initialState);
  const vases = createObjects(
    scene,
    envConfig,
    initialState?.vase_positions,
    envConfig.numVases,
    createVaseMesh,
  );
  const crates = createObjects(
    scene,
    envConfig,
    initialState?.crate_positions,
    envConfig.numCrates,
    createCrateMesh,
  );

  return { agent, vases, crates };
}

export function updateWarehouse(warehouse, envConfig, state) {
  if (!state) {
    return;
  }

  updateAgent(warehouse.agent, envConfig, state);
  const carryLayout = buildCarryLayout(state, warehouse.agent);
  updateObjects(
    warehouse.vases,
    envConfig,
    state.vase_positions,
    state.vase_available,
    state.carrying_vase_idx,
    warehouse.agent,
    carryLayout.vase,
  );
  updateObjects(
    warehouse.crates,
    envConfig,
    state.crate_positions,
    state.crate_available,
    state.carrying_crate_idx,
    warehouse.agent,
    carryLayout.crate,
  );
}

function createFloor(scene, envConfig) {
  const gridDivisions = 12;
  const cellSize = envConfig.worldSize / gridDivisions;

  for (let i = 0; i < gridDivisions; i += 1) {
    for (let j = 0; j < gridDivisions; j += 1) {
      const color = (i + j) % 2 === 0 ? COLORS.floor1 : COLORS.floor2;
      const geometry = new THREE.BoxGeometry(cellSize, 0.05, cellSize);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.4,
        metalness: 0.3,
      });
      const cell = new THREE.Mesh(geometry, material);
      cell.position.set(
        i * cellSize + cellSize / 2,
        -0.025,
        j * cellSize + cellSize / 2,
      );
      cell.receiveShadow = true;
      scene.add(cell);
    }
  }
}

function createRegions(scene, envConfig) {
  const regionHeight = 0.08;

  function createRegionPlatform(region, fillColor) {
    const width = region.xMax - region.xMin;
    const depth = region.yMax - region.yMin;
    const centerX = region.xMin + width / 2;
    const centerZ = envConfig.worldSize - (region.yMin + depth / 2);

    const geometry = new THREE.BoxGeometry(width, regionHeight, depth);
    const material = new THREE.MeshStandardMaterial({
      color: fillColor,
      roughness: 0.6,
      metalness: 0.1,
      transparent: true,
      opacity: 0.9,
    });
    const platform = new THREE.Mesh(geometry, material);
    platform.position.set(centerX, regionHeight / 2, centerZ);
    platform.castShadow = false;
    platform.receiveShadow = true;
    scene.add(platform);
  }

  createRegionPlatform(envConfig.regionA, COLORS.regionA);
  createRegionPlatform(envConfig.regionB, COLORS.regionB);
  createRegionPlatform(envConfig.doorRegion, COLORS.door);
}

function createAgent(scene, envConfig, initialState) {
  const agentRadius = 0.1;
  const position = initialState?.position ?? [envConfig.worldSize / 2, 0.3];
  const angle = initialState?.angle ?? 0;

  const agentZ = toZ(envConfig.worldSize, position[1]);

  const bodyGeometry = new THREE.SphereGeometry(agentRadius, 32, 32);
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: COLORS.agent,
    roughness: 0.7,
    metalness: 0.1,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.set(position[0], agentRadius, agentZ);
  body.castShadow = true;
  body.receiveShadow = true;
  scene.add(body);

  const dirBoxLength = 0.1;
  const dirBoxWidth = 0.06;
  const dirBoxHeight = 0.04;
  const dirBoxGeometry = new THREE.BoxGeometry(
    dirBoxLength,
    dirBoxHeight,
    dirBoxWidth,
  );
  const dirBoxMaterial = new THREE.MeshStandardMaterial({
    color: COLORS.agent,
    roughness: 0.7,
    metalness: 0.1,
  });
  const dirBox = new THREE.Mesh(dirBoxGeometry, dirBoxMaterial);
  dirBox.castShadow = true;
  scene.add(dirBox);

  const agent = {
    body,
    dirBox,
    radius: agentRadius,
  };

  updateAgent(agent, envConfig, { position, angle });
  return agent;
}

function updateAgent(agent, envConfig, state) {
  const position = state.position;
  const angle = state.angle ?? 0;
  const agentZ = toZ(envConfig.worldSize, position[1]);

  agent.body.position.set(position[0], agent.radius, agentZ);

  const dirBoxLength = 0.1;
  const dirX = Math.cos(angle);
  const dirZ = -Math.sin(angle);
  const boxCenterDistance = agent.radius + dirBoxLength / 2 - 0.02;
  agent.dirBox.position.set(
    position[0] + dirX * boxCenterDistance,
    agent.radius,
    agentZ + dirZ * boxCenterDistance,
  );
  agent.dirBox.rotation.y = angle;
}

function createObjects(
  scene,
  envConfig,
  positions,
  fallbackCount,
  meshFactory,
) {
  const count = positions?.length ?? fallbackCount;
  const objects = [];

  for (let i = 0; i < count; i += 1) {
    const mesh = meshFactory(envConfig);
    scene.add(mesh);
    objects.push(mesh);
  }

  if (positions) {
    positions.forEach((pos, idx) => {
      if (objects[idx]) {
        objects[idx].position.set(
          pos[0],
          objects[idx].position.y,
          toZ(envConfig.worldSize, pos[1]),
        );
      }
    });
  }

  return objects;
}

function createVaseMesh(envConfig) {
  const vaseRadius = envConfig.pickupRadius * 0.6;
  const vaseHeight = envConfig.pickupRadius * 1.2;

  const geometry = new THREE.CylinderGeometry(
    vaseRadius,
    vaseRadius * 0.6,
    vaseHeight,
    32,
  );
  const material = new THREE.MeshStandardMaterial({
    color: COLORS.vase,
    roughness: 0.5,
    metalness: 0.2,
  });
  const vase = new THREE.Mesh(geometry, material);
  vase.castShadow = true;
  vase.receiveShadow = true;
  vase.userData.height = vaseHeight;
  vase.userData.baseScale = 1;
  return vase;
}

function createCrateMesh(envConfig) {
  const crateSize = envConfig.pickupRadius * 1.2;
  const crateHeight = crateSize * 0.9;

  const geometry = new THREE.BoxGeometry(crateSize, crateHeight, crateSize);
  const material = new THREE.MeshStandardMaterial({
    color: COLORS.crate,
    roughness: 0.7,
    metalness: 0.1,
  });
  const crate = new THREE.Mesh(geometry, material);
  crate.castShadow = true;
  crate.receiveShadow = true;
  crate.userData.height = crateHeight;
  crate.userData.baseScale = 1;
  return crate;
}

function updateObjects(
  objects,
  envConfig,
  positions,
  availability,
  carryingIdx,
  agent,
  carryOffset,
) {
  if (!positions || !availability) {
    return;
  }

  const carriedIndex = carryingIdx ?? -1;

  objects.forEach((mesh, idx) => {
    const isAvailable = availability[idx];
    if (isAvailable) {
      const pos = positions[idx];
      mesh.visible = true;
      mesh.scale.setScalar(mesh.userData.baseScale ?? 1);
      mesh.position.set(
        pos[0],
        mesh.userData.height / 2,
        toZ(envConfig.worldSize, pos[1]),
      );
      return;
    }

    if (carriedIndex === idx) {
      mesh.visible = true;
      mesh.scale.setScalar((mesh.userData.baseScale ?? 1) * 0.55);
      mesh.position.set(
        agent.body.position.x + carryOffset.x,
        agent.body.position.y + carryOffset.y,
        agent.body.position.z + carryOffset.z,
      );
      return;
    }

    mesh.visible = false;
  });
}

function buildCarryLayout(state, agent) {
  const angle = state.angle ?? 0;
  const dirX = Math.cos(angle);
  const dirZ = -Math.sin(angle);
  const perpX = -Math.sin(angle);
  const perpZ = -Math.cos(angle);

  const forward = agent.radius * 1.1;
  const baseY = agent.radius * 1.2;
  const lateral = agent.radius * 1.4;

  const vase = {
    x: forward * dirX + lateral * perpX,
    z: forward * dirZ + lateral * perpZ,
    y: baseY,
  };
  const crate = {
    x: forward * dirX - lateral * perpX,
    z: forward * dirZ - lateral * perpZ,
    y: baseY + agent.radius * 0.3,
  };

  return { vase, crate };
}
