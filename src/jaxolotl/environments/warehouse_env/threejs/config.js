export const COLORS = {
  floor1: 0x2d4a66,
  floor2: 0x4a6b8a,
  floorEdge: 0x4a6b8a,
  regionA: 0xcc3333,
  regionABorder: 0xaa2222,
  regionB: 0x33aa33,
  regionBBorder: 0x228822,
  door: 0x8833aa,
  doorBorder: 0x662288,
  agent: 0xd9944d,
  agentDark: 0xb37840,
  vase: 0xe6b84d,
  vaseDark: 0xcc9933,
  crate: 0xa66633,
  crateDark: 0x804d26,
  ambient: 0xffffff,
  background: 0xffffff,
};

export const DEFAULT_ENV = {
  worldSize: 6.6,
  pickupRadius: 0.2,
  numVases: 4,
  numCrates: 4,
  regionA: { xMin: 0, xMax: 2.2, yMin: 1.5, yMax: 3.7 },
  regionB: { xMin: 3.7, xMax: 6.6, yMin: 0, yMax: 1.5 },
  doorRegion: { xMin: 5, xMax: 6.6, yMin: 5.8, yMax: 6.6 },
};
