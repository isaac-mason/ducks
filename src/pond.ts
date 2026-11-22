import { type Vec3, vec3 } from 'math';
import { mulberry32, random } from 'math/random';
import { flatMesh, ID_FLOOR, ID_LILY, ID_ROCK, type MeshData } from './mesh-data';

export type Rock = { x: number; z: number; radius: number; height: number; mesh: MeshData };
export type LilyPad = { x: number; z: number; radius: number; mesh: MeshData };

export type Pond = {
    floor: MeshData;
    rocks: Rock[];
    pads: LilyPad[];
};

export const PAD_THICKNESS = 0.05;

const SIZE = 90;
const SEGMENTS = 140;

const SAND = [0.93, 0.8, 0.55];
const MOSS = [0.48, 0.66, 0.32];
const STONE = [0.58, 0.6, 0.66];
const PAD_GREEN = [0.38, 0.68, 0.3];
const PAD_UNDERSIDE = [0.62, 0.32, 0.42];
const PETAL_PINK = [1, 0.6, 0.78];

const ROCKS: [x: number, z: number, radius: number, height: number][] = [
    [-6.5, -2.5, 1.3, 1.6],
    [-5.4, -3.8, 0.8, 1.3],
    [-7.4, -1.2, 0.6, 1.1],
    [7, 3.5, 1.5, 1.7],
    [5.8, 4.6, 0.7, 1.2],
    [3, -6, 1.0, 1.4],
    [-2.5, 7, 1.1, 1.5],
    [-1.4, 7.8, 0.6, 1.1],
    [11, -4, 1.6, 1.8],
    [-12, 5, 1.8, 1.8],
];

const PADS: [x: number, z: number, radius: number, flower: boolean][] = [
    [-4.2, 0.2, 0.7, true],
    [-3.2, 1.3, 0.5, false],
    [-5, 1.4, 0.55, false],
    [5, -1.5, 0.75, false],
    [4.2, -2.6, 0.5, true],
    [1.2, 4.5, 0.6, false],
    [-8, 3.2, 0.65, false],
    [8.5, -1, 0.55, true],
    [0.5, -4, 0.5, false],
];

function floorHeight(x: number, z: number): number {
    let h = -1.6 + 0.18 * Math.sin(x * 0.45 + 1.3) * Math.cos(z * 0.38) + 0.08 * Math.sin(x * 1.3 - z * 1.1);
    for (const [rx, rz, r] of ROCKS) {
        const d2 = (x - rx) ** 2 + (z - rz) ** 2;
        h += 0.7 * Math.exp(-d2 / (r * r * 1.6));
    }
    return h;
}

function createFloor(): MeshData {
    const n = SEGMENTS + 1;
    const positions = new Float32Array(n * n * 3);
    const normals = new Float32Array(n * n * 3);
    const colors = new Float32Array(n * n * 4);
    const e = 0.05;
    const normal: Vec3 = [0, 0, 0];
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const v = j * n + i;
            const x = (i / SEGMENTS - 0.5) * SIZE;
            const z = (j / SEGMENTS - 0.5) * SIZE;
            const y = floorHeight(x, z);
            positions.set([x, y, z], v * 3);
            const dx = (floorHeight(x + e, z) - floorHeight(x - e, z)) / (2 * e);
            const dz = (floorHeight(x, z + e) - floorHeight(x, z - e)) / (2 * e);
            normals.set(vec3.normalize(normal, vec3.set(normal, -dx, 1, -dz)), v * 3);
            colors.set([...SAND, ID_FLOOR / 255], v * 4);
        }
    }
    const indices = new Uint32Array(SEGMENTS * SEGMENTS * 6);
    let k = 0;
    for (let j = 0; j < SEGMENTS; j++) {
        for (let i = 0; i < SEGMENTS; i++) {
            const a = j * n + i;
            const b = a + 1;
            const c = a + n;
            const d = c + 1;
            indices.set([a, c, b, b, c, d], k);
            k += 6;
        }
    }
    return { positions, normals, colors, indices };
}

function subdivide(vertices: Vec3[], faces: number[][]): number[][] {
    const midpoints = new Map<string, number>();
    const midpoint = (a: number, b: number) => {
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        let m = midpoints.get(key);
        if (m === undefined) {
            const v = vec3.normalize(vec3.create(), vec3.add(vec3.create(), vertices[a], vertices[b]));
            m = vertices.push(v) - 1;
            midpoints.set(key, m);
        }
        return m;
    };
    return faces.flatMap(([a, b, c]) => {
        const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
}

function icosphere(): { vertices: Vec3[]; faces: number[][] } {
    const t = (1 + Math.sqrt(5)) / 2;
    const vertices = ([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ] as Vec3[]).map((v) => vec3.normalize(v, v));
    const faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    return { vertices, faces: subdivide(vertices, faces) };
}

function createRock(x: number, z: number, radius: number, height: number, rand: () => number): MeshData {
    const { vertices, faces } = icosphere();
    const buriedBase = floorHeight(x, z) - 0.15;
    const squash = random.float(rand, 0.6, 0.95);
    const top = height * random.float(rand, 0.95, 1.05);
    const slant: [number, number] = [random.float(rand, -0.25, 0.25), random.float(rand, -0.25, 0.25)];
    const world = vertices.map(([vx, vy, vz]) => {
        const jitter = random.float(rand, 0.85, 1.12);
        const flareBelowWaterline = 1 + Math.max(0, -vy) * 0.35;
        const px = vx * radius * jitter * flareBelowWaterline;
        const pz = vz * radius * jitter * flareBelowWaterline * squash;
        const slantedTop = top + (slant[0] * px + slant[1] * pz);
        const y = Math.min(buriedBase + (vy + 1) * 0.5 * (height + 0.25) * jitter, buriedBase + slantedTop + 0.15);
        return [x + px, y, z + pz];
    });
    const tris = faces.map(([a, b, c]) => [...world[a], ...world[b], ...world[c]]);
    const facesUp = (n: number[]) => n[1] > 0.8;
    return flatMesh(tris, (n) => (facesUp(n) ? MOSS : STONE), ID_ROCK);
}

function doubleSided(a: number[], b: number[], c: number[]): number[][] {
    return [[...a, ...b, ...c], [...a, ...c, ...b]];
}

function flowerPetals(lift: number): number[][] {
    const petals = 5;
    const tris: number[][] = [];
    const top = [0, lift + 0.2, 0];
    for (let i = 0; i < petals; i++) {
        const a = (i / petals) * Math.PI * 2;
        const tip = [Math.cos(a) * 0.22, lift + 0.14, Math.sin(a) * 0.22];
        const l = [Math.cos(a - 0.5) * 0.07, lift + 0.05, Math.sin(a - 0.5) * 0.07];
        const r = [Math.cos(a + 0.5) * 0.07, lift + 0.05, Math.sin(a + 0.5) * 0.07];
        tris.push(...doubleSided(top, r, tip), ...doubleSided(top, tip, l));
    }
    return tris;
}

function createPad(radius: number, flower: boolean): MeshData {
    const segments = 14;
    const notch = 1;
    const thickness = PAD_THICKNESS;
    const tris: number[][] = [];
    const at = (i: number, r: number, y: number) => {
        const a = (i / segments) * Math.PI * 2;
        return [Math.cos(a) * r, y - thickness / 2, Math.sin(a) * r];
    };
    const centre = (y: number) => [0, y - thickness / 2, 0];
    for (let i = notch; i < segments; i++) {
        const a = at(i, radius, thickness);
        const b = at(i + 1, radius, thickness);
        tris.push([...centre(thickness), ...b, ...a]);
        const a0 = at(i, radius, 0);
        const b0 = at(i + 1, radius, 0);
        tris.push([...a0, ...a, ...b], [...a0, ...b, ...b0]);
        tris.push([...centre(0), ...a0, ...b0]);
    }
    const notchTop = at(notch, radius, thickness);
    const notchBottom = at(notch, radius, 0);
    const endTop = at(segments, radius, thickness);
    const endBottom = at(segments, radius, 0);
    tris.push([...centre(0), ...centre(thickness), ...notchTop], [...centre(0), ...notchTop, ...notchBottom]);
    tris.push([...centre(0), ...endBottom, ...endTop], [...centre(0), ...endTop, ...centre(thickness)]);

    if (flower) tris.push(...flowerPetals(thickness / 2));

    const facesDown = (n: number[]) => n[1] < -0.9;
    const isPetal = (n: number[]) => flower && n[1] > 0.3 && n[1] < 0.95;
    const colorOf = (n: number[]) => {
        if (facesDown(n)) return PAD_UNDERSIDE;
        if (isPetal(n)) return PETAL_PINK;
        return PAD_GREEN;
    };
    return flatMesh(tris, colorOf, ID_LILY);
}

export function init(): Pond {
    const seed = mulberry32.create(7);
    const rand = () => mulberry32.sample(seed);
    return {
        floor: createFloor(),
        rocks: ROCKS.map(([x, z, radius, height]) => ({ x, z, radius, height, mesh: createRock(x, z, radius, height, rand) })),
        pads: PADS.map(([x, z, radius, flower]) => ({ x, z, radius, mesh: createPad(radius, flower) })),
    };
}
