import { clamp } from 'math';

export const CELLS = 288;
export const SPACING = 0.25;
export const ORIGIN = (-CELLS * SPACING) / 2;

const COURANT = 0.09;
const DAMPING = 0.997;
const SPONGE_CELLS = 14;
const MAX_HEIGHT = 1.2;
const DENT_REACH = 2.2;

export type Ripples = {
    height: Float32Array;
    velocity: Float32Array;
    displaced: Float32Array;
    wasDisplaced: Float32Array;
    damping: Float32Array;
    solid: Uint8Array;
    shore: number[];
};

export function init(): Ripples {
    const n = CELLS * CELLS;
    const damping = new Float32Array(n);
    for (let j = 0; j < CELLS; j++) {
        for (let i = 0; i < CELLS; i++) {
            const edge = Math.min(i, j, CELLS - 1 - i, CELLS - 1 - j);
            const s = Math.min(1, edge / SPONGE_CELLS);
            damping[j * CELLS + i] = DAMPING * (0.85 + 0.15 * s * s);
        }
    }
    return {
        height: new Float32Array(n),
        velocity: new Float32Array(n),
        displaced: new Float32Array(n),
        wasDisplaced: new Float32Array(n),
        damping,
        solid: new Uint8Array(n),
        shore: [],
    };
}

export function markSolid(ripples: Ripples, inside: (x: number, z: number) => boolean): void {
    const { solid } = ripples;
    for (let j = 0; j < CELLS; j++) {
        for (let i = 0; i < CELLS; i++) {
            if (inside(ORIGIN + i * SPACING, ORIGIN + j * SPACING)) solid[j * CELLS + i] = 1;
        }
    }
    ripples.shore = [];
    for (let j = 1; j < CELLS - 1; j++) {
        for (let i = 1; i < CELLS - 1; i++) {
            const k = j * CELLS + i;
            if (!solid[k] && (solid[k - 1] || solid[k + 1] || solid[k - CELLS] || solid[k + CELLS])) ripples.shore.push(k);
        }
    }
}

export function cellX(k: number): number {
    return ORIGIN + (k % CELLS) * SPACING;
}

export function cellZ(k: number): number {
    return ORIGIN + Math.floor(k / CELLS) * SPACING;
}

export function height(ripples: Ripples, x: number, z: number): number {
    const gx = (x - ORIGIN) / SPACING;
    const gz = (z - ORIGIN) / SPACING;
    const i = Math.floor(gx);
    const j = Math.floor(gz);
    if (i < 0 || j < 0 || i >= CELLS - 1 || j >= CELLS - 1) return 0;
    const fx = gx - i;
    const fz = gz - j;
    const h = ripples.height;
    const k = j * CELLS + i;
    const a = h[k] + (h[k + 1] - h[k]) * fx;
    const b = h[k + CELLS] + (h[k + CELLS + 1] - h[k + CELLS]) * fx;
    return a + (b - a) * fz;
}

export function press(ripples: Ripples, x: number, z: number, radius: number, depth: number): void {
    const reach = radius * DENT_REACH;
    const i0 = Math.max(1, Math.floor((x - reach - ORIGIN) / SPACING));
    const i1 = Math.min(CELLS - 2, Math.ceil((x + reach - ORIGIN) / SPACING));
    const j0 = Math.max(1, Math.floor((z - reach - ORIGIN) / SPACING));
    const j1 = Math.min(CELLS - 2, Math.ceil((z + reach - ORIGIN) / SPACING));
    const inv = 1 / (radius * radius);
    for (let j = j0; j <= j1; j++) {
        const dz = ORIGIN + j * SPACING - z;
        for (let i = i0; i <= i1; i++) {
            const dx = ORIGIN + i * SPACING - x;
            const d2 = (dx * dx + dz * dz) * inv;
            if (d2 < DENT_REACH * DENT_REACH) ripples.displaced[j * CELLS + i] += depth * Math.exp(-d2);
        }
    }
}

export function kick(ripples: Ripples, x: number, z: number, radius: number, depth: number, wavelength: number): void {
    const reach = radius * DENT_REACH;
    const k = (Math.PI * 2) / wavelength;
    const i0 = Math.max(1, Math.floor((x - reach - ORIGIN) / SPACING));
    const i1 = Math.min(CELLS - 2, Math.ceil((x + reach - ORIGIN) / SPACING));
    const j0 = Math.max(1, Math.floor((z - reach - ORIGIN) / SPACING));
    const j1 = Math.min(CELLS - 2, Math.ceil((z + reach - ORIGIN) / SPACING));
    const inv = 1 / (radius * radius);
    for (let j = j0; j <= j1; j++) {
        const dz = ORIGIN + j * SPACING - z;
        for (let i = i0; i <= i1; i++) {
            const dx = ORIGIN + i * SPACING - x;
            const d2 = (dx * dx + dz * dz) * inv;
            if (d2 < DENT_REACH * DENT_REACH) ripples.height[j * CELLS + i] -= depth * Math.exp(-d2) * Math.cos(Math.sqrt(d2) * radius * k);
        }
    }
}

function feedDisplacement(ripples: Ripples): void {
    const { height: h, displaced, wasDisplaced } = ripples;
    for (let k = 0; k < h.length; k++) {
        h[k] += wasDisplaced[k] - displaced[k];
        wasDisplaced[k] = displaced[k];
        displaced[k] = 0;
    }
}

function reflectingNeighbour(h: Float32Array, solid: Uint8Array, k: number, n: number): number {
    return solid[n] ? h[k] : h[n];
}

export function step(ripples: Ripples): void {
    feedDisplacement(ripples);

    const { height: h, velocity: v, damping, solid } = ripples;
    for (let j = 1; j < CELLS - 1; j++) {
        for (let i = 1; i < CELLS - 1; i++) {
            const k = j * CELLS + i;
            if (solid[k]) continue;
            const l = reflectingNeighbour(h, solid, k, k - 1);
            const r = reflectingNeighbour(h, solid, k, k + 1);
            const b = reflectingNeighbour(h, solid, k, k - CELLS);
            const f = reflectingNeighbour(h, solid, k, k + CELLS);
            const laplacian = l + r + b + f - 4 * h[k];
            v[k] = (v[k] + COURANT * laplacian) * damping[k];
        }
    }
    for (let k = 0; k < h.length; k++) {
        h[k] = solid[k] ? 0 : clamp((h[k] + v[k]) * damping[k], -MAX_HEIGHT, MAX_HEIGHT);
    }
}

export function pack(ripples: Ripples, texels: Float32Array): void {
    const { height: h } = ripples;
    const inv = 1 / (2 * SPACING);
    for (let j = 0; j < CELLS; j++) {
        for (let i = 0; i < CELLS; i++) {
            const k = j * CELLS + i;
            const l = i > 0 ? h[k - 1] : h[k];
            const r = i < CELLS - 1 ? h[k + 1] : h[k];
            const b = j > 0 ? h[k - CELLS] : h[k];
            const f = j < CELLS - 1 ? h[k + CELLS] : h[k];
            texels[k * 4] = h[k];
            texels[k * 4 + 1] = (r - l) * inv;
            texels[k * 4 + 2] = (f - b) * inv;
        }
    }
}
