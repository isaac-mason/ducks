import { type Vec2, vec2 } from 'math';

export type Wave = { dir: Vec2; k: number; amp: number; speed: number };

function wave(dx: number, dz: number, k: number, amp: number, speed: number): Wave {
    return { dir: vec2.normalize(vec2.create(), [dx, dz]), k, amp, speed };
}

export const WAVES: Wave[] = [
    wave(1, 0.35, 0.8, 0.07, 1.0),
    wave(-0.45, 1, 1.35, 0.04, 1.5),
    wave(0.7, -0.8, 2.2, 0.02, 2.1),
];

export function height(x: number, z: number, t: number): number {
    let h = 0;
    for (const w of WAVES) h += w.amp * Math.sin((w.dir[0] * x + w.dir[1] * z) * w.k + t * w.speed);
    return h;
}
