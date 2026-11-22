import { type Vec3, vec3 } from 'math';

export const ID_DUCK = 1;
export const ID_FLOOR = 2;
export const ID_ROCK = 3;
export const ID_LILY = 4;

export type MeshData = {
    positions: Float32Array;
    normals: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
};

export async function loadDuck(url: string): Promise<MeshData> {
    const json = (await (await fetch(url)).json()) as {
        positions: number[];
        normals: number[];
        colors: number[];
        indices: number[];
    };
    const count = json.positions.length / 3;
    const colors = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
        colors[i * 4] = json.colors[i * 3];
        colors[i * 4 + 1] = json.colors[i * 3 + 1];
        colors[i * 4 + 2] = json.colors[i * 3 + 2];
        colors[i * 4 + 3] = ID_DUCK / 255;
    }
    return {
        positions: new Float32Array(json.positions),
        normals: new Float32Array(json.normals),
        colors,
        indices: new Uint32Array(json.indices),
    };
}

export function flatMesh(tris: number[][], colorOf: (normal: number[]) => number[], id: number): MeshData {
    const n = tris.length;
    const positions = new Float32Array(n * 9);
    const normals = new Float32Array(n * 9);
    const colors = new Float32Array(n * 12);
    const indices = new Uint32Array(n * 3);
    const a: Vec3 = [0, 0, 0], b: Vec3 = [0, 0, 0], c: Vec3 = [0, 0, 0];
    const normal: Vec3 = [0, 0, 0];
    for (let t = 0; t < n; t++) {
        vec3.fromBuffer(a, tris[t], 0);
        vec3.fromBuffer(b, tris[t], 3);
        vec3.fromBuffer(c, tris[t], 6);
        vec3.normalize(normal, vec3.cross(normal, vec3.sub(b, b, a), vec3.sub(c, c, a)));
        const color = colorOf(normal);
        for (let k = 0; k < 3; k++) {
            const v = t * 3 + k;
            positions.set(tris[t].slice(k * 3, k * 3 + 3), v * 3);
            normals.set(normal, v * 3);
            colors.set([color[0], color[1], color[2], id / 255], v * 4);
            indices[v] = v;
        }
    }
    return { positions, normals, colors, indices };
}
