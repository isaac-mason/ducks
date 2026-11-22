import { type Vec3, vec3 } from 'math';
import { random } from 'math/random';

export const MAX_DROPLETS = 96;
const GRAVITY = 9.81;

export type Droplet = { position: Vec3; velocity: Vec3 };

export type Spray = { droplets: Droplet[] };

export function init(): Spray {
    return { droplets: [] };
}

export function emit(spray: Spray, position: Vec3, velocity: Vec3): void {
    if (spray.droplets.length >= MAX_DROPLETS) spray.droplets.shift();
    spray.droplets.push({ position: vec3.clone(position), velocity: vec3.clone(velocity) });
}

const _position: Vec3 = [0, 0, 0];
const _velocity: Vec3 = [0, 0, 0];

export function crown(spray: Spray, x: number, y: number, z: number, count: number, radius: number, up: number, out: number): void {
    for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2 + random.float(Math.random, -0.3, 0.3);
        const c = Math.cos(a);
        const s = Math.sin(a);
        vec3.set(_position, x + c * radius, y, z + s * radius);
        const speed = out * random.float(Math.random, 0.6, 1.2);
        vec3.set(_velocity, c * speed, up * random.float(Math.random, 0.6, 1.15), s * speed);
        emit(spray, _position, _velocity);
    }
}

export function step(spray: Spray, dt: number, surface: (x: number, z: number) => number, onLand: (droplet: Droplet) => void): void {
    const { droplets } = spray;
    for (let i = droplets.length - 1; i >= 0; i--) {
        const droplet = droplets[i];
        droplet.velocity[1] -= GRAVITY * dt;
        vec3.scaleAndAdd(droplet.position, droplet.position, droplet.velocity, dt);
        if (droplet.velocity[1] < 0 && droplet.position[1] < surface(droplet.position[0], droplet.position[2])) {
            onLand(droplet);
            droplets.splice(i, 1);
        }
    }
}
