import { clamp, type Quat, quat, type Vec2, type Vec3, vec2, vec3 } from 'math';
import { random } from 'math/random';
import {
    addBroadphaseLayer,
    addObjectLayer,
    type BodyId,
    convexHull,
    createWorld,
    cylinder,
    createWorldSettings,
    enableCollision,
    getShapeCenterOfMass,
    init as initTumble,
    MotionType,
    offsetCenterOfMass,
    rigidBody,
    type Shape,
    sphere,
    updateWorld,
    type World,
} from 'tumble.js';
import type { MeshData } from './mesh-data';
import { PAD_THICKNESS, type Pond } from './pond';
import * as ripples from './ripples';
import * as spray from './spray';
import * as waves from './waves';

export const MAX_DUCKS = 320;
export const MAX_SPLASHES = 32;
const STEP = 1 / 60;
const DUCK_MASS = 1;
const GRAVITY = 9.81;

const DUCK_CENTER_OF_MASS_Y = 0.16;

function hullRing(count: number, radiusX: number, radiusZ: number, y: number): Vec3[] {
    return Array.from({ length: count }, (_, i): Vec3 => {
        const a = (i / count) * Math.PI * 2;
        return [Math.cos(a) * radiusX, y, Math.sin(a) * radiusZ];
    });
}

const BELLY_POINTS: Vec3[] = [[0, 0.12, 0], [0.25, 0.14, 0], [-0.25, 0.14, 0]];
const FLOAT_POINTS: Vec3[] = [...BELLY_POINTS, ...hullRing(8, 0.5, 0.36, 0.22), ...hullRing(8, 0.36, 0.26, 0.48)];
const MAX_DEPTH = 0.4;
const UPRIGHT_WATERLINE = 0.3;
const BUOYANCY =
    (DUCK_MASS * GRAVITY) / FLOAT_POINTS.reduce((sum, p) => sum + clamp(UPRIGHT_WATERLINE - p[1], 0, MAX_DEPTH), 0);

export type Duck = {
    body: BodyId;
    position: Vec3;
    quaternion: Quat;
    wasAirborne: boolean;
};

export type Pad = { body: BodyId; home: [number, number]; radius: number; position: Vec3; quaternion: Quat };

export type Splash = { x: number; z: number; time: number; strength: number };

export type Bounds = { center: [number, number]; axes: [[number, number], [number, number]]; half: [number, number] };

export type Pointer = { active: boolean; x: number; z: number; body: BodyId; position: Vec3; poppedAt: number };

const BALL_RADIUS = 0.9;
const BALL_RIDE_HEIGHT = -0.35;
const BALL_HIDDEN_HEIGHT = -4;
const BALL_MAX_SPEED = 26;
const BALL_MAX_RISE_SPEED = 14;
const POINTER_DENT_RADIUS = 0.85;
const POINTER_DENT_DEPTH = 0.5;
const POINTER_DENT_BREATH = 0.15;
const DUCK_DENT_RADIUS = 0.42;
const DUCK_DENT_DEPTH = 0.1;
const POP_HEIGHT = 1.1;
const POP_TIME = 0.4;
const POP_MOUND_RADIUS = 1.6;
const POP_MOUND_HEIGHT = 0.45;
const POP_WAVELENGTH = 1.4;
const POP_DROPLETS = 22;
const LANDING_DROPLETS = 10;
const BOW_SPRAY_MIN_SPEED = 8;
const DROPLET_RIPPLE_RADIUS = 0.22;
const DROPLET_RIPPLE_DEPTH = 0.05;
const SLOPE_PUSH = 22;
const PAD_MASS = 0.3;
const PAD_DRAFT = 0.02;
const PAD_MAX_DEPTH = 0.08;
const PAD_RING = 6;
const PAD_BUOYANCY = (PAD_MASS * GRAVITY) / ((PAD_RING + 1) * PAD_DRAFT);
const PAD_DENT_DEPTH = 0.03;
const STEM_SLACK = 0.6;
const STEM_PULL = 2.5;
const BREAK_CREST_HEIGHT = 0.1;
const BREAK_SPRAY_CHANCE = 0.5;

export type Physics = {
    world: World;
    ripples: ripples.Ripples;
    spray: spray.Spray;
    pointer: Pointer;
    moving: number;
    duckShape: Shape;
    ducks: Duck[];
    pads: Pad[];
    splashes: Splash[];
    time: number;
    accumulator: number;
};

export async function init(duck: MeshData, pond: Pond): Promise<Physics> {
    await initTumble();

    const settings = createWorldSettings();
    const moving = addObjectLayer(settings, addBroadphaseLayer(settings));
    const fixed = addObjectLayer(settings, addBroadphaseLayer(settings));
    enableCollision(settings, moving, moving);
    enableCollision(settings, moving, fixed);
    const world = createWorld(settings);

    for (const rock of pond.rocks) {
        const shape = convexHull.create({ positions: rock.mesh.positions });
        rigidBody.create(world, { shape, objectLayer: fixed, motionType: MotionType.STATIC });
    }
    const water = ripples.init();
    ripples.markSolid(water, (x, z) => pond.rocks.some((rock) => insideAtWaterline(rock.mesh, x, z)));

    const pads = pond.pads.map(({ x, z, radius }): Pad => {
        const quaternion = quat.setAxisAngle(quat.create(), [0, 1, 0], random.float(Math.random, 0, Math.PI * 2));
        const body = rigidBody.create(world, {
            shape: cylinder.create({ radius, halfHeight: PAD_THICKNESS / 2, convexRadius: 0.01 }),
            objectLayer: moving,
            motionType: MotionType.DYNAMIC,
            position: [x, 0, z],
            quaternion,
            mass: PAD_MASS,
            linearDamping: 0.2,
            angularDamping: 1.5,
            restitution: 0.1,
            friction: 0.6,
            allowSleeping: false,
        });
        return { body, home: [x, z], radius, position: [x, 0, z], quaternion };
    });

    const hull = convexHull.create({ positions: duck.positions, hullTolerance: 0.02 });
    const centroid = getShapeCenterOfMass([0, 0, 0], hull)[1];
    const duckShape = offsetCenterOfMass.create({ shape: hull, offset: [0, DUCK_CENTER_OF_MASS_Y - centroid, 0] });

    const position: Vec3 = [0, BALL_HIDDEN_HEIGHT, 0];
    const body = rigidBody.create(world, {
        shape: sphere.create({ radius: BALL_RADIUS }),
        objectLayer: moving,
        motionType: MotionType.KINEMATIC,
        position,
        allowSleeping: false,
    });
    const pointer: Pointer = { active: false, x: 0, z: 0, body, position, poppedAt: -Infinity };
    return {
        world,
        ripples: water,
        spray: spray.init(),
        pointer,
        moving,
        duckShape,
        ducks: [],
        pads,
        splashes: [],
        time: 0,
        accumulator: 0,
    };
}

function insideAtWaterline(mesh: MeshData, x: number, z: number): boolean {
    const { positions, normals } = mesh;
    for (let v = 0; v < positions.length; v += 9) {
        const ahead = normals[v] * (x - positions[v]) + normals[v + 1] * -positions[v + 1] + normals[v + 2] * (z - positions[v + 2]);
        if (ahead > 0) return false;
    }
    return true;
}

export function addDuck(physics: Physics, x: number, y: number, z: number): Duck {
    if (physics.ducks.length >= MAX_DUCKS) {
        const oldest = physics.ducks.shift()!;
        rigidBody.remove(physics.world, oldest.body);
    }
    const yaw = random.float(Math.random, 0, Math.PI * 2);
    const quaternion = quat.setAxisAngle(quat.create(), [0, 1, 0], yaw);
    const body = rigidBody.create(physics.world, {
        shape: physics.duckShape,
        objectLayer: physics.moving,
        motionType: MotionType.DYNAMIC,
        position: [x, y, z],
        quaternion,
        mass: DUCK_MASS,
        linearDamping: 0.1,
        angularDamping: 0.6,
        restitution: 0.3,
        friction: 0.4,
        allowSleeping: false,
    });
    rigidBody.setAngularVelocity(physics.world, body, [random.float(Math.random, -0.5, 0.5), random.float(Math.random, -1, 1), random.float(Math.random, -0.5, 0.5)]);
    const duck: Duck = { body, position: [x, y, z], quaternion, wasAirborne: y > 0.5 };
    physics.ducks.push(duck);
    return duck;
}

export function movePointer(physics: Physics, x: number, z: number): void {
    const { pointer } = physics;
    const arriving = !pointer.active;
    if (arriving) vec3.set(pointer.position, x, BALL_HIDDEN_HEIGHT, z);
    pointer.active = true;
    pointer.x = x;
    pointer.z = z;
}

export function releasePointer(physics: Physics): void {
    physics.pointer.active = false;
}

export function pop(physics: Physics, x: number, z: number): void {
    movePointer(physics, x, z);
    physics.pointer.poppedAt = physics.time;
    ripples.kick(physics.ripples, x, z, POP_MOUND_RADIUS, -POP_MOUND_HEIGHT, POP_WAVELENGTH);
    addSplash(physics, x, z, 1);
    spray.crown(physics.spray, x, surface(physics, x, z), z, POP_DROPLETS, BALL_RADIUS, 6.5, 2.2);
}

const _target: Vec3 = [0, 0, 0];
const _dropletAt: Vec3 = [0, 0, 0];
const _dropletVelocity: Vec3 = [0, 0, 0];
const _stride: Vec2 = [0, 0];

function throwBowSpray(physics: Physics, heading: Vec2, speed: number): void {
    const { position } = physics.pointer;
    const x = position[0] + heading[0] * BALL_RADIUS;
    const z = position[2] + heading[1] * BALL_RADIUS;
    const side = random.float(Math.random, -1.5, 1.5);
    vec3.set(_dropletAt, x, surface(physics, x, z), z);
    vec3.set(
        _dropletVelocity,
        heading[0] * speed * 0.35 - heading[1] * side,
        random.float(Math.random, 2.5, 4.5),
        heading[1] * speed * 0.35 + heading[0] * side,
    );
    spray.emit(physics.spray, _dropletAt, _dropletVelocity);
}

function moveBall(physics: Physics): void {
    const { pointer, world, time } = physics;
    let y = BALL_HIDDEN_HEIGHT;
    if (pointer.active) {
        const popAge = (time - pointer.poppedAt) / POP_TIME;
        y = BALL_RIDE_HEIGHT + (popAge < 1 ? POP_HEIGHT * Math.sin(popAge * Math.PI) : 0);
    }
    vec3.set(_target, pointer.active ? pointer.x : pointer.position[0], y, pointer.active ? pointer.z : pointer.position[2]);

    const { position } = pointer;
    vec2.set(_stride, _target[0] - position[0], _target[2] - position[2]);
    const d = vec2.length(_stride);
    const reach = BALL_MAX_SPEED * STEP;
    if (d > reach) vec2.scale(_stride, _stride, reach / d);
    position[0] += _stride[0];
    position[2] += _stride[1];

    const speed = vec2.length(_stride) / STEP;
    const inWater = position[1] > BALL_HIDDEN_HEIGHT / 2;
    if (speed > BOW_SPRAY_MIN_SPEED && inWater && Math.random() < speed / 40) {
        throwBowSpray(physics, vec2.normalize(_stride, _stride), speed);
    }
    position[1] += clamp(_target[1] - position[1], -BALL_MAX_RISE_SPEED * STEP, BALL_MAX_RISE_SPEED * STEP);
    rigidBody.moveKinematic(world, pointer.body, position, IDENTITY, STEP);
}

const IDENTITY: Quat = [0, 0, 0, 1];

export function surface(physics: Physics, x: number, z: number): number {
    return waves.height(x, z, physics.time) + ripples.height(physics.ripples, x, z);
}

function addSplash(physics: Physics, x: number, z: number, strength: number): void {
    physics.splashes.push({ x, z, time: physics.time, strength });
    if (physics.splashes.length > MAX_SPLASHES) physics.splashes.shift();
}

const _point: Vec3 = [0, 0, 0];
const _velocity: Vec3 = [0, 0, 0];
const _force: Vec3 = [0, 0, 0];
const _slope: Vec2 = [0, 0];

function rippleSlope(out: Vec2, water: ripples.Ripples, x: number, z: number): Vec2 {
    const e = 0.35;
    out[0] = (ripples.height(water, x + e, z) - ripples.height(water, x - e, z)) / (2 * e);
    out[1] = (ripples.height(water, x, z + e) - ripples.height(water, x, z - e)) / (2 * e);
    return out;
}

function addHerding(force: Vec3, bounds: Bounds, x: number, z: number, wet: number): void {
    for (let i = 0; i < 2; i++) {
        const [ax, az] = bounds.axes[i];
        const along = (x - bounds.center[0]) * ax + (z - bounds.center[1]) * az;
        const strayed = Math.abs(along) - bounds.half[i];
        if (strayed > 0) {
            const push = -Math.sign(along) * strayed * 6 * wet;
            force[0] += ax * push;
            force[2] += az * push;
        }
    }
}

function applyWater(physics: Physics, duck: Duck, bounds: Bounds): void {
    const { world, time } = physics;
    const water = physics.ripples;
    const { position, quaternion } = duck;
    let submerged = 0;

    for (const local of FLOAT_POINTS) {
        vec3.add(_point, vec3.transformQuat(_point, local, quaternion), position);
        const depth = waves.height(_point[0], _point[2], time) + ripples.height(water, _point[0], _point[2]) - _point[1];
        if (depth <= 0) continue;
        submerged++;

        rigidBody.getVelocityAtPoint(_velocity, world, duck.body, _point);
        const lift = BUOYANCY * Math.min(depth, MAX_DEPTH);
        _force[0] = -_velocity[0] * 0.25;
        _force[1] = lift - _velocity[1] * 0.9;
        _force[2] = -_velocity[2] * 0.25;
        rigidBody.addForceAtPosition(world, duck.body, _force, _point, true);
    }

    if (submerged === 0) return;
    const wet = submerged / FLOAT_POINTS.length;

    const [x, , z] = position;
    const currentX = 0.25 * Math.sin(z * 0.21 + time * 0.13) + 0.12;
    const currentZ = 0.25 * Math.cos(x * 0.19 - time * 0.11);
    rigidBody.getLinearVelocity(_velocity, world, duck.body);
    rippleSlope(_slope, water, x, z);
    _force[0] = (currentX - _velocity[0]) * 0.8 * wet - _slope[0] * SLOPE_PUSH * wet;
    _force[1] = 0;
    _force[2] = (currentZ - _velocity[2]) * 0.8 * wet - _slope[1] * SLOPE_PUSH * wet;
    addHerding(_force, bounds, x, z, wet);
    rigidBody.addForce(world, duck.body, _force, true);

    ripples.press(water, x, z, DUCK_DENT_RADIUS, DUCK_DENT_DEPTH * wet);

    const paddleWobble = Math.sin(time * 0.7 + duck.body * 1.7) * 0.06 * wet;
    rigidBody.addTorque(world, duck.body, [0, paddleWobble, 0], true);
}

const _padPoint: Vec3 = [0, 0, 0];

function applyWaterToPad(physics: Physics, pad: Pad): void {
    const { world } = physics;
    let wet = 0;
    for (let i = 0; i <= PAD_RING; i++) {
        const a = (i / PAD_RING) * Math.PI * 2;
        const r = i === PAD_RING ? 0 : pad.radius * 0.75;
        vec3.set(_padPoint, Math.cos(a) * r, -PAD_THICKNESS / 2, Math.sin(a) * r);
        vec3.add(_point, vec3.transformQuat(_point, _padPoint, pad.quaternion), pad.position);
        const depth = surface(physics, _point[0], _point[2]) - _point[1];
        if (depth <= 0) continue;
        wet++;
        rigidBody.getVelocityAtPoint(_velocity, world, pad.body, _point);
        _force[0] = -_velocity[0] * 0.12;
        _force[1] = PAD_BUOYANCY * Math.min(depth, PAD_MAX_DEPTH) - _velocity[1] * 1.2;
        _force[2] = -_velocity[2] * 0.12;
        rigidBody.addForceAtPosition(world, pad.body, _force, _point, true);
    }
    if (wet === 0) return;
    const share = wet / (PAD_RING + 1);

    const [x, , z] = pad.position;
    const water = physics.ripples;
    rippleSlope(_slope, water, x, z);
    vec2.set(_stride, x - pad.home[0], z - pad.home[1]);
    const strayed = vec2.length(_stride);
    const stemTug = strayed > STEM_SLACK ? ((strayed - STEM_SLACK) * STEM_PULL * PAD_MASS) / strayed : 0;
    _force[0] = (-_slope[0] * SLOPE_PUSH - _stride[0] * stemTug) * PAD_MASS * share;
    _force[1] = 0;
    _force[2] = (-_slope[1] * SLOPE_PUSH - _stride[1] * stemTug) * PAD_MASS * share;
    rigidBody.addForce(world, pad.body, _force, true);
    ripples.press(water, x, z, pad.radius * 0.8, PAD_DENT_DEPTH * share);
}

function breakOnRocks(physics: Physics): void {
    const { height, velocity, shore } = physics.ripples;
    for (const k of shore) {
        const crest = height[k] - BREAK_CREST_HEIGHT;
        if (crest <= 0 || velocity[k] <= 0 || Math.random() > crest * BREAK_SPRAY_CHANCE) continue;
        const x = ripples.cellX(k);
        const z = ripples.cellZ(k);
        vec3.set(_dropletAt, x, surface(physics, x, z), z);
        vec3.set(
            _dropletVelocity,
            random.float(Math.random, -1, 1),
            random.float(Math.random, 2.5, 4) + crest * 6,
            random.float(Math.random, -1, 1),
        );
        spray.emit(physics.spray, _dropletAt, _dropletVelocity);
    }
}

function pressBall(physics: Physics): void {
    const { position } = physics.pointer;
    const surfaced = clamp((position[1] - BALL_HIDDEN_HEIGHT) / (BALL_RIDE_HEIGHT - BALL_HIDDEN_HEIGHT), 0, 1);
    const breath = 1 + POINTER_DENT_BREATH * Math.sin(physics.time * 6);
    ripples.press(physics.ripples, position[0], position[2], POINTER_DENT_RADIUS, POINTER_DENT_DEPTH * breath * surfaced);
}

function splashLandings(physics: Physics): void {
    for (const duck of physics.ducks) {
        rigidBody.getTransform(duck.position, duck.quaternion, physics.world, duck.body);
        const water = surface(physics, duck.position[0], duck.position[2]);
        const airborne = duck.position[1] > water + 0.15;
        if (duck.wasAirborne && !airborne) {
            rigidBody.getLinearVelocity(_velocity, physics.world, duck.body);
            if (_velocity[1] < -1.5) {
                const strength = Math.min(1, -_velocity[1] / 8);
                addSplash(physics, duck.position[0], duck.position[2], strength);
                const count = Math.round(LANDING_DROPLETS * strength);
                spray.crown(physics.spray, duck.position[0], water, duck.position[2], count, 0.4, 2 + 3 * strength, 1.3);
            }
        }
        duck.wasAirborne = airborne;
    }
}

export function update(physics: Physics, dt: number, bounds: Bounds): void {
    physics.accumulator = Math.min(physics.accumulator + dt, STEP * 4);
    while (physics.accumulator >= STEP) {
        physics.accumulator -= STEP;
        for (const duck of physics.ducks) applyWater(physics, duck, bounds);
        for (const pad of physics.pads) applyWaterToPad(physics, pad);
        moveBall(physics);
        pressBall(physics);
        spray.step(
            physics.spray,
            STEP,
            (x, z) => surface(physics, x, z),
            ({ position }) => ripples.kick(physics.ripples, position[0], position[2], DROPLET_RIPPLE_RADIUS, DROPLET_RIPPLE_DEPTH, 1),
        );
        ripples.step(physics.ripples);
        breakOnRocks(physics);
        updateWorld(physics.world, undefined, STEP);
        physics.time += STEP;

        for (const pad of physics.pads) rigidBody.getTransform(pad.position, pad.quaternion, physics.world, pad.body);
        splashLandings(physics);
    }
}
