import {
    add,
    clamp,
    cos,
    d,
    div,
    dot,
    exp,
    Fn,
    f32,
    floor,
    fract,
    If,
    index,
    Loop,
    length,
    max,
    min,
    mix,
    type Node,
    normalize,
    reflect,
    round,
    select,
    sign,
    sin,
    smoothstep,
    sub,
    type TextureNode,
    Var,
    vec2,
    vec2i,
    vec3,
    vec4,
} from 'gpucat';
import type { Vec3 } from 'math';
import { ID_DUCK, ID_FLOOR, ID_LILY, ID_ROCK } from './mesh-data';
import * as palette from './palette';
import { MAX_DUCKS, MAX_SPLASHES } from './physics';
import { PIXELS_PER_UNIT } from './pixel-camera';
import { FAR, NEAR } from './pixel-camera';
import { CELLS, ORIGIN, SPACING } from './ripples';
import { MAX_DROPLETS } from './spray';
import { WAVES } from './waves';

export type ShaderConstants = {
    light: Vec3;
    forward: Vec3;
    right: Vec3;
    up: Vec3;
    viewLight: Vec3;
};

export type WaterInputs = {
    ducks: Node<d.sizedArray<d.vec4f, typeof MAX_DUCKS>>;
    duckCount: Node<d.i32>;
    splashes: Node<d.sizedArray<d.vec4f, typeof MAX_SPLASHES>>;
    droplets: TextureNode<d.texture2d>;
    ripples: TextureNode<d.texture2d>;
};

export type Neighbour = { depth: Node<d.f32>; normal: Node<d.vec3f> };

const constant = (v: Vec3) => vec3(v[0], v[1], v[2]);
const isId = (id: Node<d.f32>, value: number) => id.equal(value);

export function init(c: ShaderConstants, water: WaterInputs) {
    const LIGHT = constant(c.light);
    const FORWARD = constant(c.forward);
    const RIGHT = constant(c.right);
    const UP = constant(c.up);
    const VIEW_LIGHT = constant(c.viewLight);
    const SHADOW_TINT = vec3(0.42, 0.44, 0.72);
    const FOAM = vec3(0.93, 0.98, 1);
    const WATERHIT_STEPS = 6;
    const WATERHIT_RELAXATION = 0.7;
    const DROPLET_HALF_SIZE = 1 / PIXELS_PER_UNIT;
    const SPLASH_RING_LIFE = 1.6;
    const SPLASH_RING_BRIGHTNESS = 0.7;
    const PERCEPTUAL_WEIGHTS = vec3(2, 4, 3);
    const PALETTE = palette.rgb().map((c) => constant(c));

    const hash = Fn(
        (p: Node<d.vec2f>) => {
            const q = fract(p.mul(vec2(123.34, 456.21))).toVar('q');
            const r = q.add(dot(q, q.add(45.32))).toVar('r');
            return fract(r.x.mul(r.y));
        },
        { name: 'hash', params: [{ name: 'p', type: d.vec2f }], return: d.f32 },
    );

    const noise = Fn(
        (p: Node<d.vec2f>) => {
            const i = floor(p).toVar('i');
            const f = fract(p).toVar('f');
            const u = f
                .mul(f)
                .mul(vec2(3, 3).sub(f.mul(2)))
                .toVar('u');
            const a = hash(i);
            const b = hash(i.add(vec2(1, 0)));
            const cc = hash(i.add(vec2(0, 1)));
            const dd = hash(i.add(vec2(1, 1)));
            return mix(mix(a, b, u.x), mix(cc, dd, u.x), u.y);
        },
        { name: 'noise', params: [{ name: 'p', type: d.vec2f }], return: d.f32 },
    );

    const ripple = Fn(
        (p: Node<d.vec2f>) => {
            const g = p.sub(ORIGIN).div(SPACING).toVar('g');
            const last = CELLS - 2;
            const cell = clamp(floor(g), vec2(0, 0), vec2(last, last)).toVar('cell');
            const f = clamp(g.sub(cell), vec2(0, 0), vec2(1, 1)).toVar('f');
            const c = vec2i(cell).toVar('c');
            const a = water.ripples.load(c).xyz;
            const b = water.ripples.load(c.add(vec2i(1, 0))).xyz;
            const cc = water.ripples.load(c.add(vec2i(0, 1))).xyz;
            const dd = water.ripples.load(c.add(vec2i(1, 1))).xyz;
            return mix(mix(a, b, f.x), mix(cc, dd, f.x), f.y);
        },
        { name: 'ripple', params: [{ name: 'p', type: d.vec2f }], return: d.vec3f },
    );

    const phase = (p: Node<d.vec2f>, t: Node<d.f32>, w: (typeof WAVES)[number]) =>
        dot(p, vec2(w.dir[0], w.dir[1]))
            .mul(w.k)
            .add(t.mul(w.speed));

    const waveHeight = Fn(
        (p: Node<d.vec2f>, t: Node<d.f32>) =>
            WAVES.map((w) => sin(phase(p, t, w)).mul(w.amp)).reduce((sum, term) => sum.add(term), ripple(p).x),
        {
            name: 'waveHeight',
            params: [
                { name: 'p', type: d.vec2f },
                { name: 't', type: d.f32 },
            ],
            return: d.f32,
        },
    );

    const waveNormal = Fn(
        (p: Node<d.vec2f>, t: Node<d.f32>) => {
            const slopes = WAVES.map((w) => cos(phase(p, t, w)).mul(w.amp * w.k));
            const r = ripple(p).toVar('r');
            const dx = WAVES.map((w, i) => slopes[i].mul(w.dir[0])).reduce((sum, term) => sum.add(term), r.y);
            const dz = WAVES.map((w, i) => slopes[i].mul(w.dir[1])).reduce((sum, term) => sum.add(term), r.z);
            return normalize(vec3(dx.negate(), 1, dz.negate()));
        },
        {
            name: 'waveNormal',
            params: [
                { name: 'p', type: d.vec2f },
                { name: 't', type: d.f32 },
            ],
            return: d.vec3f,
        },
    );

    const surface = Fn(
        (base: Node<d.vec4f>, world: Node<d.vec3f>, n: Node<d.vec3f>, shadow: Node<d.f32>, time: Node<d.f32>) => {
            const id = round(base.w.mul(255)).toVar('id');
            const albedo = Var('albedo', base.xyz);

            If(isId(id, ID_FLOOR), () => {
                const sandDrift = noise(world.xz.mul(0.6)).add(noise(world.xz.mul(1.7)).mul(0.5));
                If(sandDrift.greaterThan(0.95), () => albedo.assign(albedo.mul(0.88)));
                const palePebble = hash(floor(world.xz.mul(2.5))).greaterThan(0.985);
                If(palePebble, () => albedo.assign(vec3(0.84, 0.76, 0.6)));
            });

            const depth = world.y.negate();
            const sunlitFloorUnderWater = isId(id, ID_FLOOR)
                .and(depth.greaterThan(0.2))
                .and(shadow.greaterThan(0.75));
            If(sunlitFloorUnderWater, () => {
                const causticCoord = world.xz
                    .mul(0.9)
                    .add(ripple(world.xz).yz.mul(1.5))
                    .toVar('causticCoord');
                const a = noise(causticCoord.add(vec2(time.mul(0.21), time.mul(0.13))));
                const b = noise(causticCoord.mul(1.3).sub(vec2(time.mul(0.17), time.mul(-0.11))));
                const causticWidth = div(0.045, add(1, depth.mul(0.3)));
                If(a.sub(b).abs().lessThan(causticWidth), () => albedo.assign(min(albedo.mul(1.3), vec3(1, 1, 1))));
            });

            If(isId(id, ID_ROCK), () => {
                const isMoss = albedo.y.greaterThan(albedo.x.mul(1.1));
                const strata = fract(world.y.mul(2.4).add(noise(world.xz.mul(1.3)).mul(0.6)));
                If(isMoss.not().and(strata.lessThan(0.22)), () => albedo.assign(albedo.mul(vec3(0.84, 0.84, 0.9))));
                const waterline = waveHeight(world.xz, time).toVar('waterline');
                const wetBand = add(0.1, noise(world.xz.mul(3).add(time.mul(0.4))).mul(0.1));
                If(world.y.lessThan(waterline.add(wetBand)), () => albedo.assign(albedo.mul(vec3(0.5, 0.54, 0.7))));
                const underwaterWeed = world.y.lessThan(waterline.sub(0.25));
                If(underwaterWeed, () => albedo.assign(mix(albedo, vec3(0.3, 0.42, 0.22), 0.55)));
            });

            const ndl = max(dot(n, LIGHT), 0).mul(shadow).toVar('ndl');
            const band = select(select(f32(0), 0.55, ndl.greaterThan(0.2)), 1, ndl.greaterThan(0.55));
            const color = Var('color', albedo.mul(mix(SHADOW_TINT, vec3(1.04, 1, 0.96), band)));

            const glint = dot(n, normalize(LIGHT.sub(FORWARD))).greaterThan(0.965);
            If(
                isId(id, ID_DUCK)
                    .and(shadow.greaterThan(0.5))
                    .and(glint),
                () => color.assign(mix(color, vec3(1, 1, 0.92), 0.75)),
            );
            return color;
        },
        {
            name: 'surface',
            params: [
                { name: 'base', type: d.vec4f },
                { name: 'world', type: d.vec3f },
                { name: 'n', type: d.vec3f },
                { name: 'shadow', type: d.f32 },
                { name: 'time', type: d.f32 },
            ],
            return: d.vec3f,
        },
    );

    const outline = (center: Node<d.vec4f>, n: Node<d.vec3f>, depth: Node<d.f32>, neighbours: Neighbour[]) => {
        const range = FAR - NEAR;
        const id = round(center.w.mul(255));
        const z = depth.mul(range);

        let fartherNeighbours: Node<d.f32> = f32(0);
        let litCrease: Node<d.f32> = f32(0);
        for (const nb of neighbours) {
            const zn = nb.depth.mul(range);
            fartherNeighbours = fartherNeighbours.add(clamp(zn.sub(z), 0, 1));
            const facesLight = smoothstep(-0.01, 0.01, dot(n.sub(nb.normal), VIEW_LIGHT));
            const isNearer = clamp(sign(zn.sub(z).mul(0.25).add(0.0025)), 0, 1);
            litCrease = litCrease.add(sub(1, dot(n, nb.normal)).mul(facesLight).mul(isNearer));
        }

        const hasInlines = isId(id, ID_DUCK).or(isId(id, ID_ROCK)).or(isId(id, ID_LILY));
        const dark = center.xyz.mul(vec3(0.42, 0.36, 0.48));
        const light = min(center.xyz.mul(1.22).add(0.05), vec3(1, 1, 1));
        const inlined = select(center.xyz, light, hasInlines.and(litCrease.greaterThan(0.12)));
        return select(inlined, dark, fartherNeighbours.greaterThan(0.35));
    };

    const waterHit = Fn(
        (origin: Node<d.vec3f>, time: Node<d.f32>) => {
            const toFlatWater = origin.y.negate().div(FORWARD.y);
            const t = Var('t', toFlatWater);
            Loop(WATERHIT_STEPS, () => {
                const p = origin.add(FORWARD.mul(t));
                const toWaveUnderP = waveHeight(p.xz, time).sub(origin.y).div(FORWARD.y);
                t.assign(mix(t, toWaveUnderP, WATERHIT_RELAXATION));
            });
            return vec4(origin.add(FORWARD.mul(t)), t);
        },
        {
            name: 'waterHit',
            params: [
                { name: 'origin', type: d.vec3f },
                { name: 'time', type: d.f32 },
            ],
            return: d.vec4f,
        },
    );

    const shadeWater = Fn(
        (
            scene: Node<d.vec3f>,
            sceneT: Node<d.f32>,
            sceneId: Node<d.f32>,
            origin: Node<d.vec3f>,
            hit: Node<d.vec4f>,
            sunlit: Node<d.f32>,
            shadow: Node<d.f32>,
            time: Node<d.f32>,
        ) => {
            const result = Var('result', scene);

            If(sceneT.greaterThanEqual(hit.w), () => {
                const p = hit.xyz.toVar('p');
                const gap = p.y.sub(origin.add(FORWARD.mul(sceneT)).y).toVar('gap');

                const absorb = sub(1, exp(gap.mul(-0.8)));
                const threeDepthBands = clamp(floor(absorb.mul(3).add(0.35)).div(3), 0, 1);
                const shallow = scene.mul(vec3(0.55, 0.85, 0.95)).add(vec3(0.02, 0.08, 0.12));
                const color = Var('color', mix(shallow, vec3(0.12, 0.32, 0.52), threeDepthBands.mul(0.85)));

                const n = waveNormal(p.xz, time).toVar('n');
                const rip = ripple(p.xz).toVar('rip');

                const tiltTowardsSun = dot(n, LIGHT).sub(LIGHT.y).toVar('tiltTowardsSun');
                If(tiltTowardsSun.greaterThan(0.08).and(sunlit.greaterThan(0.75)), () =>
                    color.assign(mix(color, FOAM, select(f32(0.25), 0.5, tiltTowardsSun.greaterThan(0.22)))),
                );
                If(tiltTowardsSun.lessThan(-0.08), () =>
                    color.assign(
                        color.mul(
                            select(vec3(0.74, 0.82, 0.92), vec3(0.5, 0.6, 0.78), tiltTowardsSun.lessThan(-0.22)),
                        ),
                    ),
                );

                const flow = p.xz
                    .mul(0.3)
                    .add(vec2(time.mul(0.06), time.mul(0.035)))
                    .add(rip.yz.mul(0.5))
                    .toVar('flow');
                const contour = noise(flow)
                    .mul(0.8)
                    .add(noise(flow.mul(2.1).sub(time.mul(0.05))).mul(0.2));
                If(contour.sub(0.6).abs().lessThan(0.008), () => color.assign(mix(color, FOAM, 0.3)));

                const glitter = hash(floor(p.xz.mul(4)).add(floor(time.mul(2))));
                If(
                    dot(reflect(FORWARD, n), LIGHT)
                        .greaterThan(0.99)
                        .and(glitter.greaterThan(0.8))
                        .and(sunlit.greaterThan(0.75)),
                    () => color.assign(FOAM),
                );

                const shadowedWater = color.mul(vec3(0.6, 0.68, 0.84));
                color.assign(mix(shadowedWater, color, shadow));

                const isLily = isId(round(sceneId), ID_LILY);
                const touchesSurface = gap.lessThan(0.07);
                const foam = Var('foam', select(f32(0), 1, touchesSurface.and(isLily.not())));
                If(isId(round(sceneId), ID_ROCK), () => {
                    const submergedRock = vec3(0.1, 0.26, 0.42);
                    color.assign(mix(color, submergedRock, 0.45));
                    const swell = max(rip.x, 0).toVar('swell');
                    If(gap.lessThan(add(0.07, swell.mul(0.3))), () => foam.assign(1));
                    const lap = fract(time.mul(0.35).sub(gap.mul(1.6)).add(noise(p.xz.mul(0.8)).mul(0.5)));
                    const onLappingRing = gap.lessThan(0.4).and(lap.lessThan(sub(0.05, gap.mul(0.08))));
                    If(onLappingRing, () => color.assign(mix(color, FOAM, 0.25)));
                    If(swell.greaterThan(0.2).and(gap.lessThan(0.2)), () => foam.assign(1));
                });

                const whitecap = rip.x.greaterThan(0.24);
                If(whitecap, () => foam.assign(1));
                Loop({ end: water.duckCount }, ({ i }) => {
                    const duck = index(water.ducks, i).toVar('duck');
                    const dp = p.xz.sub(duck.xy);
                    const local = vec2(dot(dp, duck.zw), dot(dp, vec2(duck.w.negate(), duck.z)));
                    const distFromHull = length(local.div(vec2(0.66, 0.5)))
                        .sub(1)
                        .mul(0.5);
                    const onCollar = distFromHull.lessThan(0.035);
                    If(onCollar, () => foam.assign(1));
                });

                const splashRing = Var('splashRing', f32(0));
                Loop(MAX_SPLASHES, ({ i }) => {
                    const s = index(water.splashes, i).toVar('splash');
                    const age = time.sub(s.z).toVar('age');
                    const life = age.div(SPLASH_RING_LIFE).toVar('life');
                    const alive = s.w
                        .greaterThan(0)
                        .and(life.greaterThanEqual(0))
                        .and(life.lessThan(1));
                    const fade = sub(1, smoothstep(0.15, 1, life)).mul(s.w.mul(0.5).add(0.5)).toVar('fade');
                    const radius = age.mul(add(1.6, s.w.mul(1.6)));
                    const width = mix(0.08, 0.035, life);
                    const onRing = length(p.xz.sub(s.xy)).sub(radius).abs().lessThan(width);
                    const artPixel = floor(p.xz.mul(PIXELS_PER_UNIT));
                    const survivesDissolve = hash(artPixel.add(s.xy)).lessThan(fade);
                    If(alive.and(onRing).and(survivesDissolve), () => splashRing.assign(max(splashRing, add(0.65, fade.mul(0.35)))));
                });
                color.assign(mix(color, FOAM, splashRing.mul(SPLASH_RING_BRIGHTNESS)));

                result.assign(select(color, FOAM, foam.greaterThan(0.5)));
            });
            return result;
        },
        {
            name: 'shadeWater',
            params: [
                { name: 'scene', type: d.vec3f },
                { name: 'sceneT', type: d.f32 },
                { name: 'sceneId', type: d.f32 },
                { name: 'origin', type: d.vec3f },
                { name: 'hit', type: d.vec4f },
                { name: 'sunlit', type: d.f32 },
                { name: 'shadow', type: d.f32 },
                { name: 'time', type: d.f32 },
            ],
            return: d.vec3f,
        },
    );

    const spray = Fn(
        (color: Node<d.vec3f>, origin: Node<d.vec3f>, nearest: Node<d.f32>) => {
            const result = Var('sprayed', color);
            Loop(MAX_DROPLETS, ({ i }) => {
                const droplet = water.droplets.load(vec2i(i, 0)).toVar('droplet');
                const rel = droplet.xyz.sub(origin).toVar('rel');
                const near = droplet.w.greaterThan(0).and(dot(rel, FORWARD).lessThan(nearest));
                const onPixel = dot(rel, RIGHT).abs().lessThan(DROPLET_HALF_SIZE).and(dot(rel, UP).abs().lessThan(DROPLET_HALF_SIZE));
                If(near.and(onPixel), () => result.assign(FOAM));
            });
            return result;
        },
        {
            name: 'spray',
            params: [
                { name: 'color', type: d.vec3f },
                { name: 'origin', type: d.vec3f },
                { name: 'nearest', type: d.f32 },
            ],
            return: d.vec3f,
        },
    );

    const quantize = Fn(
        (color: Node<d.vec3f>) => {
            const distance = (c: Node<d.vec3f>) => {
                const diff = color.sub(c);
                return dot(diff.mul(diff), PERCEPTUAL_WEIGHTS);
            };
            const best = Var('best', PALETTE[0]);
            const bestDistance = Var('bestDistance', distance(PALETTE[0]));
            for (const c of PALETTE.slice(1)) {
                const d2 = distance(c).toVar();
                If(d2.lessThan(bestDistance), () => {
                    bestDistance.assign(d2);
                    best.assign(c);
                });
            }
            return best;
        },
        {
            name: 'quantize',
            params: [{ name: 'color', type: d.vec3f }],
            return: d.vec3f,
        },
    );

    return { surface, outline, ripple, waterHit, shadeWater, spray, quantize };
}

export type Shaders = ReturnType<typeof init>;
