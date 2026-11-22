import {
    add,
    attribute,
    clamp,
    type CanvasTarget,
    cameraProjectionMatrix,
    cameraViewMatrix,
    comparisonSampler,
    createCanvasTarget,
    createDataTexture,
    createIndexBuffer,
    createMaterial,
    createRenderTarget,
    createVertexBuffer,
    cross,
    d,
    type DataTexture,
    depthTexture,
    dot,
    f32,
    floor,
    Fn,
    frame,
    greaterThan,
    i32,
    If,
    instanceIndex,
    lessThan,
    or,
    select,
    smoothstep,
    fullscreen,
    Geometry,
    init as initRenderer,
    Mesh,
    modelNormalMatrix,
    modelWorldMatrix,
    max,
    min,
    mix,
    mod,
    mrt,
    mul,
    type Node,
    normalize,
    OrthographicCamera,
    type Renderer,
    type RenderTarget,
    screenCoordinate,
    sub,
    type Texture,
    texture,
    textureSampleCompareLevel,
    Uniform,
    uniform,
    Var,
    varying,
    vec2,
    vec2i,
    vec3f,
    vec4,
    webgpu,
} from 'gpucat';
import { mat4, quat, type Vec3, vec3 } from 'math';
import { ID_FLOOR, type MeshData } from './mesh-data';
import { MAX_DUCKS, MAX_SPLASHES, type Physics, surface } from './physics';
import { FAR, NEAR, PIXELS_PER_UNIT, type PixelCamera } from './pixel-camera';
import type { Pond } from './pond';
import * as ripples from './ripples';
import * as shaders from './shaders';
import { MAX_DROPLETS } from './spray';

const SHADOW_SIZE = 2048;
const SHADOW_EXTENT = 42;
const SHADOW_DEPTH_RANGE = 79.9;
const SHADOW_BIAS = 0.0006;
const SHADOW_SEARCH_RADIUS = 10;
const SHADOW_SEARCH: [number, number][] = [
    [0, 0],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [0.7, 0.7],
    [-0.7, 0.7],
    [0.7, -0.7],
    [-0.7, -0.7],
];
const SHADOW_FILTER_DISC: [number, number][] = [
    [0, 0],
    [0.5, 0],
    [-0.5, 0],
    [0, 0.5],
    [0, -0.5],
    [0.71, 0.71],
    [-0.71, 0.71],
    [0.71, -0.71],
    [-0.71, -0.71],
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
];
const SHADOW_PENUMBRA_TEXELS_PER_UNIT = 2;
const HIGH_CASTER_SHADOW_STRENGTH = 0.3;
const HIGH_CASTER_HEIGHT = 6;
const UNDERWATER_SHADOW_WOBBLE = 0.9;
const REFRACTION_ART_PIXELS_PER_SLOPE = 5;
const DUCK_FOAM_MAX_DISTANCE_FROM_WATER = 0.45;

export type Render = {
    renderer: Renderer;
    view: CanvasTarget;
    gbuffer: RenderTarget;
    composite: RenderTarget;
    shadow: RenderTarget;
    lightCamera: OrthographicCamera;
    light: Vec3;
    shadowMaterial: ReturnType<typeof createMaterial>;
    duckShadowMaterial: ReturnType<typeof createMaterial>;
    gbufferMrt: ReturnType<typeof mrt>;
    floor: Mesh;
    rocks: Mesh[];
    pads: Mesh[];
    ducks: Mesh;
    compositeMesh: Mesh;
    upscaleMesh: Mesh;
    ripples: DataTexture;
    duckTransforms: DataTexture;
    droplets: DataTexture;
    uniforms: {
        lightViewProjection: Uniform<d.mat4x4f>;
        time: Uniform<d.f32>;
        camPosition: Uniform<d.vec3f>;
        camRight: Uniform<d.vec3f>;
        camUp: Uniform<d.vec3f>;
        halfSize: Uniform<d.vec2f>;
        ducks: Uniform<d.sizedArray<d.vec4f, typeof MAX_DUCKS>>;
        duckCount: Uniform<d.i32>;
        splashes: Uniform<d.sizedArray<d.vec4f, typeof MAX_SPLASHES>>;
        upscaleShift: Uniform<d.vec2f>;
        artPixelSize: Uniform<d.f32>;
    };
};

function geometry(mesh: MeshData): Geometry {
    const g = new Geometry();
    g.setBuffer('position', createVertexBuffer(d.vec3f, mesh.positions));
    g.setBuffer('normal', createVertexBuffer(d.vec3f, mesh.normals));
    g.setBuffer('color', createVertexBuffer(d.vec4f, mesh.colors));
    g.setIndex(createIndexBuffer(mesh.indices));
    return g;
}

function sunUpAndLeftOfScreen(cam: PixelCamera): Vec3 {
    const ground: Vec3 = vec3.normalize(vec3.create(), [cam.forward[0], 0, cam.forward[2]]);
    return vec3.normalize(vec3.create(), [-cam.right[0] * 0.75 + ground[0] * 0.35, 1.1, -cam.right[2] * 0.75 + ground[2] * 0.35]);
}

export async function init(
    canvas: OffscreenCanvas,
    cam: PixelCamera,
    duck: MeshData,
    pond: Pond,
    onError: (message: string) => void,
): Promise<Render> {
    const view = createCanvasTarget(canvas, { samples: 1 });
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no webgpu adapter');
    const device = await adapter.requestDevice();
    const renderer = await initRenderer(webgpu({ adapter, device }));
    renderer.onValidationError = onError;

    const light = sunUpAndLeftOfScreen(cam);
    const viewRotation = mat4.invert(mat4.create(), mat4.fromQuat(mat4.create(), cam.camera.quaternion))!;
    const viewLight = vec3.normalize(vec3.create(), vec3.transformMat4(vec3.create(), light, viewRotation));

    const uniforms: Render['uniforms'] = {
        lightViewProjection: new Uniform(d.mat4x4f, mat4.create()),
        time: new Uniform(d.f32, 0),
        camPosition: new Uniform(d.vec3f, [0, 0, 0]),
        camRight: new Uniform(d.vec3f, [1, 0, 0]),
        camUp: new Uniform(d.vec3f, [0, 1, 0]),
        halfSize: new Uniform(d.vec2f, [1, 1]),
        ducks: new Uniform(d.sizedArray(d.vec4f, MAX_DUCKS), new Float32Array(MAX_DUCKS * 4)),
        duckCount: new Uniform(d.i32, 0),
        splashes: new Uniform(d.sizedArray(d.vec4f, MAX_SPLASHES), new Float32Array(MAX_SPLASHES * 4)),
        upscaleShift: new Uniform(d.vec2f, [0, 0]),
        artPixelSize: new Uniform(d.f32, 1),
    };

    const rippleTexture = createDataTexture(new Float32Array(ripples.CELLS * ripples.CELLS * 4), ripples.CELLS, ripples.CELLS, {
        format: 'rgba32float',
        magFilter: 'nearest',
        minFilter: 'nearest',
    });
    rippleTexture.name = 'ripples';
    const droplets = createDataTexture(new Float32Array(MAX_DROPLETS * 4), MAX_DROPLETS, 1, {
        format: 'rgba32float',
        magFilter: 'nearest',
        minFilter: 'nearest',
    });
    droplets.name = 'droplets';

    const sh = shaders.init(
        { light, forward: cam.forward, right: cam.right, up: cam.up, viewLight },
        {
            ducks: uniform(uniforms.ducks),
            duckCount: uniform(uniforms.duckCount),
            splashes: uniform(uniforms.splashes),
            droplets: texture(droplets),
            ripples: texture(rippleTexture),
        },
    );

    const shadow = createRenderTarget(SHADOW_SIZE, SHADOW_SIZE, {
        count: 0,
        depthFormat: 'depth32float',
        depthSampled: true,
    });
    const lightCamera = new OrthographicCamera(-SHADOW_EXTENT, SHADOW_EXTENT, SHADOW_EXTENT, -SHADOW_EXTENT, 0.1, 80);
    const shadowDepth = depthTexture(shadow.depthTexture!);
    const shadowSampler = comparisonSampler(shadow.depthTexture!, 'less');
    const lightViewProjection = uniform(uniforms.lightViewProjection);

    const bayer2 = (p: Node<d.vec2f>) =>
        mod(
            mod(p.x, 2)
                .mul(2)
                .add(mod(p.y, 2).mul(3)),
            4,
        );
    const bayer4 = (pixel: Node<d.vec2f>) => {
        const cell = floor(pixel).toVar('cell');
        return bayer2(cell)
            .mul(4)
            .add(bayer2(floor(cell.div(2))))
            .add(0.5)
            .div(16);
    };

    const shadowAt = Fn(
        (world: Node<d.vec3f>, pixel: Node<d.vec2f>, stipple: Node<d.f32>) => {
            const clip = mul(lightViewProjection, vec4(world, 1));
            const ndc = clip.xyz.div(clip.w);
            const uv = vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(-0.5).add(0.5)).toVar('uv');
            const texel = uv.mul(SHADOW_SIZE).toVar('texel');
            const z = ndc.z.toVar('z');
            const at = (offset: Node<d.vec2f>) => shadowDepth.load(vec2i(floor(texel.add(offset))));

            const blockers = Var('blockers', f32(0));
            const blockerDepth = Var('blockerDepth', f32(0));
            for (const [x, y] of SHADOW_SEARCH) {
                const depth = at(vec2(x * SHADOW_SEARCH_RADIUS, y * SHADOW_SEARCH_RADIUS));
                If(depth.lessThan(z.sub(SHADOW_BIAS)), () => {
                    blockers.assign(blockers.add(1));
                    blockerDepth.assign(blockerDepth.add(depth));
                });
            }
            const lit = Var('lit', f32(1));
            If(blockers.greaterThan(0), () => {
                const height = max(z.sub(blockerDepth.div(blockers)), 0)
                    .mul(SHADOW_DEPTH_RANGE)
                    .toVar('height');
                const radius = add(1, height.mul(SHADOW_PENUMBRA_TEXELS_PER_UNIT)).toVar('radius');
                const bias = mul(SHADOW_BIAS, add(1, radius.mul(0.5)));
                const covered = Var('covered', f32(0));
                for (const [x, y] of SHADOW_FILTER_DISC) {
                    If(at(vec2(x, y).mul(radius)).lessThan(z.sub(bias)), () => covered.assign(covered.add(1)));
                }
                const strength = mix(1, HIGH_CASTER_SHADOW_STRENGTH, smoothstep(0, HIGH_CASTER_HEIGHT, height));
                lit.assign(sub(1, covered.div(SHADOW_FILTER_DISC.length).mul(strength)));
            });

            const roundOrDither = mix(0.5, bayer4(pixel), stipple);
            const litHalfOrDark = clamp(floor(lit.mul(2).add(roundOrDither)).div(2), 0, 1);

            const outsideLightFrustum = or(
                or(lessThan(uv.x, 0), greaterThan(uv.x, 1)),
                or(lessThan(uv.y, 0), greaterThan(uv.y, 1)),
            );
            return select(litHalfOrDark, 1, outsideLightFrustum);
        },
        {
            name: 'shadowAt',
            params: [
                { name: 'world', type: d.vec3f },
                { name: 'pixel', type: d.vec2f },
                { name: 'stipple', type: d.f32 },
            ],
            return: d.f32,
        },
    );

    const wobbledByRipples = (world: Node<d.vec3f>) => {
        const depth = max(world.y.negate(), 0);
        const lightEntry = world.xz.add(vec2(light[0] / light[1], light[2] / light[1]).mul(depth));
        const slope = sh.ripple(lightEntry);
        return world.add(vec3f(slope.y, 0, slope.z).mul(depth.mul(UNDERWATER_SHADOW_WOBBLE)));
    };

    const position = attribute('position', d.vec3f);
    const normal = attribute('normal', d.vec3f);
    const color = attribute('color', d.vec4f);

    const surfaceMaterials = (world: Node<d.vec3f>, worldNormal: Node<d.vec3f>) => {
        const clipPosition = mul(cameraProjectionMatrix, mul(cameraViewMatrix, vec4(world, 1)));
        const vWorld = varying(world, 'vWorld');
        const vNormal = varying(worldNormal, 'vNormal');
        const vColor = varying(color, 'vColor');

        const n = normalize(vNormal);
        const isFloor = vColor.w.mul(255).sub(ID_FLOOR).abs().lessThan(0.5);
        const stipple = select(f32(0), 1, isFloor);
        const shade = shadowAt(wobbledByRipples(vWorld.add(n.mul(0.04))), screenCoordinate, stipple);
        const lit = sh.surface(vColor, vWorld, n, shade, uniform(uniforms.time));
        const viewNormal = mul(cameraViewMatrix, vec4(n, 0)).xyz;
        const output = mrt({
            output: vec4(lit, vColor.w),
            normal: vec4(viewNormal, 1),
        });
        return {
            output,
            material: createMaterial({ vertex: clipPosition, fragment: output }),
            shadow: createMaterial({
                vertex: clipPosition,
                fragment: undefined,
                depthBias: 2,
                depthBiasSlopeScale: 2,
            }),
        };
    };

    const statics = surfaceMaterials(mul(modelWorldMatrix, vec4(position, 1)).xyz, normalize(mul(modelNormalMatrix, normal)));
    const material = statics.material;
    const shadowMaterial = statics.shadow;
    const gbufferOutput = statics.output;

    const duckTransforms = createDataTexture(new Float32Array(MAX_DUCKS * 8), MAX_DUCKS * 2, 1, {
        format: 'rgba32float',
        magFilter: 'nearest',
        minFilter: 'nearest',
    });
    duckTransforms.name = 'duckTransforms';
    const transforms = texture(duckTransforms);
    const texel = i32(instanceIndex).mul(2);
    const duckAt = transforms.load(vec2i(texel, 0));
    const duckRotation = transforms.load(vec2i(texel.add(1), 0));
    const rotate = (q: Node<d.vec4f>, v: Node<d.vec3f>) => v.add(cross(q.xyz, cross(q.xyz, v).add(v.mul(q.w))).mul(2));
    const instanced = surfaceMaterials(rotate(duckRotation, position).add(duckAt.xyz), rotate(duckRotation, normal));

    const floor_ = new Mesh(geometry(pond.floor), material);
    const rocks = pond.rocks.map((rock) => new Mesh(geometry(rock.mesh), material));
    const pads = pond.pads.map((pad) => {
        const mesh = new Mesh(geometry(pad.mesh), material);
        mesh.position[0] = pad.x;
        mesh.position[2] = pad.z;
        return mesh;
    });
    const ducks = new Mesh(geometry(duck), instanced.material);
    ducks.frustumCulled = false;
    ducks.count = 0;
    for (const mesh of [floor_, ...rocks]) mesh.updateWorldMatrix();

    const gbuffer = createRenderTarget(cam.artWidth, cam.artHeight, {
        count: 2,
        colorFormat: 'rgba8unorm',
        depthFormat: 'depth32float',
        depthSampled: true,
        clearColor: [0.1, 0.3, 0.5, 0],
    });
    gbuffer.textures[1].format = 'rgba16float';
    gbuffer.textures[1].name = 'normal';

    const colorTex = texture(gbuffer.textures[0] as Texture);
    const normalTex = texture(gbuffer.textures[1] as Texture);
    const depthTex = depthTexture(gbuffer.depthTexture!);
    const coord = vec2i(screenCoordinate);
    const sceneAt = (c: Node<d.vec2i>) =>
        depthTex
            .load(c)
            .mul(FAR - NEAR)
            .add(NEAR);

    const halfSize = uniform(uniforms.halfSize);
    const ndc = screenCoordinate.div(halfSize).sub(1);
    const halfWorld = halfSize.div(PIXELS_PER_UNIT);
    const camRight = uniform(uniforms.camRight);
    const camUp = uniform(uniforms.camUp);
    const rayOrigin = uniform(uniforms.camPosition)
        .add(camRight.mul(ndc.x.mul(halfWorld.x)))
        .add(camUp.mul(ndc.y.negate().mul(halfWorld.y)));
    const time = uniform(uniforms.time);
    const hit = sh.waterHit(rayOrigin, time);

    const slope = sh.ripple(hit.xz);
    const tilt = vec3f(slope.y.negate(), 0, slope.z.negate());
    const shift = vec2(dot(tilt, camRight), dot(tilt, camUp).negate()).mul(REFRACTION_ART_PIXELS_PER_SLOPE);
    const shifted = coord.add(vec2i(floor(shift.add(0.5))));
    const bothUnderWater = sceneAt(coord).greaterThanEqual(hit.w).and(sceneAt(shifted).greaterThanEqual(hit.w));
    const under = select(coord, shifted, bothUnderWater);
    const at = (dx: number, dy: number) => under.add(vec2i(dx, dy));

    const outlined = sh.outline(
        colorTex.load(under),
        normalTex.load(under).xyz,
        depthTex.load(under),
        [at(1, 0), at(-1, 0), at(0, 1), at(0, -1)].map((c) => ({
            depth: depthTex.load(c),
            normal: normalTex.load(c).xyz,
        })),
    );
    const partWayDown = rayOrigin.add(
        vec3f(cam.forward[0], cam.forward[1], cam.forward[2]).mul(
            hit.w.add(min(sceneAt(under).sub(hit.w), 1.2).mul(0.5)),
        ),
    );
    const water = sh.shadeWater(
        outlined,
        sceneAt(under),
        colorTex.load(under).w.mul(255),
        rayOrigin,
        hit,
        shadowAt(hit.xyz, screenCoordinate, f32(0)),
        shadowAt(wobbledByRipples(partWayDown), screenCoordinate, f32(1)),
        time,
    );
    const sprayed = sh.spray(water, rayOrigin, min(sceneAt(coord), hit.w));
    const compositeMesh = fullscreen(vec4(sh.quantize(sprayed), 1));
    const composite = createRenderTarget(cam.artWidth, cam.artHeight, {
        colorFormat: 'rgba8unorm',
        depthBuffer: false,
    });

    const compositeTex = texture(composite.texture as Texture);
    const src = vec2i(floor(screenCoordinate.add(uniform(uniforms.upscaleShift)).div(uniform(uniforms.artPixelSize)))).add(vec2i(1, 1));
    const upscaleMesh = fullscreen(vec4(compositeTex.load(src).xyz, 1));

    return {
        renderer,
        view,
        gbuffer,
        composite,
        shadow,
        lightCamera,
        light,
        shadowMaterial,
        duckShadowMaterial: instanced.shadow,
        gbufferMrt: gbufferOutput,
        floor: floor_,
        rocks,
        pads,
        ducks,
        compositeMesh,
        upscaleMesh,
        ripples: rippleTexture,
        duckTransforms,
        droplets,
        uniforms,
    };
}

export function resize(render: Render, cam: PixelCamera, cssWidth: number, cssHeight: number, pixelRatio: number): void {
    render.view.setPixelRatio(pixelRatio);
    render.view.setSize(cssWidth, cssHeight);
    render.gbuffer.setSize(cam.artWidth, cam.artHeight);
    render.composite.setSize(cam.artWidth, cam.artHeight);
}

const _lightPosition = vec3.create();
const _heading = vec3.create();
const X_AXIS: Vec3 = [1, 0, 0];
const _lightViewProjection = mat4.create();

function updateLight(render: Render, cam: PixelCamera): void {
    const { lightCamera, light } = render;
    const texelSize = (SHADOW_EXTENT * 2) / SHADOW_SIZE;
    const snappedTarget: Vec3 = [Math.round(cam.target[0] / texelSize) * texelSize, 0, Math.round(cam.target[2] / texelSize) * texelSize];
    vec3.scaleAndAdd(_lightPosition, snappedTarget, light, 40);
    vec3.copy(lightCamera.position, _lightPosition);
    lightCamera.lookAt(snappedTarget);
    lightCamera.updateWorldMatrix();
    lightCamera.updateViewMatrix();
    mat4.multiply(_lightViewProjection, lightCamera.projectionMatrix, lightCamera.matrixWorldInverse);
    render.uniforms.lightViewProjection.value = _lightViewProjection;
}

export function draw(render: Render, cam: PixelCamera, physics: Physics): void {
    const { renderer, uniforms } = render;
    const time = physics.time;

    updateLight(render, cam);

    uniforms.time.value = time;
    uniforms.camPosition.value = [...cam.camera.position] as Vec3;
    uniforms.camRight.value = [...cam.right] as Vec3;
    uniforms.camUp.value = [...cam.up] as Vec3;
    uniforms.halfSize.value = [cam.artWidth / 2, cam.artHeight / 2];
    uniforms.upscaleShift.value = [cam.upscaleShift[0], cam.upscaleShift[1]];
    uniforms.artPixelSize.value = cam.artPixelSize;

    const transformData = render.duckTransforms.data as Float32Array;
    const duckData = uniforms.ducks.value as Float32Array;
    let floating = 0;
    for (let i = 0; i < physics.ducks.length; i++) {
        const duck = physics.ducks[i];
        transformData.set(duck.position, i * 8);
        transformData.set(duck.quaternion, i * 8 + 4);

        const water = surface(physics, duck.position[0], duck.position[2]);
        if (Math.abs(duck.position[1] - water) > DUCK_FOAM_MAX_DISTANCE_FROM_WATER) continue;
        vec3.transformQuat(_heading, X_AXIS, duck.quaternion);
        _heading[1] = 0;
        vec3.normalize(_heading, _heading);
        duckData.set([duck.position[0], duck.position[2], _heading[0], _heading[2]], floating * 4);
        floating++;
    }
    render.duckTransforms.needsUpdate = true;
    render.ducks.count = physics.ducks.length;
    uniforms.ducks.value = duckData;
    uniforms.duckCount.value = floating;

    const splashData = uniforms.splashes.value as Float32Array;
    splashData.fill(0);
    physics.splashes.forEach((s, i) => splashData.set([s.x, s.z, s.time, s.strength], i * 4));
    uniforms.splashes.value = splashData;

    const dropletData = render.droplets.data as Float32Array;
    dropletData.fill(0);
    physics.spray.droplets.forEach(({ position }, i) => dropletData.set([position[0], position[1], position[2], 1], i * 4));
    render.droplets.needsUpdate = true;

    render.pads.forEach((mesh, i) => {
        vec3.copy(mesh.position, physics.pads[i].position);
        quat.copy(mesh.quaternion, physics.pads[i].quaternion);
        mesh.updateWorldMatrix();
    });

    ripples.pack(physics.ripples, render.ripples.data as Float32Array);
    render.ripples.needsUpdate = true;

    const casters = [...render.rocks, ...render.pads];

    const f = frame(renderer);

    const shadowPass = f.pass({
        target: render.shadow,
        camera: render.lightCamera,
        label: 'shadow',
    });
    shadowPass.draw(render.ducks, { material: render.duckShadowMaterial });
    for (const mesh of casters) shadowPass.draw(mesh, { material: render.shadowMaterial });
    shadowPass.end();

    const gbufferPass = f.pass({
        target: render.gbuffer,
        camera: cam.camera,
        mrt: render.gbufferMrt,
        label: 'gbuffer',
    });
    gbufferPass.draw(render.floor);
    gbufferPass.draw(render.ducks);
    for (const mesh of casters) gbufferPass.draw(mesh);
    gbufferPass.end();

    const compositePass = f.pass({
        target: render.composite,
        label: 'composite',
    });
    compositePass.draw(render.compositeMesh);
    compositePass.end();

    const upscalePass = f.pass({ target: render.view, label: 'upscale' });
    upscalePass.draw(render.upscaleMesh);
    upscalePass.end();

    f.submit();
}
