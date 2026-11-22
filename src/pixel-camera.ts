import { OrthographicCamera } from 'gpucat';
import { quat, type Vec3, vec3 } from 'math';

export const PIXELS_PER_UNIT = 15;
export const NEAR = 1;
export const FAR = 80;
const DISTANCE = 40;
const PITCH = (-34 * Math.PI) / 180;
const YAW = (40 * Math.PI) / 180;
const CSS_PIXELS_PER_ART_PIXEL = 3;
const CSS_PIXELS_PER_ART_PIXEL_ON_PHONES = 2.5;
const PHONE_MAX_CSS_SIZE = 600;
const MAX_ART_WIDTH = 660;
const MAX_ART_HEIGHT = 400;
const SNAP_MARGIN = 1;
const ART_PIXEL_WORLD_SIZE = 1 / PIXELS_PER_UNIT;

export type PixelCamera = {
    camera: OrthographicCamera;
    target: Vec3;
    right: Vec3;
    up: Vec3;
    forward: Vec3;
    artPixelSize: number;
    artWidth: number;
    artHeight: number;
    upscaleShift: [number, number];
};

export function init(): PixelCamera {
    const q = quat.create();
    quat.fromEuler(q, [PITCH, YAW, 0, 'yxz']);

    const camera = new OrthographicCamera(-1, 1, 1, -1, NEAR, FAR);
    quat.copy(camera.quaternion, q);

    return {
        camera,
        target: [0, 0, 0],
        right: vec3.transformQuat(vec3.create(), [1, 0, 0], q),
        up: vec3.transformQuat(vec3.create(), [0, 1, 0], q),
        forward: vec3.transformQuat(vec3.create(), [0, 0, -1], q),
        artPixelSize: 1,
        artWidth: 2,
        artHeight: 2,
        upscaleShift: [0, 0],
    };
}

export function resize(cam: PixelCamera, cssWidth: number, cssHeight: number, pixelRatio: number): void {
    const isPhone = Math.min(cssWidth, cssHeight) < PHONE_MAX_CSS_SIZE;
    const cssArtPixel = isPhone ? CSS_PIXELS_PER_ART_PIXEL_ON_PHONES : CSS_PIXELS_PER_ART_PIXEL;
    const deviceWidth = cssWidth * pixelRatio;
    const deviceHeight = cssHeight * pixelRatio;
    cam.artPixelSize = Math.max(1, Math.round(cssArtPixel * pixelRatio), Math.ceil(deviceWidth / MAX_ART_WIDTH), Math.ceil(deviceHeight / MAX_ART_HEIGHT));
    cam.artWidth = Math.ceil(deviceWidth / cam.artPixelSize) + SNAP_MARGIN * 2;
    cam.artHeight = Math.ceil(deviceHeight / cam.artPixelSize) + SNAP_MARGIN * 2;

    const halfW = cam.artWidth / 2 / PIXELS_PER_UNIT;
    const halfH = cam.artHeight / 2 / PIXELS_PER_UNIT;
    cam.camera.left = -halfW;
    cam.camera.right = halfW;
    cam.camera.top = halfH;
    cam.camera.bottom = -halfH;
    cam.camera.updateProjectionMatrix();
}

const _base = vec3.create();

export function update(cam: PixelCamera): void {
    vec3.scaleAndAdd(_base, cam.target, cam.forward, -DISTANCE);

    const right = vec3.dot(_base, cam.right) / ART_PIXEL_WORLD_SIZE;
    const up = vec3.dot(_base, cam.up) / ART_PIXEL_WORLD_SIZE;
    const snappedRight = Math.round(right);
    const snappedUp = Math.round(up);

    const position = cam.camera.position;
    vec3.scaleAndAdd(position, _base, cam.right, (snappedRight - right) * ART_PIXEL_WORLD_SIZE);
    vec3.scaleAndAdd(position, position, cam.up, (snappedUp - up) * ART_PIXEL_WORLD_SIZE);
    cam.camera.updateWorldMatrix();
    cam.camera.updateViewMatrix();

    cam.upscaleShift[0] = Math.round((right - snappedRight) * cam.artPixelSize);
    cam.upscaleShift[1] = Math.round((snappedUp - up) * cam.artPixelSize);
}

export function screenToWater(out: Vec3, cam: PixelCamera, deviceX: number, deviceY: number): Vec3 {
    const artX = (deviceX + cam.upscaleShift[0]) / cam.artPixelSize + SNAP_MARGIN - cam.artWidth / 2;
    const artY = cam.artHeight / 2 - ((deviceY + cam.upscaleShift[1]) / cam.artPixelSize + SNAP_MARGIN);
    vec3.scaleAndAdd(out, cam.camera.position, cam.right, artX * ART_PIXEL_WORLD_SIZE);
    vec3.scaleAndAdd(out, out, cam.up, artY * ART_PIXEL_WORLD_SIZE);
    const toWater = -out[1] / cam.forward[1];
    return vec3.scaleAndAdd(out, out, cam.forward, toWater);
}
