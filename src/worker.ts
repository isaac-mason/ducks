import { clamp, type Vec3, vec3 } from 'math';
import { random } from 'math/random';
import type { FromWorker, ToWorker } from './messages';
import { loadDuck } from './mesh-data';
import * as physics from './physics';
import * as camera from './pixel-camera';
import { PIXELS_PER_UNIT } from './pixel-camera';
import * as pond from './pond';
import * as render from './render';

const FLOCK_PER_SQUARE_UNIT = 0.25;
const FLOCK_MIN = 40;
const DUCK_HALF_WIDTH_ON_SCREEN: [number, number] = [1.2, 1.6];

const post = (message: FromWorker) => self.postMessage(message);

function fitBoundsToScreen(bounds: physics.Bounds, cam: camera.PixelCamera): void {
    bounds.center[0] = cam.target[0];
    bounds.center[1] = cam.target[2];
    bounds.half[0] = cam.artWidth / 2 / PIXELS_PER_UNIT - DUCK_HALF_WIDTH_ON_SCREEN[0];
    bounds.half[1] = cam.artHeight / 2 / PIXELS_PER_UNIT / -cam.forward[1] - DUCK_HALF_WIDTH_ON_SCREEN[1];
}

function rainInFlock(sim: physics.Physics, bounds: physics.Bounds): void {
    const visibleArea = 4 * bounds.half[0] * bounds.half[1];
    const flock = clamp(Math.round(visibleArea * FLOCK_PER_SQUARE_UNIT), FLOCK_MIN, physics.MAX_DUCKS);
    for (let i = 0; i < flock; i++) {
        const a = random.float(Math.random, -bounds.half[0], bounds.half[0]);
        const b = random.float(Math.random, -bounds.half[1], bounds.half[1]);
        const x = bounds.center[0] + bounds.axes[0][0] * a + bounds.axes[1][0] * b;
        const z = bounds.center[1] + bounds.axes[0][1] * a + bounds.axes[1][1] * b;
        physics.addDuck(sim, x, random.float(Math.random, 0.5, 4.5), z);
    }
}

async function start(init: Extract<ToWorker, { type: 'init' }>) {
    const duck = await loadDuck(init.duckUrl);
    const scene = pond.init();
    const sim = await physics.init(duck, scene);
    const cam = camera.init();
    camera.resize(cam, init.cssWidth, init.cssHeight, init.pixelRatio);
    camera.update(cam);
    let reported = false;
    const gfx = await render.init(init.canvas, cam, duck, scene, (message) => {
        if (!reported) post({ type: 'error', message });
        reported = true;
    });
    render.resize(gfx, cam, init.cssWidth, init.cssHeight, init.pixelRatio);

    const ground = vec3.normalize(vec3.create(), [cam.forward[0], 0, cam.forward[2]]);
    const bounds: physics.Bounds = { center: [0, 0], axes: [[cam.right[0], cam.right[2]], [ground[0], ground[2]]], half: [1, 1] };
    fitBoundsToScreen(bounds, cam);
    rainInFlock(sim, bounds);

    const point: Vec3 = [0, 0, 0];

    self.onmessage = (event: MessageEvent<ToWorker>) => {
        const msg = event.data;
        switch (msg.type) {
            case 'resize':
                camera.resize(cam, msg.cssWidth, msg.cssHeight, msg.pixelRatio);
                render.resize(gfx, cam, msg.cssWidth, msg.cssHeight, msg.pixelRatio);
                fitBoundsToScreen(bounds, cam);
                break;
            case 'pointermove':
                camera.screenToWater(point, cam, msg.deviceX, msg.deviceY);
                physics.movePointer(sim, point[0], point[2]);
                break;
            case 'pointerdown':
                camera.screenToWater(point, cam, msg.deviceX, msg.deviceY);
                physics.pop(sim, point[0], point[2]);
                break;
            case 'pointerleave':
                physics.releasePointer(sim);
                break;
        }
    };

    let last = performance.now();
    let drawn = false;
    const loop = () => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;

        try {
            physics.update(sim, dt, bounds);
            camera.update(cam);
            render.draw(gfx, cam, sim);
            if (!drawn) post({ type: 'ready' });
            drawn = true;
        } catch (error) {
            console.error(error);
            post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
            return;
        }
        self.requestAnimationFrame(loop);
    };
    self.requestAnimationFrame(loop);
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
    if (event.data.type !== 'init') return;
    start(event.data).catch((error: unknown) => {
        console.error(error);
        post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    });
};
