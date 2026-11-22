import type { FromWorker, ToWorker } from './messages';

const canvas = document.querySelector<HTMLCanvasElement>('#pond')!;
const status = document.querySelector<HTMLElement>('#status')!;

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (message: ToWorker, transfer: Transferable[] = []) => worker.postMessage(message, transfer);

worker.onmessage = (event: MessageEvent<FromWorker>) => {
    const msg = event.data;
    if (msg.type === 'ready') document.body.classList.add('ready');
    if (msg.type === 'error') status.textContent = `the pond froze over: ${msg.message}`;
};

if (!('gpu' in navigator)) {
    status.textContent = 'this pond needs webgpu';
} else {
    const offscreen = canvas.transferControlToOffscreen();
    send({ type: 'init', canvas: offscreen, duckUrl: new URL('duck.json', document.baseURI).href, cssWidth: innerWidth, cssHeight: innerHeight, pixelRatio: devicePixelRatio }, [offscreen]);

    addEventListener('resize', () => send({ type: 'resize', cssWidth: innerWidth, cssHeight: innerHeight, pixelRatio: devicePixelRatio }));

    canvas.addEventListener('pointermove', (e) => send({ type: 'pointermove', deviceX: e.clientX * devicePixelRatio, deviceY: e.clientY * devicePixelRatio }));
    canvas.addEventListener('pointerdown', (e) => send({ type: 'pointerdown', deviceX: e.clientX * devicePixelRatio, deviceY: e.clientY * devicePixelRatio }));
    canvas.addEventListener('pointerleave', () => send({ type: 'pointerleave' }));
    canvas.addEventListener('pointerup', (e) => {
        const fingerLifted = e.pointerType !== 'mouse';
        if (fingerLifted) send({ type: 'pointerleave' });
    });
}
