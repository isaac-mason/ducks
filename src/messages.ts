export type ToWorker =
    | { type: 'init'; canvas: OffscreenCanvas; duckUrl: string; cssWidth: number; cssHeight: number; pixelRatio: number }
    | { type: 'resize'; cssWidth: number; cssHeight: number; pixelRatio: number }
    | { type: 'pointermove'; deviceX: number; deviceY: number }
    | { type: 'pointerdown'; deviceX: number; deviceY: number }
    | { type: 'pointerleave' };

export type FromWorker = { type: 'ready' } | { type: 'error'; message: string };
