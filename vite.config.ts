import { defineConfig } from 'vite';

export default defineConfig({
    // relative, so the build works from any path (github pages serves it from /ducks/)
    base: './',
    build: { target: 'esnext' },
    // tumble.js finds its wasm relative to its own module, which the dep optimizer would move
    optimizeDeps: { exclude: ['tumble.js'] },
});
