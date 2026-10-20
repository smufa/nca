import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

// The ONNX models + manifest live in ../exports and are served at the site root
// (e.g. /manifest.json, /smile-demo/model.onnx).
export default defineConfig({
  publicDir: fileURLToPath(new URL('../exports', import.meta.url)),
  server: { port: 5173, host: true },
  build: { target: 'es2022' },
});
