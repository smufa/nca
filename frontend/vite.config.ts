import { defineConfig, type Plugin } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

// The exported ONNX models live in ../exports (outside the Vite root). Vite
// ignores a publicDir outside the project root in dev, so we serve it ourselves
// at /models and copy it into the build output.
const exportsDir = fileURLToPath(new URL('../exports', import.meta.url));
const MOUNT = '/models';

const MIME: Record<string, string> = {
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.png': 'image/png',
  '.gif': 'image/gif',
};

function ncaModels(): Plugin {
  let root = process.cwd();
  let outDir = 'dist';

  return {
    name: 'nca-models',

    configResolved(config) {
      root = config.root;
      outDir = config.build.outDir;
    },

    configureServer(server) {
      server.middlewares.use(MOUNT, (req, res, next) => {
        const rel = decodeURIComponent((req.url || '/').split('?')[0]);
        const file = path.resolve(exportsDir, '.' + rel);
        if (!file.startsWith(path.resolve(exportsDir))) return next();
        if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return next();
        res.setHeader(
          'Content-Type',
          MIME[path.extname(file)] ?? 'application/octet-stream',
        );
        fs.createReadStream(file).pipe(res);
      });
    },

    closeBundle() {
      const dest = path.resolve(root, outDir, 'models');
      fs.rmSync(dest, { recursive: true, force: true });
      fs.cpSync(exportsDir, dest, { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [ncaModels()],
  server: { port: 5173, host: true },
  build: { target: 'es2022' },
});
