# Deploying the playground

The frontend is a static site: one `index.html`, one JS bundle, and `models/`
(the committed `exports/` directory, 25 MB of ONNX + PNG). The flake builds that
directory and gives you three ways to serve it.

```bash
nix run github:smufa/nca            # http://127.0.0.1:8080
nix run .#playground                # same, from a checkout
PORT=9000 HOST=0.0.0.0 nix run .#playground
```

`HOST` defaults to `127.0.0.1` on purpose: the playground has no auth and no
rate limiting, so exposing it to a network is a decision, not a default.

## NixOS

```nix
{
  inputs.nca.url = "github:smufa/nca";

  outputs = { self, nixpkgs, nca, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nca.nixosModules.playground
        {
          services.nca-playground = {
            enable = true;
            port = 8080;
            openFirewall = true;      # or leave false behind a reverse proxy
          };
        }
      ];
    };
  };
}
```

That installs the built site into the store and runs one hardened systemd unit
(`DynamicUser`, `ProtectSystem=strict`, no capabilities above port 1024). Point
Caddy/nginx at `127.0.0.1:8080` for TLS:

```nix
services.caddy.virtualHosts."nca.example.org".extraConfig = ''
  reverse_proxy 127.0.0.1:8080
'';
```

## What the build needs, and what it does not

- **Needs network at build time**: the npm registry, for the four frontend
  dependencies (fetched as a fixed-output derivation).
- **Does not need Python, torch, ONNX Runtime, or a GPU**: the models are baked in
  as files; inference happens in the browser (WebGPU, or the WASM fallback).
- **Runs anywhere**: the output is `$out/share/nca-playground`, plain files. Copy
  it to any web server if you would rather not use the module.

The Python side of the repo is deliberately *not* in the flake's deployment
output: training needs a CUDA torch and a cluster, which is `slurm/`'s problem.
`nix develop` gives you `uv`, Python, Node and the Vulkan loader for local work.

## Maintenance: bumping the npm hash

`npmDeps` is a fixed-output derivation, so dependency changes fail the build with
the expected hash. After editing `frontend/package.json` (or `package-lock.json`):

```bash
nix build .#playground
# error: hash mismatch
#   specified: sha256-…
#   got:       sha256-…
# paste the "got" value over `hash = ...` in flake.nix
```

`nix flake check` builds the site, so a broken frontend is a red flake — that is
the deployment's entire test suite.

## Without Nix

`cd frontend && npm ci && npm run build` produces the same `dist/` (the Vite
plugin copies `../exports` into `dist/models`). Serve `dist/` with anything.
