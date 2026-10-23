{
  description = "Neural CA playground: the Growing-NCA frontend, built and servable";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      lib = nixpkgs.lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      # `node_modules` and a stale `dist` are never source: the deps come from the
      # fixed-output derivation, and `dist` is what we are building.
      frontendFiles = lib.fileset.difference ./frontend (
        lib.fileset.unions [
          (lib.fileset.maybeMissing ./frontend/node_modules)
          (lib.fileset.maybeMissing ./frontend/dist)
        ]
      );

      # The Vite build reads model files from `../exports` (see frontend/vite.config.ts),
      # so the build source keeps `frontend/` and `exports/` as siblings. The npm deps
      # are fetched from `frontend/` alone, which is where the lockfile is.
      playgroundSource = lib.fileset.toSource {
        root = ./.;
        fileset = lib.fileset.unions [
          ./exports
          frontendFiles
        ];
      };
      npmDepsSource = lib.fileset.toSource {
        root = ./frontend;
        fileset = frontendFiles;
      };

      # One static server for both `nix run` and the NixOS module: python's
      # ThreadingHTTPServer, because a playground needs no rewrite rules and the
      # alternative is pulling nginx in for one directory of files.
      mkServe =
        pkgs: site:
        pkgs.writeShellApplication {
          name = "nca-playground";
          runtimeInputs = [ pkgs.python3 ];
          text = ''
            port="''${PORT:-8080}"
            host="''${HOST:-127.0.0.1}"
            echo "NCA playground: http://$host:$port" >&2
            exec python -m http.server "$port" --bind "$host" --directory ${site}
          '';
        };
    in
    {
      packages = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          playground = pkgs.buildNpmPackage {
            pname = "nca-playground";
            version = "0.1.0";
            src = playgroundSource;
            # Build from `frontend/`: the lockfile is there, and `../exports` is then the
            # sibling directory the Vite plugin expects (see frontend/vite.config.ts).
            sourceRoot = "source/frontend";
            # Fetched from `frontend/` (the lockfile's home), while `src` above is the
            # whole tree: `buildNpmPackage` would look for the lockfile at the root.
            npmDeps = pkgs.fetchNpmDeps {
              src = npmDepsSource;
              hash = "sha256-g+/CAMHbqE3xSzrMwyzb/qeoCUzotmjGpjUpawlkZuc=";
            };
            # `tsc && vite build`; the plugin copies ../exports into dist/models.
            npmBuildScript = "build";
            doCheck = false;
            installPhase = ''
              runHook preInstall
              mkdir -p $out/share/nca-playground
              cp -r dist/. $out/share/nca-playground/
              runHook postInstall
            '';
            meta = {
              description = "Growing Neural CA playground (static site with the exported models)";
              platforms = lib.platforms.unix;
            };
          };
        in
        {
          playground = playground;
          # `nix run .#playground`: the site plus a one-file HTTP server.
          server = mkServe pkgs "${playground}/share/nca-playground";
          default = playground;
        }
      );

      apps = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          playground = {
            type = "app";
            program = "${self.packages.${system}.server}/bin/nca-playground";
            meta.description = "Serve the NCA playground on http://127.0.0.1:8080 (PORT/HOST to change)";
          };
          default = self.apps.${system}.playground;
        }
      );

      # `services.nca-playground.enable = true;` on a NixOS host.
      nixosModules.playground =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.nca-playground;
        in
        {
          options.services.nca-playground = {
            enable = lib.mkEnableOption "the NCA playground static site";
            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.playground;
              defaultText = lib.literalExpression "nca.packages.\${system}.playground";
              description = "The built playground (a directory of static files under share/nca-playground).";
            };
            host = lib.mkOption {
              type = lib.types.str;
              default = "0.0.0.0";
              description = "Address to bind. Use 127.0.0.1 behind a reverse proxy.";
            };
            port = lib.mkOption {
              type = lib.types.port;
              default = 8080;
              description = "TCP port to serve on.";
            };
            openFirewall = lib.mkOption {
              type = lib.types.bool;
              default = false;
              description = "Open the port in the host firewall.";
            };
          };

          config = lib.mkIf cfg.enable {
            systemd.services.nca-playground = {
              description = "NCA playground (static site)";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];
              serviceConfig = {
                ExecStart = "${pkgs.python3}/bin/python -m http.server ${toString cfg.port} --bind ${cfg.host} --directory ${cfg.package}/share/nca-playground";
                DynamicUser = true;
                Restart = "on-failure";
                RestartSec = 2;
                # A static file server needs nothing else. Hardening is worth it even
                # for a demo page: the service is reachable from the network.
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ProtectKernelTunables = true;
                ProtectControlGroups = true;
                RestrictNamespaces = true;
                RestrictSUIDSGID = true;
                LockPersonality = true;
                MemoryDenyWriteExecute = false; # python is fine, but keep it explicit
                SystemCallFilter = [ "@system-service" ];
                SystemCallErrorNumber = "EPERM";
                AmbientCapabilities = lib.optionals (cfg.port < 1024) [ "CAP_NET_BIND_SERVICE" ];
                CapabilityBoundingSet = lib.optionals (cfg.port < 1024) [ "CAP_NET_BIND_SERVICE" ];
              };
            };

            networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
          };
        };
      nixosModules.default = self.nixosModules.playground;

      # `nix develop`: what the Python/TS side of the repo needs. uv installs the
      # CPU torch wheel into a local `.venv`; Vulkan is here for scripts/bench.sh.
      devShells = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              pkgs.python3
              pkgs.nodejs_22
              pkgs.vulkan-loader
              pkgs.vulkan-headers
            ]
            ++ lib.optionals pkgs.stdenv.isLinux [ pkgs.mesa ];
            shellHook = ''
              echo "python $(python --version | cut -d' ' -f2)  node $(node --version)"
              echo "  uv sync --group dev   # python env (CPU torch)"
              echo "  cd frontend && npm ci  # web deps"
              echo "  bash scripts/verify.sh # everything"
            '';
          };
        }
      );

      formatter = lib.genAttrs systems (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);

      # `nix flake check` builds the site: a broken frontend is a red flake, which is
      # the only CI this repo needs for deployment.
      checks = lib.genAttrs systems (system: {
        playground = self.packages.${system}.playground;
      });
    };
}
