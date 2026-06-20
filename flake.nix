{
  description = "ilang";

  inputs = {
    stable-pkgs.url = "github:nixos/nixpkgs?ref=nixos-26.05";
  };

  outputs =
    { self, stable-pkgs }:
    let
      system = "x86_64-linux";
      stable = import stable-pkgs {
        inherit system;
      };
    in
    {
      devShells.${system}.default = stable.mkShellNoCC {
        packages = with stable; [
          cargo
          clippy
          python314Packages.numpy
          ruff
          rust-analyzer
          rustc
          rustfmt
          ty
        ];
      };
    };
}
