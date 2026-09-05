{
  description = "ilang";

  inputs = {
    stable-pkgs.url = "github:nixos/nixpkgs?ref=nixos-26.05";
  };

  outputs =
    {
      self,
      stable-pkgs,
    }:
    let
      system = "x86_64-linux";
      stable = import stable-pkgs {
        inherit system;
      };
      py = stable.python314.withPackages (ps: with ps; [
        numpy
        torch
        torchvision
      ]);
    in
    {
      devShells.${system}.default = stable.mkShellNoCC {
        packages = with stable; [
          cargo
          clippy
          py
          ruff
          rust-analyzer
          rustc
          rustfmt
          ty
        ];
      };
    };
}
