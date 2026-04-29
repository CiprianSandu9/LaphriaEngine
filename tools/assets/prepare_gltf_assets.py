#!/usr/bin/env python3
"""
Prepare runtime-friendly glTF assets for LaphriaEngine.

Workflow (speed-first default):
1) meshopt geometry compression
2) KTX2 ETC1S for all texture slots
3) KTX2 UASTC override for color-critical slots
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(">", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def resolve_gltf_transform_prefix() -> list[str]:
    direct = shutil.which("gltf-transform")
    if direct is not None and Path(direct).exists():
        return [direct]

    direct_cmd = shutil.which("gltf-transform.cmd")
    if direct_cmd is not None and Path(direct_cmd).exists():
        return [direct_cmd]

    npx = shutil.which("npx")
    if npx is not None and Path(npx).exists():
        # Fallback that does not require global installation.
        return [npx, "--yes", "@gltf-transform/cli"]

    raise RuntimeError(
        "Could not find glTF-Transform CLI. Install Node.js LTS and then either:\n"
        "  1) npm i -g @gltf-transform/cli\n"
        "  2) ensure npx is available in PATH"
    )


def build_pipeline(cli_prefix: list[str], input_path: Path, output_path: Path, tmp_dir: Path, uastc_slots: str) -> list[list[str]]:
    step_meshopt = tmp_dir / "step_meshopt.glb"
    step_etc1s = tmp_dir / "step_etc1s.glb"
    commands = [
        cli_prefix + [
            "meshopt",
            str(input_path),
            str(step_meshopt),
        ],
        cli_prefix + [
            "etc1s",
            str(step_meshopt),
            str(step_etc1s),
            "--quality",
            "255",
        ],
    ]

    if uastc_slots.strip().lower() in {"", "none", "off"}:
        commands.append(
            cli_prefix + [
                "copy",
                str(step_etc1s),
                str(output_path),
            ]
        )
    else:
        commands.append(
            cli_prefix + [
                "uastc",
                str(step_etc1s),
                str(output_path),
                "--slots",
                uastc_slots,
            ]
        )

    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare glTF runtime assets (meshopt + hybrid KTX2).")
    parser.add_argument("--input", required=True, help="Source GLB/GLTF path (for example: Assets/sponza_source.glb)")
    parser.add_argument("--output", required=True, help="Runtime GLB path (for example: Assets/sponza_runtime.glb)")
    parser.add_argument(
        "--uastc-slots",
        default="baseColorTexture,normalTexture,emissiveTexture",
        help=(
            "Comma-separated texture slots to re-encode as UASTC after ETC1S pass. "
            "Use 'none' to keep ETC1S for all textures. "
            "Default: baseColorTexture,normalTexture,emissiveTexture"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        cli_prefix = resolve_gltf_transform_prefix()
        with tempfile.TemporaryDirectory(prefix="laphria_asset_prep_") as tmp_dir:
            commands = build_pipeline(cli_prefix, input_path, output_path, Path(tmp_dir), args.uastc_slots)
            print("Laphria asset prep profile: speed-first hybrid")
            print("  - ETC1S pass: all textures")
            if args.uastc_slots.strip().lower() in {"", "none", "off"}:
                print("  - UASTC override: disabled")
            else:
                print(f"  - UASTC override slots: {args.uastc_slots}")
            for command in commands:
                run_cmd(command, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Runtime asset ready: {output_path}")
    except subprocess.CalledProcessError as err:
        print(f"Command failed with exit code {err.returncode}: {' '.join(err.cmd)}", file=sys.stderr)
        return err.returncode
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 3
    except FileNotFoundError as err:
        print(f"Failed to launch conversion tool: {err}", file=sys.stderr)
        print("Tip: install Node.js LTS and run: npm i -g @gltf-transform/cli", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
