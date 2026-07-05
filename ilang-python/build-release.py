#!/usr/bin/env python3
"""Build and test the ilang-python release wheel locally.

This script does not publish anything. It:
  1. cleans previous Python build output and bundled native libs
  2. builds the Rust i-core cdylib in release mode
  3. copies the platform native library into src/ilang/
  4. builds a platform wheel
  5. verifies the wheel contains the native library
  6. runs twine check
  7. installs the wheel into a temporary venv with --no-deps
  8. smoke-tests `import ilang`
  9. optionally runs pytest against this package's tests
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


PY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PY_DIR.parent
SRC_ILANG = PY_DIR / "src" / "ilang"
DIST_DIR = PY_DIR / "dist"
BUILD_DIR = PY_DIR / "build"
NATIVE_NAMES = ("libi_core.so", "libi_core.dylib")


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def rm_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def python_in_venv(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def native_artifact_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libi_core.dylib"
    if system == "Linux":
        return "libi_core.so"
    raise SystemExit(f"unsupported platform for release build: {system}; expected macOS or Linux")


def clean() -> None:
    print("==> Cleaning previous build output")
    rm_path(DIST_DIR)
    rm_path(BUILD_DIR)
    for egg_info in PY_DIR.glob("*.egg-info"):
        rm_path(egg_info)
    for name in NATIVE_NAMES:
        rm_path(SRC_ILANG / name)


def build_rust() -> Path:
    print("==> Building Rust core")
    run(["cargo", "build", "-p", "i-core", "--release"], cwd=REPO_ROOT)
    name = native_artifact_name()
    artifact = REPO_ROOT / "target" / "release" / name
    if not artifact.exists():
        raise SystemExit(f"expected Rust artifact not found: {artifact}")
    return artifact


def copy_native_lib(artifact: Path) -> Path:
    print("==> Copying native library into Python package")
    dest = SRC_ILANG / artifact.name
    shutil.copy2(artifact, dest)
    print(f"copied {artifact} -> {dest}")
    return dest


def build_wheel() -> Path:
    print("==> Building wheel")
    run([sys.executable, "-m", "build", "--wheel"], cwd=PY_DIR)
    wheels = sorted(DIST_DIR.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {DIST_DIR}, found {len(wheels)}")
    wheel = retag_platform_wheel(wheels[0])
    print(f"built {wheel}")
    return wheel


def current_platform_tag() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        machine = "x86_64"

    if system == "Darwin":
        if machine not in {"x86_64", "arm64"}:
            raise SystemExit(f"unsupported macOS architecture: {machine}")
        version = os.environ.get("MACOSX_DEPLOYMENT_TARGET") or platform.mac_ver()[0]
        major, minor, *_ = version.split(".") + ["0"]
        # For macOS 11+, wheel tags use major.0, not the actual OS minor
        # version. For example, macOS 14.8 must be tagged macosx_14_0_arm64.
        if int(major) >= 11:
            minor = "0"
        return f"macosx_{major}_{minor}_{machine}"

    if system == "Linux":
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
        if machine == "x86_64":
            return "linux_x86_64"
        raise SystemExit(f"unsupported Linux architecture: {machine}")

    raise SystemExit(f"unsupported platform for release build: {system}; expected macOS or Linux")


def retag_platform_wheel(wheel: Path) -> Path:
    """Convert Hatch's py3-none-any wheel into py3-none-PLATFORM.

    Hatchling does not infer platform-specific tags from ctypes-loaded shared
    libraries. The wheel contents are already correct; this fixes the wheel
    filename and WHEEL metadata so installers do not treat it as universal.
    """
    old_tag = "py3-none-any"
    new_tag = f"py3-none-{current_platform_tag()}"

    if old_tag not in wheel.name:
        return wheel

    new_wheel = wheel.with_name(wheel.name.replace(old_tag, new_tag))
    print(f"retagging wheel {old_tag} -> {new_tag}")

    with tempfile.TemporaryDirectory(prefix="ilang-python-retag-") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(tmpdir)

        wheel_metadata_files = list(tmpdir.glob("*.dist-info/WHEEL"))
        if len(wheel_metadata_files) != 1:
            raise SystemExit(f"expected exactly one WHEEL metadata file, found {len(wheel_metadata_files)}")

        metadata = wheel_metadata_files[0]
        text = metadata.read_text()
        text = re.sub(r"^Root-Is-Purelib: true$", "Root-Is-Purelib: false", text, flags=re.MULTILINE)
        text = re.sub(r"^Tag: py3-none-any$", f"Tag: {new_tag}", text, flags=re.MULTILINE)
        metadata.write_text(text)
        rewrite_record(tmpdir)

        with zipfile.ZipFile(new_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(tmpdir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmpdir).as_posix())

    wheel.unlink()
    return new_wheel


def rewrite_record(root: Path) -> None:
    record_files = list(root.glob("*.dist-info/RECORD"))
    if len(record_files) != 1:
        raise SystemExit(f"expected exactly one RECORD metadata file, found {len(record_files)}")

    record = record_files[0]
    rows: list[list[str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path == record:
            rows.append([rel, "", ""])
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([rel, f"sha256={digest}", str(len(data))])

    with record.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def verify_wheel(wheel: Path, native_name: str) -> None:
    print("==> Verifying wheel contents")
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    expected_native = f"ilang/{native_name}"
    if expected_native not in names:
        raise SystemExit(f"wheel is missing {expected_native}")

    license_files = [name for name in names if name.endswith(".dist-info/licenses/LICENSE") or name.endswith(".dist-info/LICENSE")]
    if not license_files:
        print("warning: did not find LICENSE in wheel metadata")

    if "none-any" in wheel.name:
        raise SystemExit(f"wheel was tagged as pure/universal, but contains a native library: {wheel.name}")

    print(f"verified {expected_native} in {wheel.name}")


def twine_check() -> None:
    print("==> Running twine check")
    run([sys.executable, "-m", "twine", "check", *map(str, DIST_DIR.glob("*"))], cwd=PY_DIR)


def test_wheel(wheel: Path, *, run_pytest: bool) -> None:
    print("==> Testing wheel in temporary venv")
    with tempfile.TemporaryDirectory(prefix="ilang-python-wheel-test-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        py = python_in_venv(venv_dir)

        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "--no-deps", str(wheel)])
        run([str(py), "-c", "import ilang; print('ilang import ok')"])

        if run_pytest:
            run([str(py), "-m", "pip", "install", "pytest"])
            run([str(py), "-m", "pytest", str(PY_DIR / "tests")], cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and test the ilang-python wheel locally; does not publish.")
    parser.add_argument("--no-clean", action="store_true", help="do not clean dist/build/native libs before building")
    parser.add_argument("--skip-twine", action="store_true", help="skip `python -m twine check dist/*`")
    parser.add_argument("--skip-test", action="store_true", help="skip clean-venv install/import test")
    parser.add_argument("--pytest", action="store_true", help="install pytest in the test venv and run ilang-python/tests")
    parser.add_argument("--keep-native", action="store_true", help="leave copied native library in src/ilang after build")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied_native: Path | None = None
    try:
        if not args.no_clean:
            clean()
        artifact = build_rust()
        copied_native = copy_native_lib(artifact)
        wheel = build_wheel()
        verify_wheel(wheel, artifact.name)
        if not args.skip_twine:
            twine_check()
        if not args.skip_test:
            test_wheel(wheel, run_pytest=args.pytest)
        print(f"==> Success: {wheel}")
    finally:
        if copied_native is not None and not args.keep_native:
            print(f"==> Removing temporary native library {copied_native}")
            rm_path(copied_native)


if __name__ == "__main__":
    main()
