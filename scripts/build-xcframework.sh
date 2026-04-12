#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Release}"
OUT_DIR="${1:-"$ROOT_DIR/.build/xcframework"}"
BUILD_ROOT="$OUT_DIR/build"
HEADERS_DIR="$OUT_DIR/Headers"
XCFRAMEWORK="$OUT_DIR/CLibrosa.xcframework"

MACOS_ARCHS="${MACOS_ARCHS:-$(uname -m)}"
IOS_ARCHS="${IOS_ARCHS:-arm64}"
IOS_SIM_ARCHS="${IOS_SIM_ARCHS:-$(uname -m)}"
VISIONOS_ARCHS="${VISIONOS_ARCHS:-arm64}"
VISIONOS_SIM_ARCHS="${VISIONOS_SIM_ARCHS:-$(uname -m)}"
REQUESTED_PLATFORMS="${LIBROSA_XCFRAMEWORK_PLATFORMS:-macos ios ios-simulator visionos visionos-simulator}"

DESTINATIONS=(
  "macos|Darwin|macosx|$MACOS_ARCHS|12.0"
  "ios|iOS|iphoneos|$IOS_ARCHS|13.0"
  "ios-simulator|iOS|iphonesimulator|$IOS_SIM_ARCHS|13.0"
  "visionos|visionOS|xros|$VISIONOS_ARCHS|1.0"
  "visionos-simulator|visionOS|xrsimulator|$VISIONOS_SIM_ARCHS|1.0"
)

rm -rf "$OUT_DIR"
mkdir -p "$BUILD_ROOT" "$HEADERS_DIR"

cp "$ROOT_DIR/Sources/CLibrosa/include/librosa_c.h" "$HEADERS_DIR/librosa_c.h"
cat > "$HEADERS_DIR/module.modulemap" <<'MODULEMAP'
module CLibrosa {
  umbrella header "librosa_c.h"
  export *
}
MODULEMAP

library_args=()

should_build_platform() {
  local platform="$1"
  [[ " $REQUESTED_PLATFORMS " == *" $platform "* ]]
}

build_slice() {
  local name="$1"
  local system_name="$2"
  local sdk_name="$3"
  local archs="$4"
  local deployment_target="$5"
  local build_dir="$BUILD_ROOT/$name"
  local combined="$OUT_DIR/libCLibrosa-$name.a"

  cmake -S "$ROOT_DIR" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE="$CONFIGURATION" \
    -DCMAKE_SYSTEM_NAME="$system_name" \
    -DCMAKE_OSX_SYSROOT="$(xcrun --sdk "$sdk_name" --show-sdk-path)" \
    -DCMAKE_OSX_ARCHITECTURES="$archs" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DLIBROSA_BUILD_TESTS=OFF \
    -DLIBROSA_BUILD_CLI=OFF \
    -DLIBROSA_BUILD_SWIFT_C_WRAPPER=ON \
    -DLIBROSA_USE_AUDIOTOOLBOX=ON \
    -DLIBROSA_FFT_BACKEND=accelerate

  cmake --build "$build_dir" --config "$CONFIGURATION" --target librosa_c

  /usr/bin/libtool -static -o "$combined" \
    "$build_dir/liblibrosa.a" \
    "$build_dir/libCLibrosa.a"

  library_args+=("-library" "$combined" "-headers" "$HEADERS_DIR")
}

for entry in "${DESTINATIONS[@]}"; do
  IFS='|' read -r name system_name sdk_name archs deployment_target <<< "$entry"
  if ! should_build_platform "$name"; then
    continue
  fi
  build_slice "$name" "$system_name" "$sdk_name" "$archs" "$deployment_target"
done

if [[ "${#library_args[@]}" -eq 0 ]]; then
  echo "error: no platforms selected by LIBROSA_XCFRAMEWORK_PLATFORMS" >&2
  exit 1
fi

xcodebuild -create-xcframework \
  "${library_args[@]}" \
  -output "$XCFRAMEWORK"

echo "$XCFRAMEWORK"
