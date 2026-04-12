#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/release.sh <version>

Cuts and publishes a librosa.cpp release.

The script:
  1. verifies the working tree is clean and the current branch is main
  2. updates packages/librosa-wasm/package.json and package-lock.json
  3. commits the version bump if needed
  4. creates and pushes a v<version> git tag
  5. publishes a GitHub release
  6. waits for release asset and npm publish workflows
  7. downloads CLibrosa-<version>.xcframework.zip
  8. computes the SwiftPM binary-target checksum
  9. adds the checksum and binaryTarget snippet to the GitHub release notes

Example:
  scripts/release.sh 0.1.0

After completion, update librosa-swift with:
  cd /Users/oli/Dev/librosa-swift
  scripts/update-binary-target.sh <version>
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

run() {
  echo "+ $*"
  "$@"
}

workflow_run_id() {
  local workflow="$1"
  local commit_sha="$2"

  gh run list \
    --workflow "$workflow" \
    --event release \
    --commit "$commit_sha" \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId // empty'
}

wait_for_workflow() {
  local workflow="$1"
  local label="$2"
  local commit_sha="$3"

  echo "Waiting for ${label} workflow..."

  local run_id=""
  for _ in {1..60}; do
    run_id="$(workflow_run_id "$workflow" "$commit_sha")"
    if [[ -n "$run_id" ]]; then
      break
    fi
    sleep 10
  done

  [[ -n "$run_id" ]] || die "timed out waiting for ${label} workflow to start"

  run gh run watch "$run_id" --exit-status
}

replace_swift_release_notes_section() {
  local tag="$1"
  local asset_url="$2"
  local swift_checksum="$3"
  local release_sha256="$4"
  local notes_file="$5"
  local output_file="$6"

  gh release view "$tag" --json body --jq '.body // ""' > "$notes_file"

  python3 - "$notes_file" "$output_file" "$asset_url" "$swift_checksum" "$release_sha256" <<'PY'
import pathlib
import re
import sys

notes_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
asset_url = sys.argv[3]
swift_checksum = sys.argv[4]
release_sha256 = sys.argv[5]

body = notes_path.read_text().rstrip()
section = f"""<!-- librosa-swift-checksum:start -->
## Swift Package Binary Target

Use this checksum in the `librosa-swift` package manifest:

```swift
.binaryTarget(
    name: "CLibrosa",
    url: "{asset_url}",
    checksum: "{swift_checksum}"
)
```

Asset SHA-256: `{release_sha256}`
SwiftPM checksum: `{swift_checksum}`
<!-- librosa-swift-checksum:end -->"""

pattern = re.compile(
    r"\n*<!-- librosa-swift-checksum:start -->.*?<!-- librosa-swift-checksum:end -->",
    re.DOTALL,
)

if pattern.search(body):
    body = pattern.sub("\n\n" + section, body)
else:
    body = body + "\n\n" + section if body else section

output_path.write_text(body.rstrip() + "\n")
PY
}

if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_command git
require_command gh
require_command npm
require_command python3
require_command shasum

version="${1#v}"
tag="v${version}"

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] ||
  die "version must look like 0.1.0 or v0.1.0"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

[[ "$(git branch --show-current)" == "main" ]] ||
  die "release must be cut from main"

git diff --quiet || die "working tree has unstaged changes"
git diff --cached --quiet || die "working tree has staged changes"

git remote get-url origin >/dev/null || die "missing origin remote"
run git fetch origin main --tags

if ! git merge-base --is-ancestor origin/main HEAD; then
  die "local main is behind or diverged from origin/main; update it before releasing"
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  die "local tag already exists: ${tag}"
fi

if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
  die "remote tag already exists: ${tag}"
fi

if gh release view "$tag" >/dev/null 2>&1; then
  die "GitHub release already exists: ${tag}"
fi

echo "Updating WASM package version to ${version}"
run npm --prefix packages/librosa-wasm version "$version" --no-git-tag-version --allow-same-version

if ! git diff --quiet -- packages/librosa-wasm/package.json packages/librosa-wasm/package-lock.json; then
  run git add packages/librosa-wasm/package.json packages/librosa-wasm/package-lock.json
  run git commit -m "Release ${tag}"
else
  echo "WASM package version already set to ${version}; no version commit needed."
fi

commit_sha="$(git rev-parse HEAD)"

run git tag -a "$tag" -m "$tag"
run git push origin main
run git push origin "$tag"

run gh release create "$tag" \
  --verify-tag \
  --title "$tag" \
  --generate-notes

wait_for_workflow "cli-release.yml" "release assets" "$commit_sha"
wait_for_workflow "npm-publish.yml" "npm publish" "$commit_sha"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

archive="CLibrosa-${version}.xcframework.zip"
checksum_asset="${archive}.sha256"

run gh release download "$tag" \
  --pattern "$archive" \
  --pattern "$checksum_asset" \
  --dir "$tmpdir" \
  --clobber

archive_path="${tmpdir}/${archive}"
[[ -f "$archive_path" ]] || die "missing downloaded asset: ${archive}"

release_sha256="$(shasum -a 256 "$archive_path" | awk '{print $1}')"

if [[ -f "${tmpdir}/${checksum_asset}" ]]; then
  published_sha256="$(awk '{print $1}' "${tmpdir}/${checksum_asset}")"
  [[ "$published_sha256" == "$release_sha256" ]] ||
    die "downloaded asset checksum does not match published ${checksum_asset}"
fi

if command -v swift >/dev/null 2>&1; then
  swift_checksum="$(swift package compute-checksum "$archive_path")"
else
  swift_checksum="$release_sha256"
fi

repo_name="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
asset_url="https://github.com/${repo_name}/releases/download/${tag}/${archive}"

notes_file="${tmpdir}/release-notes.md"
updated_notes_file="${tmpdir}/release-notes-updated.md"
replace_swift_release_notes_section \
  "$tag" \
  "$asset_url" \
  "$swift_checksum" \
  "$release_sha256" \
  "$notes_file" \
  "$updated_notes_file"

run gh release edit "$tag" --notes-file "$updated_notes_file"

cat <<EOF

Release ${tag} is published.

XCFramework asset:
  ${asset_url}

SwiftPM checksum:
  ${swift_checksum}

Asset SHA-256:
  ${release_sha256}

Update librosa-swift:
  cd /Users/oli/Dev/librosa-swift
  scripts/update-binary-target.sh ${version}
  swift test
  git add Package.swift
  git commit -m "Use CLibrosa ${version} binary"
  git tag ${tag}
  git push origin main ${tag}
EOF
