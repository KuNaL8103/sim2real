#!/usr/bin/env bash
# Fetch the Franka Emika Panda MJCF from MuJoCo Menagerie at a pinned commit.
#
# Only the franka_emika_panda/ folder is checked out (sparse checkout). The
# upstream mesh directory is large (~34 MB), so it is git-ignored; this script
# restores it. The XML, LICENSE, and provenance notes are committed.
#
# Pin: MuJoCo Menagerie commit 71f066ad0be9cd271f7ed58c030243ef157af9f4
#      (2026-07-04, "Clean up scene <statistic> tags (#298)").
# License: Apache-2.0, as stated in franka_emika_panda/LICENSE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="71f066ad0be9cd271f7ed58c030243ef157af9f4"
DEST="$ROOT/assets/mujoco/franka_emika_panda"
REPO_URL="https://github.com/google-deepmind/mujoco_menagerie.git"

if [ -f "$DEST/panda.xml" ] && [ -d "$DEST/assets" ]; then
  echo "Franka assets already present at $DEST"
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to fetch Menagerie assets." >&2
  exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Sparse-checkout franka_emika_panda at $SHA"
git clone --filter=blob:none --no-checkout --quiet "$REPO_URL" "$TMP/menagerie"
git -C "$TMP/menagerie" sparse-checkout init --cone
git -C "$TMP/menagerie" sparse-checkout set franka_emika_panda
git -C "$TMP/menagerie" checkout --quiet "$SHA"

mkdir -p "$DEST"
cp -r "$TMP/menagerie/franka_emika_panda/." "$DEST/"
# The preview image is large and not needed by the simulator.
rm -f "$DEST/panda.png"

echo "Fetched into $DEST"
echo "Meshes are git-ignored; XML/LICENSE/ATTRIBUTION.md are committed."
