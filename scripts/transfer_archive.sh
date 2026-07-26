#!/bin/bash
# Resumable transfer of the NHQG run archive to an ACCESS-CI cluster.
#
# Both PSC and SDSC need interactive auth (password + Duo), so run this
# yourself -- it cannot be backgrounded unattended on the first leg. It IS
# resumable: re-run the identical command after any interruption and rsync
# picks up mid-file.
#
#   scripts/transfer_archive.sh <user>@data.bridges2.psc.edu:/ocean/projects/<grant>/<user>/
#   scripts/transfer_archive.sh kaushiks@login.expanse.sdsc.edu:/expanse/lustre/projects/<grant>/<user>/
#
#   VERIFY_ONLY=1 scripts/transfer_archive.sh <dest>    # compare counts, transfer nothing
#
# Use the DATA-TRANSFER node where one exists (data.bridges2.psc.edu), not the
# login node.

set -euo pipefail

ARCHIVE=${ARCHIVE:-$(cd "$(dirname "$0")/../.." && pwd)/NHQG_runs_archive_2026-07}
DEST=${1:?usage: transfer_archive.sh <user>@<host>:<remote-parent-dir>/}
NAME=$(basename "$ARCHIVE")

if [ ! -d "$ARCHIVE" ]; then
  echo "FATAL: archive not found at $ARCHIVE" >&2
  echo "       build it with: python scripts/build_data_archive.py --dest $ARCHIVE" >&2
  exit 1
fi

LOCAL_FILES=$(find "$ARCHIVE" -type f | wc -l)
LOCAL_BYTES=$(du -sb "$ARCHIVE" | cut -f1)
echo "local:  $LOCAL_FILES files, $(numfmt --to=iec "$LOCAL_BYTES")  ($ARCHIVE)"

REMOTE_HOST=${DEST%%:*}
REMOTE_PATH=${DEST#*:}
REMOTE_DIR="${REMOTE_PATH%/}/$NAME"

if [ "${VERIFY_ONLY:-0}" = 1 ]; then
  echo "remote: querying $REMOTE_HOST:$REMOTE_DIR ..."
  # shellcheck disable=SC2029
  ssh "$REMOTE_HOST" "find '$REMOTE_DIR' -type f | wc -l; du -sb '$REMOTE_DIR' | cut -f1"
  echo "(expect $LOCAL_FILES files and $LOCAL_BYTES bytes)"
  exit 0
fi

echo "transferring to $REMOTE_HOST:$REMOTE_DIR"
echo "  resumable: re-run this identical command if it drops."
echo

# --partial --append-verify: resume interrupted files instead of restarting them.
# --no-H deliberately: the local tree is hardlinked into output/, but each file
# appears exactly once here, so there is nothing to preserve and -H would cost a
# full-tree inode scan.
rsync -avhP --partial --append-verify \
      --exclude '.nfs*' \
      "$ARCHIVE/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "transfer finished. verify with:"
echo "  VERIFY_ONLY=1 $0 $DEST"
