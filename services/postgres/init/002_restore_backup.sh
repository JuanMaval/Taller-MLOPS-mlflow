#!/bin/bash
# Restore the business data on top of the schema created by 001_schemas.sql.
#
# Runs once, right after 001, when the postgres data volume is empty.
# The dump is a pg_dump custom-format archive that also contains the schema,
# so only its data (rows + sequence values) is restored here. A data-only
# restore does not order tables by foreign key, so FK triggers are disabled
# while the rows are copied (requires superuser; POSTGRES_USER is one).
set -euo pipefail

DUMP="${BACKUP_DUMP:-/backup/cubierta_forestal_2026-10-03.dump}"

if [[ ! -f "$DUMP" ]]; then
    echo "002_restore_backup: no dump at $DUMP, skipping data restore"
    exit 0
fi

echo "002_restore_backup: restoring data from $DUMP"
pg_restore --data-only --disable-triggers --no-owner --exit-on-error \
    -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    "$DUMP"
echo "002_restore_backup: done"
