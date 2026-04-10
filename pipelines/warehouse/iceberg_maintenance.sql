-- Compact small files
CALL lakehouse.system.rewrite_data_files('banking.daily_kpis');
CALL lakehouse.system.rewrite_data_files('banking.customer_features');
CALL lakehouse.system.rewrite_data_files('banking.daily_kpis');

-- Expire snapshots
CALL lakehouse.system.expire_snapshots('banking.daily_kpis', TIMESTAMP '2026-01-01 00:00:00');
CALL lakehouse.system.expire_snapshots('banking.customer_features', TIMESTAMP '2026-01-01 00:00:00');
CALL lakehouse.system.expire_snapshots('banking.daily_kpis', TIMESTAMP '2026-01-01 00:00:00');

-- Remove orphan files
CALL lakehouse.system.remove_orphan_files('banking.daily_kpis');
CALL lakehouse.system.remove_orphan_files('banking.customer_features');
CALL lakehouse.system.remove_orphan_files('banking.daily_kpis');