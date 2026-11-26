-- ML-Pipeline-Anomaly-Detection: who may connect to project4_ml (one-time, re-runnable).
--
-- Acts on the database it is connected to (current_database()):
--   1. GRANT CONNECT to project5_reader (the decision-intelligence layer's read-only role,
--      created by apply-schema);
--   2. REVOKE CONNECT, TEMPORARY from PUBLIC, so any other role (for example
--      project4_mlflow, which owns only its own database) can no longer connect.
-- Superusers always connect: this pipeline's app and the decision-intelligence layer today
-- use the owner role, unaffected.
-- It refuses, changing nothing, when project5_reader does not exist yet: run
-- `python -m anomaly_detection.database apply-schema` first. Plain SQL (no psql
-- meta-commands), so the tests run the same file against the disposable database.
--
-- Run it, from the repository root:
--   docker compose exec -T anomaly-postgres psql -v ON_ERROR_STOP=1 -U manshu -d project4_ml < sql/restrict_project4_ml_connect.sql
-- Undo it:
--   docker compose exec -T anomaly-postgres psql -v ON_ERROR_STOP=1 -U manshu -d project4_ml -c 'GRANT CONNECT, TEMPORARY ON DATABASE project4_ml TO PUBLIC'

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'project5_reader') THEN
        RAISE EXCEPTION 'role project5_reader does not exist; run apply-schema first. Nothing was changed.';
    END IF;
    EXECUTE 'GRANT CONNECT ON DATABASE ' || quote_ident(current_database())
        || ' TO project5_reader';
    EXECUTE 'REVOKE CONNECT, TEMPORARY ON DATABASE ' || quote_ident(current_database())
        || ' FROM PUBLIC';
END
$$;

-- Every role that can log in: superuser, and whether it may connect to this database now.
SELECT r.rolname AS role,
       r.rolsuper AS superuser,
       has_database_privilege(r.rolname, current_database(), 'CONNECT') AS can_connect
FROM pg_roles AS r
WHERE r.rolcanlogin
ORDER BY r.rolname;
