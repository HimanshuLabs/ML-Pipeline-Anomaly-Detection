-- Project 4: MLflow tracking backend store (one-time setup, re-runnable).
--
-- Creates the dedicated login role project4_mlflow and the database project4_mlflow it
-- owns, in the anomaly-postgres instance. The MLflow server creates and migrates its
-- own tables inside project4_mlflow on first start. The role gets no rights anywhere
-- else, and this script never opens Project 4's metadata database.
--
-- Re-runnable: the role and the database are created only when absent; the password
-- is (re)set from the environment variable PROJECT4_MLFLOW_DB_PASSWORD on every run
-- and is never written in this file. Needs psql 15+ (\getenv). Run it as the instance
-- superuser against the maintenance database, from the repository root (only this one
-- variable is read from .env):
-- PROJECT4_MLFLOW_DB_PASSWORD="$(grep '^PROJECT4_MLFLOW_DB_PASSWORD=' .env | cut -d= -f2-)" \
--   docker compose exec -T -e PROJECT4_MLFLOW_DB_PASSWORD anomaly-postgres \
--   psql -v ON_ERROR_STOP=1 -U manshu -d postgres < sql/create_mlflow_database.sql

\set mlflow_password ''
\getenv mlflow_password PROJECT4_MLFLOW_DB_PASSWORD
SELECT :'mlflow_password' = '' AS mlflow_password_missing \gset
\if :mlflow_password_missing
    \echo 'PROJECT4_MLFLOW_DB_PASSWORD is not set; refusing to create the MLflow role.'
    SELECT 1 / 0 AS refused_without_password;
\endif

SELECT 'CREATE ROLE project4_mlflow LOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'project4_mlflow') \gexec

ALTER ROLE project4_mlflow
    WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'mlflow_password';

SELECT 'CREATE DATABASE project4_mlflow OWNER project4_mlflow'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'project4_mlflow') \gexec

-- Only the owner role connects to the MLflow database.
REVOKE ALL ON DATABASE project4_mlflow FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE project4_mlflow TO project4_mlflow;
