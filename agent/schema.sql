PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, state TEXT NOT NULL, config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments(
  experiment_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
  plan TEXT NOT NULL, state TEXT NOT NULL, attempt_no INTEGER NOT NULL DEFAULT 0,
  lease_id TEXT, fencing_token INTEGER, duration REAL, error TEXT,
  UNIQUE(run_id,fingerprint));
CREATE TABLE IF NOT EXISTS experiment_parents(
  experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
  parent_id TEXT NOT NULL REFERENCES experiments(experiment_id), PRIMARY KEY(experiment_id,parent_id));
CREATE TABLE IF NOT EXISTS evaluations(
  evaluation_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
  record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS job_events(event_id TEXT PRIMARY KEY, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS selections(run_id TEXT PRIMARY KEY, experiment_id TEXT, revision INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS publications(run_id TEXT PRIMARY KEY, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS publication_journal(
  journal_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, state TEXT NOT NULL, record TEXT NOT NULL);
