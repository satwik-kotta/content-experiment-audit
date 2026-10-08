-- Load the Upworthy Research Archive (exploratory, confirmatory, holdout splits) into one table.
-- One row = one variant ("package") inside a headline A/B test.
DROP TABLE IF EXISTS raw_packages;
CREATE TABLE raw_packages (
  row_id INTEGER, created_at TIMESTAMP, updated_at TIMESTAMP, test_id TEXT,
  excerpt TEXT, headline TEXT, lede TEXT, slug TEXT, eyecatcher_id TEXT,
  impressions INTEGER, clicks INTEGER, significance DOUBLE PRECISION,
  first_place BOOLEAN, winner BOOLEAN, share_text TEXT, square TEXT, test_week INTEGER,
  split TEXT
);
CREATE TEMP TABLE stage (LIKE raw_packages INCLUDING DEFAULTS);
ALTER TABLE stage DROP COLUMN split;
\copy stage FROM 'data/upworthy-archive-exploratory-packages-03.12.2020.csv' WITH (FORMAT csv, HEADER true)
INSERT INTO raw_packages SELECT *, 'exploratory' FROM stage; TRUNCATE stage;
\copy stage FROM 'data/upworthy-archive-confirmatory-packages-03.12.2020.csv' WITH (FORMAT csv, HEADER true)
INSERT INTO raw_packages SELECT *, 'confirmatory' FROM stage; TRUNCATE stage;
\copy stage FROM 'data/upworthy-archive-holdout-packages-03.12.2020.csv' WITH (FORMAT csv, HEADER true)
INSERT INTO raw_packages SELECT *, 'holdout' FROM stage;

-- Clean variants: drop rows with no traffic or impossible counts
DROP TABLE IF EXISTS packages;
CREATE TABLE packages AS
SELECT * FROM raw_packages
WHERE impressions > 0 AND clicks >= 0 AND clicks <= impressions
  -- archive authors: tests created in this window were not fully randomized
  AND NOT (created_at >= '2013-06-25' AND created_at < '2014-01-11');
CREATE INDEX ON packages (test_id);
