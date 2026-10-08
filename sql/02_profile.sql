-- Scale and quality profile
SELECT COALESCE(split, 'ALL')                          AS split,
       COUNT(DISTINCT test_id)                         AS tests,
       COUNT(*)                                        AS variants,
       SUM(impressions)                                AS impressions,
       SUM(clicks)                                     AS clicks,
       ROUND(100.0 * SUM(clicks) / SUM(impressions), 3) AS ctr_pct,
       MIN(created_at)::date AS first_test, MAX(created_at)::date AS last_test
FROM packages GROUP BY ROLLUP(split) ORDER BY split NULLS LAST;

SELECT (SELECT COUNT(*) FROM raw_packages) - (SELECT COUNT(*) FROM packages) AS dropped_rows;

-- Tests left with fewer than two variants (nothing to compare)
WITH t AS (SELECT test_id, COUNT(*) n FROM packages GROUP BY test_id)
SELECT COUNT(*) FILTER (WHERE n < 2) AS single_variant_tests, COUNT(*) AS tests FROM t;

-- Identical variants inside the same test: a built-in A/A test
WITH dup AS (
  SELECT test_id, headline, eyecatcher_id, excerpt, lede, COUNT(*) AS copies
  FROM packages
  GROUP BY test_id, headline, eyecatcher_id, excerpt, lede
  HAVING COUNT(*) > 1)
SELECT COUNT(DISTINCT test_id) AS tests_with_identical_variants, SUM(copies) AS identical_variants FROM dup;

-- Traffic per test and per variant
WITH t AS (SELECT test_id, SUM(impressions) imp, COUNT(*) arms,
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY impressions) med_arm
           FROM packages GROUP BY test_id)
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY imp)     AS median_test_impressions,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY med_arm) AS median_variant_impressions,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY arms)    AS median_variants_per_test
FROM t;

-- Within-test spread: how far the best variant sits above the test's average CTR
WITH v AS (
  SELECT test_id, clicks::float / impressions AS ctr,
         AVG(clicks::float / impressions) OVER (PARTITION BY test_id) AS test_avg,
         RANK() OVER (PARTITION BY test_id ORDER BY clicks::float / impressions DESC) AS rnk,
         COUNT(*) OVER (PARTITION BY test_id) AS arms
  FROM packages)
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY ctr / NULLIF(test_avg, 0) - 1) AS median_best_vs_avg_lift
FROM v WHERE rnk = 1 AND arms >= 2 AND test_avg > 0;
