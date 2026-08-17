-- BigQuery Complex SQL Sample
-- This query demonstrates advanced BigQuery features including CTEs, window functions,
-- array handling, struct operations, and complex joins.

WITH
-- Base customer data with customer lifetime value calculation
customer_metrics AS (
  SELECT
    customer_id,
    customer_name,
    email,
    signup_date,
    ARRAY_AGG(STRUCT(
      order_id AS order_id, order_date AS order_date, total_amount AS amount,
      product_category AS category
    ) ORDER BY order_date DESC
    LIMIT 10) AS recent_orders,
    SUM(total_amount) OVER (PARTITION BY customer_id) AS lifetime_value,
    COUNT(order_id) OVER (PARTITION BY customer_id) AS total_orders,
    RANK() OVER (ORDER BY SUM(total_amount) DESC) AS customer_rank
  FROM
    `project.dataset.orders`
  WHERE
    order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
  GROUP BY customer_id, customer_name, email, signup_date
),

-- Product performance metrics with rolling averages
product_analytics AS (
  SELECT
    product_id,
    product_name,
    category,
    SUM(quantity) AS total_sold,
    SUM(revenue) AS total_revenue,
    AVG(revenue / quantity) AS avg_price,
    AVG(SUM(revenue)) OVER (
      PARTITION BY category
      ORDER BY DATE_TRUNC(order_date, MONTH)
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3month_revenue,
    PERCENTILE_CONT(0.5) OVER (PARTITION BY category ORDER BY revenue)
      AS median_revenue
  FROM
    `project.dataset.order_items`
  WHERE
    order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
  GROUP BY product_id, product_name, category, DATE_TRUNC(order_date, MONTH)
),

-- Customer segmentation using RFM analysis
customer_segments AS (
  SELECT
    customer_id,
    customer_name,
    lifetime_value,
    total_orders,
    MAX(order_date) AS last_order_date,
    DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) AS days_since_last_order,
    CASE
      WHEN DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) <= 30 THEN 'Active'
      WHEN DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) <= 90 THEN 'At Risk'
      WHEN DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) <= 180 THEN 'Lapsed'
      ELSE 'Churned'
    END AS recency_segment,
    CASE
      WHEN total_orders >= 10 THEN 'High Frequency'
      WHEN total_orders >= 5 THEN 'Medium Frequency'
      ELSE 'Low Frequency'
    END AS frequency_segment,
    CASE
      WHEN lifetime_value >= 10000 THEN 'High Value'
      WHEN lifetime_value >= 5000 THEN 'Medium Value'
      ELSE 'Low Value'
    END AS monetary_segment,
    NTILE(4) OVER (ORDER BY lifetime_value DESC) AS value_quartile
  FROM
    `project.dataset.customer_orders`
  GROUP BY customer_id, customer_name, lifetime_value, total_orders
),

-- Channel attribution with multi-touch modeling
channel_attribution AS (
  SELECT
    session_id,
    customer_id,
    ARRAY_AGG(
      STRUCT(
        channel AS channel, touchpoint AS touchpoint, timestamp AS timestmp, campaign AS campaign
      )
      ORDER BY timestmp ASC
    ) AS customer_journey,
    FIRST_VALUE(channel) OVER (PARTITION BY session_id ORDER BY timestmp ASC)
      AS first_touch,
    LAST_VALUE(channel) OVER (PARTITION BY session_id ORDER BY timestmp ASC)
      AS last_touch,
    COUNT(DISTINCT channel) OVER (PARTITION BY session_id) AS channel_count
  FROM
    `project.dataset.attribution_events`
  WHERE timestamp >= DATE_SUB(
    CURRENT_DATE(
    ), INTERVAL 3 MONTH
  )
),


-- Cohort analysis for retention
cohort_analysis AS (
  SELECT
    cohort_month,
    activity_month,
    COUNT(
      DISTINCT customer_id
    ) AS active_customers,
    FIRST_VALUE(
      COUNT(
        DISTINCT customer_id)) OVER (
      PARTITION BY cohort_month ORDER BY activity_month
    ) AS cohort_size,
    COUNT(
      DISTINCT customer_id) / FIRST_VALUE(
      COUNT(
        DISTINCT customer_id)) OVER (
      PARTITION BY cohort_month ORDER BY activity_month
    ) AS retention_rate,
    DATE_DIFF(
      activity_month, cohort_month, MONTH
    ) AS month_number
  FROM
    customer_metrics
  GROUP BY cohort_month, activity_month
)

-- Final query combining all metrics
SELECT
  cm.customer_id,
  cm.customer_name,
  cm.email,
  cm.signup_date,
  cm.lifetime_value,
  cm.total_orders,
  cm.customer_rank,
  cm.recent_orders,
  cs.recency_segment,
  cs.frequency_segment,
  cs.monetary_segment,
  cs.value_quartile,
  ca.first_touch,
  ca.last_touch,
  ca.channel_count,
  ca.customer_journey,
  cohort.cohort_month,
  cohort.month_number,
  cohort.retention_rate,
  pa.category AS top_category,
  pa.total_sold,
  pa.rolling_3month_revenue,
  STRUCT(
    cm.lifetime_value AS ltv, cm.total_orders AS order_count,
    cs.days_since_last_order AS days_inactive
  ) AS customer_health_score
FROM
  customer_metrics AS cm
LEFT JOIN customer_segments AS cs ON cm.customer_id = cs.customer_id
LEFT JOIN channel_attribution AS ca ON cm.customer_id = ca.customer_id
LEFT JOIN cohort_analysis AS cohort ON cm.customer_id = cohort.customer_id
LEFT JOIN product_analytics AS pa
  ON pa.category = (
    SELECT category FROM UNNEST(cm.recent_orders)
    ORDER BY amount DESC LIMIT 1
  )
WHERE
  cm.customer_rank <= 1000
  AND cs.recency_segment IN (
    'Active',
    'At Risk'
  )
ORDER BY cm.lifetime_value DESC, cs.days_since_last_order ASC LIMIT 500;
