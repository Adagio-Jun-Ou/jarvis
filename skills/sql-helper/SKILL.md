---
name: sql-helper
description: SQL query building, optimization, and database interaction helper for multiple databases
author: jarvis
version: 1.0.0
tags: [sql, database, postgres, mysql, sqlite, query]
metadata:
  jarvis:
    requires:
      bins: []
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🗄️"
---

# SQL Helper Skill

Expert assistance for SQL query building, optimization, and database operations across PostgreSQL, MySQL, SQLite, and more.

## Supported Databases

| Database | Connect String Format |
|----------|----------------------|
| PostgreSQL | `postgresql://user:pass@host:5432/db` |
| MySQL | `mysql://user:pass@host:3306/db` |
| SQLite | `sqlite:///path/to/database.db` |
| DuckDB | `duckdb:///path/to/database.duckdb` |

## Query Building

### SELECT Queries
```sql
-- Basic
SELECT * FROM users WHERE active = true;

-- With joins
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01'
ORDER BY o.total DESC
LIMIT 100;

-- Aggregations
SELECT 
    category,
    COUNT(*) as count,
    AVG(price) as avg_price,
    SUM(quantity) as total_qty
FROM products
GROUP BY category
HAVING COUNT(*) > 10
ORDER BY avg_price DESC;
```

### CTEs (Common Table Expressions)
```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        SUM(total) as revenue
    FROM orders
    GROUP BY 1
),
ranked AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_revenue
    FROM monthly_sales
)
SELECT 
    month,
    revenue,
    (revenue - prev_revenue) / prev_revenue * 100 as growth_pct
FROM ranked
WHERE prev_revenue IS NOT NULL;
```

### Window Functions
```sql
-- Running totals
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;

-- Rankings
SELECT 
    name,
    score,
    ROW_NUMBER() OVER (ORDER BY score DESC) as rank,
    DENSE_RANK() OVER (ORDER BY score DESC) as dense_rank
FROM players;

-- Partitioned aggregations
SELECT 
    department,
    employee,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;
```

## Query Optimization

### EXPLAIN ANALYZE
```sql
EXPLAIN ANALYZE SELECT ...;
```

### Index Recommendations
- Create indexes on columns used in:
  - WHERE clauses
  - JOIN conditions
  - ORDER BY / GROUP BY
  
```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
```

### Common Optimizations

1. **Avoid SELECT *** - Specify columns
2. **Use LIMIT** - Especially for exploratory queries
3. **Index strategically** - Not too many, not too few
4. **Use appropriate types** - Don't use TEXT for numbers
5. **Paginate properly**:
```sql
-- Keyset pagination (fastest)
SELECT * FROM items
WHERE id > :last_id
ORDER BY id
LIMIT 20;

-- Offset pagination (avoid for large offsets)
SELECT * FROM items
ORDER BY created_at DESC
LIMIT 20 OFFSET 1000;
```

## Schema Operations

### Inspect Schema
```sql
-- PostgreSQL
\d table_name
\dt  -- list tables

-- MySQL
DESCRIBE table_name;
SHOW TABLES;

-- SQLite
.schema table_name
.tables
```

### Safe Migrations
```sql
-- Check before altering
SELECT COUNT(*) FROM users WHERE email IS NULL;

-- Add column with default
ALTER TABLE users 
ADD COLUMN preferences JSONB DEFAULT '{}';

-- Create index concurrently (PostgreSQL)
CREATE INDEX CONCURRENTLY idx_users_name ON users(name);

-- Create index online (MySQL)
CREATE INDEX idx_users_name ON users(name) ALGORITHM=INPLACE;
```

## Common Patterns

### Upsert (Insert or Update)
```sql
-- PostgreSQL
INSERT INTO users (id, name, email)
VALUES (1, 'John', 'john@example.com')
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    email = EXCLUDED.email;

-- MySQL
INSERT INTO users (id, name, email)
VALUES (1, 'John', 'john@example.com')
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    email = VALUES(email);
```

### Full Text Search
```sql
-- PostgreSQL
SELECT * FROM articles
WHERE to_tsvector(content) @@ to_tsquery('postgresql & performance');

-- MySQL
SELECT * FROM articles
WHERE MATCH(content) AGAINST('postgresql performance' IN NATURAL LANGUAGE MODE);
```

### JSON Operations
```sql
-- PostgreSQL
SELECT data->>'name' as name
FROM events
WHERE data->>'type' = 'purchase';

-- MySQL
SELECT JSON_EXTRACT(data, '$.name') as name
FROM events
WHERE JSON_EXTRACT(data, '$.type') = 'purchase';
```

## Safety Rules

1. Never run DELETE or UPDATE without WHERE
2. Use transactions for multi-statement operations
3. Test on a copy first for destructive operations
4. Always backup before schema changes