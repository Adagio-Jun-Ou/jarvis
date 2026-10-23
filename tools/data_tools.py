"""Data analysis tools for Jarvis.

Provides comprehensive data analysis capabilities including:
- Statistical analysis
- Data visualization
- Data transformation
- File format support (CSV, JSON, Excel, Parquet)
- Memory-bounded DataFrame caching for 32GB systems
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from collections import OrderedDict

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


class DataFrameCache:
    """Bounded cache for DataFrames with memory limit enforcement.
    
    Optimized for 32GB RAM - allows generous memory for data analysis
    while preventing unbounded growth. Default 2GB allows multiple
    large datasets without significant RAM pressure.
    """

    MAX_MEMORY_MB = 2048  # 2GB max - ~6% of 32GB RAM
    MAX_ENTRIES = 50      # Up to 50 different DataFrames

    def __init__(self, max_memory_mb: int = 2048, max_entries: int = 50):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self._cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._current_memory = 0
        self._lock = asyncio.Lock()
        self._stats = {
            "evictions": 0,
            "hits": 0,
            "misses": 0,
            "peak_memory_mb": 0.0,
        }

    def _estimate_memory(self, df: Any) -> int:
        """Estimate DataFrame memory usage in bytes."""
        try:
            return int(df.memory_usage(deep=True).sum())
        except Exception:
            return len(df) * len(df.columns) * 8 if hasattr(df, 'columns') else 1024

    async def get(self, key: str) -> Any | None:
        """Get DataFrame from cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            df, size = self._cache[key]
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return df

    async def set(self, key: str, df: Any) -> bool:
        """Store DataFrame in cache with memory management."""
        size = self._estimate_memory(df)

        if size > self.max_memory_bytes:
            return False

        async with self._lock:
            if key in self._cache:
                _, old_size = self._cache[key]
                self._current_memory -= old_size
                del self._cache[key]

            while (self._current_memory + size > self.max_memory_bytes or
                   len(self._cache) >= self.max_entries):
                if not self._evict_oldest():
                    break

            self._cache[key] = (df, size)
            self._current_memory += size

            peak = self._current_memory / (1024 * 1024)
            if peak > self._stats["peak_memory_mb"]:
                self._stats["peak_memory_mb"] = peak

        return True

    def _evict_oldest(self) -> bool:
        """Evict oldest (least recently used) entry."""
        if not self._cache:
            return False

        key = next(iter(self._cache))
        _, size = self._cache[key]
        del self._cache[key]
        self._current_memory -= size
        self._stats["evictions"] += 1
        return True

    async def remove(self, key: str) -> bool:
        """Remove specific DataFrame."""
        async with self._lock:
            if key not in self._cache:
                return False
            _, size = self._cache[key]
            del self._cache[key]
            self._current_memory -= size
            return True

    async def clear(self) -> int:
        """Clear all DataFrames, return count cleared."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_memory = 0
            return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._stats,
            "entries": len(self._cache),
            "current_memory_mb": self._current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
        }


class DataAnalysisTool(Tool):
    """Tool for data analysis operations with memory-bounded caching."""

    name = "data_analysis"
    description = "Analyze data files (CSV, JSON, Excel) with statistics and visualizations"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Analysis action to perform",
                "enum": [
                    "load",
                    "stats",
                    "describe",
                    "head",
                    "columns",
                    "filter",
                    "groupby",
                    "correlation",
                    "visualize",
                    "transform",
                    "export",
                    "clear_cache",
                    "cache_stats",
                ],
            },
            "file_path": {
                "type": "string",
                "description": "Path to the data file",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to analyze",
            },
            "filter_conditions": {
                "type": "object",
                "description": "Filter conditions (column: value pairs)",
            },
            "group_by": {
                "type": "string",
                "description": "Column to group by",
            },
            "aggregation": {
                "type": "string",
                "description": "Aggregation function (sum, mean, count, etc.)",
            },
            "chart_type": {
                "type": "string",
                "description": "Type of visualization (bar, line, pie, scatter, histogram)",
            },
            "output_format": {
                "type": "string",
                "description": "Output format for export (csv, json, excel)",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, max_cache_mb: int = 500):
        self._cache = DataFrameCache(max_memory_mb=max_cache_mb)

    async def execute(
        self,
        action: str,
        file_path: str | None = None,
        columns: list[str] | None = None,
        filter_conditions: dict | None = None,
        group_by: str | None = None,
        aggregation: str | None = None,
        chart_type: str | None = None,
        output_format: str | None = None,
        **kwargs  # Accept extra params for flexibility
    ) -> str:
        """Execute data analysis action.
        
        Args:
            action: Analysis action to perform
            file_path: Path to the data file
            columns: Columns to analyze
            filter_conditions: Filter conditions (column: value pairs)
            group_by: Column to group by
            aggregation: Aggregation function (sum, mean, count, etc.)
            chart_type: Type of visualization (bar, line, pie, scatter, histogram)
            output_format: Output format for export (csv, json, excel)
            **kwargs: Additional parameters (ignored for compatibility)
        
        Returns:
            Analysis results as formatted string
        """
        # Handle common parameter variations
        if not file_path:
            file_path = kwargs.get('path', kwargs.get('file', kwargs.get('filename', None)))
        
        # Cache operations don't need pandas
        if action == "clear_cache":
            count = await self._cache.clear()
            return f"Cleared {count} DataFrames from cache"
        elif action == "cache_stats":
            return json.dumps(self._cache.get_stats(), indent=2)

        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            return "Error: pandas and numpy required. Install with: pip install pandas numpy"

        try:
            if action == "load":
                return await self._load_data(file_path)
            elif action == "stats":
                return await self._get_stats(file_path)
            elif action == "describe":
                return await self._describe_data(file_path)
            elif action == "head":
                return await self._get_head(file_path, n=10)
            elif action == "columns":
                return await self._get_columns(file_path)
            elif action == "filter":
                return await self._filter_data(file_path, filter_conditions)
            elif action == "groupby":
                return await self._groupby_data(file_path, group_by, aggregation, columns)
            elif action == "correlation":
                return await self._correlation(file_path, columns)
            elif action == "visualize":
                return await self._visualize(file_path, chart_type, columns)
            elif action == "transform":
                return await self._transform(file_path, columns)
            elif action == "export":
                return await self._export(file_path, output_format)
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: {safe_error_message(e)}"

    async def _load_data(self, file_path: str | None) -> str:
        """Load data from file with bounded caching."""
        if not file_path:
            return "Error: file_path required for load action"

        import pandas as pd

        path = Path(file_path).expanduser()
        if not path.exists():
            return f"File not found: {file_path}"

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            return f"Unsupported file type: {suffix}"

        stored = await self._cache.set(file_path, df)
        if not stored:
            return f"Loaded {len(df)} rows but cache full (file too large). Using temporary session only."

        stats = self._cache.get_stats()
        return f"Loaded {len(df)} rows x {len(df.columns)} columns from {path.name}\nCache: {stats['entries']} DataFrames, {stats['current_memory_mb']:.1f}MB / {stats['max_memory_mb']:.0f}MB"

    async def _get_stats(self, file_path: str | None) -> str:
        """Get basic statistics."""
        import numpy as np

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded. Use 'load' action first."

        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "null_counts": df.isnull().sum().to_dict(),
        }

        return json.dumps(stats, indent=2, default=str)

    async def _describe_data(self, file_path: str | None) -> str:
        """Get descriptive statistics."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        return df.describe(include='all').to_string()

    async def _get_head(self, file_path: str | None, n: int = 10) -> str:
        """Get first n rows."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        return df.head(n).to_string()

    async def _get_columns(self, file_path: str | None) -> str:
        """Get column information."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        lines = ["Columns:"]
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()
            lines.append(f"  - {col}: {dtype} ({unique} unique, {nulls} nulls)")

        return "\n".join(lines)

    async def _filter_data(self, file_path: str | None, filter_conditions: dict | None) -> str:
        """Filter data by conditions."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        if not filter_conditions:
            return "No filter conditions provided."

        filtered_df = df.copy()
        for col, value in filter_conditions.items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]

        return f"Filtered to {len(filtered_df)} rows\n{filtered_df.head(10).to_string()}"

    async def _groupby_data(
        self,
        file_path: str | None,
        group_by: str | None,
        aggregation: str | None,
        columns: list[str] | None,
    ) -> str:
        """Group and aggregate data."""
        import numpy as np

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        if not group_by:
            return "group_by column required"

        if group_by not in df.columns:
            return f"Column '{group_by}' not found"

        agg_cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
        agg_func = aggregation or "mean"

        result = df.groupby(group_by)[agg_cols].agg(agg_func)
        return result.to_string()

    async def _correlation(self, file_path: str | None, columns: list[str] | None) -> str:
        """Calculate correlation matrix."""
        import numpy as np

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        numeric_df = df.select_dtypes(include=[np.number])
        if columns:
            numeric_df = numeric_df[columns]

        corr = numeric_df.corr()
        return corr.to_string()

    async def _visualize(
        self,
        file_path: str | None,
        chart_type: str | None,
        columns: list[str] | None,
    ) -> str:
        """Generate visualization."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        chart = chart_type or "histogram"

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            if chart == "histogram" and columns:
                df[columns[0]].hist(ax=ax, bins=30)
                ax.set_title(f"Histogram of {columns[0]}")
            elif chart == "bar" and columns:
                if len(columns) >= 2:
                    df.groupby(columns[0])[columns[1]].sum().plot(kind='bar', ax=ax)
                else:
                    df[columns[0]].value_counts().plot(kind='bar', ax=ax)
            elif chart == "line" and columns:
                df[columns[0]].plot(ax=ax)
            elif chart == "scatter" and len(columns) >= 2:
                df.plot.scatter(x=columns[0], y=columns[1], ax=ax)
            elif chart == "pie" and columns:
                df[columns[0]].value_counts().plot(kind='pie', ax=ax)
            else:
                return f"Invalid chart configuration: {chart} with columns {columns}"

            # Save to file
            output_path = Path("/tmp/jarvis_viz.png")
            plt.savefig(output_path)
            plt.close()

            return f"Visualization saved to {output_path}"
        except ImportError:
            return "matplotlib required for visualization. Install with: pip install matplotlib"

    async def _transform(self, file_path: str | None, columns: list[str] | None) -> str:
        """Transform data (normalize, encode, etc.)."""
        import pandas as pd

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        if not columns:
            return "Columns required for transform"

        transformed = df.copy()
        for col in columns:
            if col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    transformed = pd.concat([transformed, dummies], axis=1)
                    transformed.drop(col, axis=1, inplace=True)
                else:
                    transformed[col] = (df[col] - df[col].mean()) / df[col].std()

        cache_key = (file_path or "data") + "_transformed"
        await self._cache.set(cache_key, transformed)
        return f"Transformed data cached as {cache_key}: {len(transformed.columns)} columns"

    async def _export(self, file_path: str | None, output_format: str | None) -> str:
        """Export data to file."""

        df = await self._get_dataframe(file_path)
        if df is None:
            return "No data loaded."

        fmt = output_format or "csv"
        output_path = Path(f"/tmp/exported_data.{fmt}")

        if fmt == "csv":
            df.to_csv(output_path, index=False)
        elif fmt == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif fmt == "excel":
            df.to_excel(output_path, index=False)
        else:
            return f"Unsupported format: {fmt}"

        return f"Data exported to {output_path}"

    async def _get_dataframe(self, file_path: str | None):
        """Get dataframe from cache."""
        if file_path:
            df = await self._cache.get(file_path)
            if df is not None:
                return df
        return None


class ConnectionPool:
    """Bounded connection pool with automatic cleanup.
    
    Prevents connection leaks by limiting total connections and
    providing automatic cleanup of idle connections. Each connection
    is lightweight, so generous limit is fine.
    """
    MAX_CONNECTIONS = 20
    IDLE_TIMEOUT_SECONDS = 600  # 10 minutes

    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self._connections: dict[str, Any] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, connection_string: str) -> Any | None:
        """Get connection from pool or create new one."""
        import time

        async with self._lock:
            if connection_string in self._connections:
                self._last_used[connection_string] = time.time()
                return self._connections[connection_string]
            return None

    async def set(self, connection_string: str, conn: Any) -> bool:
        """Add connection to pool with limit enforcement."""
        import time

        evicted_conn = None
        async with self._lock:
            if len(self._connections) >= self.max_connections:
                oldest_key = min(self._last_used, key=self._last_used.get)
                evicted_conn = self._connections.pop(oldest_key, None)
                self._last_used.pop(oldest_key, None)

            self._connections[connection_string] = conn
            self._last_used[connection_string] = time.time()

        if evicted_conn is not None:
            try:
                await asyncio.to_thread(evicted_conn.close)
            except Exception as e:
                logger.debug(f"Failed to close evicted database connection: {e}")

        return True

    async def remove(self, connection_string: str) -> bool:
        """Remove and close a connection."""
        return await self._close_connection(connection_string)

    async def _close_connection(self, connection_string: str) -> bool:
        """Close and remove connection (releases lock before closing)."""
        conn = None
        async with self._lock:
            if connection_string in self._connections:
                conn = self._connections.pop(connection_string)
                self._last_used.pop(connection_string, None)
        if conn:
            try:
                await asyncio.to_thread(conn.close)
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")
            return True
        return False

    async def cleanup_idle(self) -> int:
        """Close idle connections, return count closed."""
        import time

        closed_conns = []
        async with self._lock:
            now = time.time()
            to_close = [
                key for key, last in self._last_used.items()
                if now - last > self.IDLE_TIMEOUT_SECONDS
            ]
            for key in to_close:
                conn = self._connections.pop(key, None)
                self._last_used.pop(key, None)
                if conn:
                    closed_conns.append(conn)
        for conn in closed_conns:
            try:
                await asyncio.to_thread(conn.close)
            except Exception as e:
                logger.debug(f"Failed to close idle database connection: {e}")
        return len(closed_conns)

    async def close_all(self) -> int:
        """Close all connections."""
        conns = []
        async with self._lock:
            conns = list(self._connections.values())
            self._connections.clear()
            self._last_used.clear()
        for conn in conns:
            try:
                await asyncio.to_thread(conn.close)
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")
        return len(conns)

    def stats(self) -> dict:
        """Get pool statistics."""
        return {
            "active_connections": len(self._connections),
            "max_connections": self.max_connections,
        }


class SQLQueryTool(Tool):
    """Tool for SQL database queries with bounded connection pool.

    Use parameterized queries (passing `params`) to prevent SQL injection.
    When `params` is provided, the query uses `cursor.execute(query, params)`
    with proper parameter binding. The keyword/pattern blocklist below serves
    as defense-in-depth only — it is NOT sufficient on its own to prevent
    SQL injection. Always prefer parameterized queries for user-supplied values.
    """

    name = "sql_query"
    description = "Execute SQL queries on databases (SQLite, PostgreSQL, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["connect", "query", "tables", "schema", "disconnect", "pool_stats"],
                "description": "SQL action to perform",
            },
            "connection_string": {
                "type": "string",
                "description": "Database connection string",
            },
            "query": {
                "type": "string",
                "description": "SQL query to execute (use ? placeholders with params for safe execution)",
            },
            "params": {
                "type": "array",
                "description": "Query parameters for parameterized execution (recommended for any user-supplied values). Passed as cursor.execute(query, params). Prevents SQL injection.",
                "items": {"type": ["string", "number", "boolean", "null"]},
            },
        },
        "required": ["action"],
    }
    requires_permission = True  # SQL connections and queries require permission

    def __init__(self, max_connections: int = 10):
        self._pool = ConnectionPool(max_connections=max_connections)

    async def execute(
        self,
        action: str,
        connection_string: str | None = None,
        query: str | None = None,
        params: list | None = None,
        **kwargs
    ) -> str:
        """Execute SQL action."""
        try:
            import sqlite3
        except ImportError:
            return "Error: sqlite3 required"

        if action == "connect":
            if not connection_string:
                return "Connection string required"
            try:
                db_path = Path(connection_string).resolve()
                allowed_dirs = [
                    Path.home() / ".jarvis",
                    Path.home() / "projects",
                    Path("/tmp"),
                ]
                if not any(str(db_path).startswith(str(d.resolve())) for d in allowed_dirs):
                    if not connection_string.startswith(":memory:"):
                        return f"Error: Database path must be within allowed directories ({', '.join(str(d) for d in allowed_dirs)})"
                loop = asyncio.get_running_loop()
                conn = await loop.run_in_executor(
                    None, lambda: sqlite3.connect(connection_string)
                )
                await self._pool.set(connection_string, conn)
                return f"Connected to database"
            except Exception as e:
                return f"Connection failed: {safe_error_message(e)}"

        elif action == "pool_stats":
            return json.dumps(self._pool.stats(), indent=2)

        elif action == "tables":
            conn = await self._pool.get(connection_string or "")
            if not connection_string or conn is None:
                return "Not connected. Use 'connect' first."
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            return f"Tables: {', '.join(tables)}"

        elif action == "schema":
            conn = await self._pool.get(connection_string or "")
            if not connection_string or conn is None:
                return "Not connected. Use 'connect' first."
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
            )
            schemas = [row[0] for row in cursor.fetchall() if row[0]]
            return "\n\n".join(schemas)

        elif action == "query":
            if not query:
                return "Query required"

            query_upper = query.strip().upper()
            # Defense-in-depth keyword blocklist: NOT sufficient to prevent
            # SQL injection on its own. Use parameterized queries (params=)
            # for any user-supplied values. This blocklist only blocks
            # dangerous statement types that should never be executable via
            # this read-only query tool.
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE', 'REPLACE', 'GRANT', 'REVOKE', 'ATTACH', 'PRAGMA']
            for keyword in dangerous_keywords:
                if re.search(rf'\b{keyword}\b', query_upper):
                    return f"Security: '{keyword}' operations are not allowed. Only SELECT queries are permitted."
            
            dangerous_patterns = [
                # Defense-in-depth pattern blocklist: these patterns catch common
                # injection techniques but are NOT a complete solution. Always use
                # parameterized queries (params=) for user-supplied values.
                r'UNION\s+(ALL\s+)?SELECT',
                r';\s*SELECT',
                r'--',
                r'/\*',
                r'\*/',
                r'EXEC\s*\(',
                r'EXECUTE\s+',
                r'LOAD_EXTENSION',
                r'SQLITE_MASTER',
                r'PRAGMA_',
                r'GROUP_CONCAT\s*\(\s*SQL\s*\)',
                r'CHAR\s*\(',
                r'HEX\s*\(',
                r'QUOTE\s*\(',
                r'LOAD_EXTENSION',
                r'READFILE',
                r'WRITEFILE',
                r'FTS',
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, query_upper, re.IGNORECASE):
                    return "Security: Potentially dangerous query pattern detected. Only simple SELECT queries are permitted."

            if 'LIMIT' not in query_upper:
                query = query.rstrip(';') + ' LIMIT 100'

            conn = await self._pool.get(connection_string or "")
            if conn is None and connection_string:
                try:
                    loop = asyncio.get_running_loop()
                    conn = await loop.run_in_executor(
                        None, lambda: sqlite3.connect(connection_string)
                    )
                    await self._pool.set(connection_string, conn)
                except Exception:
                    return "Not connected. Use 'connect' first."
            if conn is None:
                return "Not connected. Use 'connect' first."
            try:
                if params is not None:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                rows = cursor.fetchmany(100)  # Limit rows fetched
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                result = [dict(zip(columns, row)) for row in rows]
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                return f"Query error: {safe_error_message(e)}"

        elif action == "disconnect":
            if connection_string:
                closed = await self._pool.remove(connection_string)
                if closed:
                    return "Disconnected"
            return "No connection to close"

        return f"Unknown action: {action}"


class StatisticalTool(Tool):
    """Tool for statistical analysis."""

    name = "statistics"
    description = "Perform statistical analysis on data (mean, median, std, regression, etc.)"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "describe",
                    "ttest",
                    "anova",
                    "correlation",
                    "regression",
                    "chi_square",
                    "distribution",
                ],
                "description": "Statistical operation to perform",
            },
            "data": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Data array for analysis",
            },
            "data2": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Second data array (for comparisons)",
            },
            "alpha": {
                "type": "number",
                "description": "Significance level (default 0.05)",
            },
        },
        "required": ["operation", "data"],
    }
    requires_permission = False

    async def execute(
        self,
        operation: str = "",
        data: list[float] | None = None,
        data2: list[float] | None = None,
        alpha: float = 0.05,
        **kwargs
    ) -> str:
        """Execute statistical operation."""
        try:
            import numpy as np
            from scipy import stats
        except ImportError:
            return "Error: numpy and scipy required. Install with: pip install numpy scipy"

        if not data:
            return "Data required for analysis"

        arr = np.array(data)

        if operation == "describe":
            result = {
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "var": float(np.var(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "skew": float(stats.skew(arr)),
                "kurtosis": float(stats.kurtosis(arr)),
            }
            return json.dumps(result, indent=2)

        elif operation == "ttest" and data2:
            arr2 = np.array(data2)
            t_stat, p_value = stats.ttest_ind(arr, arr2)
            return json.dumps({
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
            }, indent=2)

        elif operation == "correlation" and data2:
            arr2 = np.array(data2)
            r, p = stats.pearsonr(arr, arr2)
            return json.dumps({
                "correlation": float(r),
                "p_value": float(p),
                "significant": p < alpha,
            }, indent=2)

        elif operation == "regression" and data2:
            arr2 = np.array(data2)
            slope, intercept, r, p, se = stats.linregress(arr, arr2)
            return json.dumps({
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r ** 2),
                "p_value": float(p),
                "std_error": float(se),
            }, indent=2)

        elif operation == "distribution":
            # Test normality
            stat, p_normal = stats.shapiro(arr[:5000] if len(arr) > 5000 else arr)
            return json.dumps({
                "normality_test": {
                    "statistic": float(stat),
                    "p_value": float(p_normal),
                    "is_normal": p_normal > alpha,
                },
                "histogram": np.histogram(arr, bins=10)[0].tolist(),
            }, indent=2)

        else:
            return f"Operation '{operation}' not implemented or requires additional data"


# Convenience function to get all data analysis tools
def get_data_tools():
    """Get all data analysis tools."""
    return [
        DataAnalysisTool(),
        SQLQueryTool(),
        StatisticalTool(),
    ]
