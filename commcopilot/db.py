"""PostgreSQL + pgvector database connection and schema management."""

import logging
import asyncpg
from commcopilot.config import DATABASE_URL

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            logger.info("Database pool created")
        except Exception as e:
            logger.error("Failed to create database pool: %s", e)
            raise
    return _pool


async def init_schema():
    """Create tables if they don't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_embeddings (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                text_chunk TEXT NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_recaps (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                transcript TEXT,
                phrases_used TEXT[],
                recap TEXT,
                scenario TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        logger.info("Database schema initialized")


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")
