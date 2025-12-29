"""
Database initialization and migrations for AutoSeg.
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Current schema version
SCHEMA_VERSION = 1


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> None:
    """
    Initialize a new database with all required tables.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        # Version tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Projects table (for multi-project support, though typically one per DB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                root_dir TEXT NOT NULL,
                db_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings_json TEXT
            )
        """)
        
        # Labels/classes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                color_hex TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, name)
            )
        """)
        
        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                order_index INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, path)
            )
        """)
        
        # Create index for efficient ordering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_order 
            ON images(project_id, order_index)
        """)
        
        # Annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                bbox_xyxy_json TEXT,
                polygon_norm_json TEXT,
                mask_rle_json TEXT,
                source TEXT DEFAULT 'manual',
                confidence REAL,
                status TEXT DEFAULT 'approved',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE
            )
        """)
        
        # Create index for efficient lookup by image
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_annotations_image 
            ON annotations(image_id)
        """)
        
        # Key-value store for settings, last open image, etc.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Record schema version
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version) VALUES (?)
        """, (SCHEMA_VERSION,))
        
        conn.commit()
    finally:
        conn.close()


def get_schema_version(db_path: str) -> int:
    """Get the current schema version of the database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def migrate_db(db_path: str) -> None:
    """
    Run any pending migrations on the database.
    
    Args:
        db_path: Path to the SQLite database file
    """
    current_version = get_schema_version(db_path)
    
    if current_version < SCHEMA_VERSION:
        # Run migrations
        conn = get_connection(db_path)
        try:
            # Migration 0 -> 1: Initial schema (handled by init_db)
            if current_version < 1:
                init_db(db_path)
            
            # Future migrations would go here:
            # if current_version < 2:
            #     _migrate_v1_to_v2(conn)
            
            conn.commit()
        finally:
            conn.close()


def _set_kv(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a key-value pair in the kv table."""
    conn.execute(
        "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
        (key, value)
    )


def _get_kv(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get a value from the kv table."""
    cursor = conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None
