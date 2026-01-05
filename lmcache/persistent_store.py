# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
import json
import sqlite3
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, parse_cache_key

logger = init_logger(__name__)


class DiskCacheMetadataStore:
    """
    Manages the persistence of DiskCacheMetadata using an SQLite database
    to ensure that disk cache mappings survive application restarts.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        # `check_same_thread=False` is acceptable here because we serialize
        # all access through our own lock.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._create_table()

    def _create_table(self):
        """Creates the necessary table if it doesn't exist."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS disk_cache_meta (
                        cache_key TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        shape TEXT,
                        dtype TEXT
                    )
                """
                )

    def load_all(self) -> Dict[CacheEngineKey, DiskCacheMetadata]:
        """Loads all metadata from the database into a dictionary."""
        logger.info(f"Loading disk cache metadata from {self._db_path}")
        all_meta: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT cache_key, path, size, shape, dtype FROM disk_cache_meta"
            )
            for row in cursor.fetchall():
                cache_key_str, path, size, shape_str, dtype_str = row
                try:
                    key = parse_cache_key(cache_key_str)
                    meta_dict = {
                        "path": path,
                        "size": size,
                        "shape": json.loads(shape_str) if shape_str else None,
                        "dtype": dtype_str,
                    }
                    metadata = DiskCacheMetadata.from_dict(meta_dict)
                    all_meta[key] = metadata
                except Exception as e:
                    logger.warning(
                        f"Failed to load metadata for key '{cache_key_str}': {e}. Deleting invalid entry."
                    )
                    self.delete_by_str(cache_key_str)
        logger.info(f"Loaded {len(all_meta)} disk cache metadata entries.")
        return all_meta

    def save(self, key: CacheEngineKey, metadata: DiskCacheMetadata):
        """Saves or updates a metadata entry in the database."""
        key_str = key.to_string()
        meta_dict = metadata.to_dict()
        shape_str = json.dumps(meta_dict["shape"]) if meta_dict.get("shape") else None

        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO disk_cache_meta VALUES (?, ?, ?, ?, ?)",
                    (key_str, meta_dict["path"], meta_dict["size"], shape_str, meta_dict["dtype"]),
                )

    def delete_by_str(self, key_str: str):
        """Deletes a metadata entry from the database using a key string."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM disk_cache_meta WHERE cache_key = ?", (key_str,)
                )

    def batched_delete_by_str(self, key_strs: List[str]):
        """Deletes multiple metadata entries from the database in a batch."""
        if not key_strs:
            return
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    "DELETE FROM disk_cache_meta WHERE cache_key = ?",
                    [(key,) for key in key_strs],
                )

    def close(self):
        """Closes the database connection."""
        self._conn.close()