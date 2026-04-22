# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""
Terrakit Data Fetch Cache Manager - Persistent Volume Edition
Caches geospatial data using Redis (metadata) + Shared PV (files)
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import redis
from gfm_data_processing.common import logger

REDIS_URL = os.getenv("REDIS_URL", "")


def get_redis_client(redis_url: str = REDIS_URL):
    """Establish a connection to the Redis server."""
    try:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except redis.exceptions.ConnectionError:
        logger.exception("❌ Redis: Connection error: %s", redis_url)
    except Exception:
        logger.exception("❌ Redis: An unexpected error occurred: %s", redis_url)


redis_client = get_redis_client()


class TerrakitPVCacheManager:
    """
    Manages caching using Redis + Shared Persistent Volume.

    - Redis: Stores metadata and file paths (fast lookups)
    - PV: Stores actual GeoTIFF files (shared across pods)
    - Uses existing redis_client singleton
    - No S3 dependency - files stay on shared disk
    """

    def __init__(
        self,
        cache_dir: str,
        cache_ttl_days: int = 30,
        enabled: bool = True,
        max_cache_size_gb: Optional[float] = None,
    ):
        """
        Initialize PV cache manager.

        Args:
            cache_dir: Path to shared PV mount point (e.g., /data/cache)
            cache_ttl_days: Cache expiration in days (default: 30)
            enabled: Enable/disable caching (default: True)
            max_cache_size_gb: Optional max cache size in GB
        """
        self.enabled = enabled
        self.cache_ttl_seconds = cache_ttl_days * 86400
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_bytes = (
            max_cache_size_gb * 1024**3 if max_cache_size_gb else None
        )

        # Use existing redis_client singleton
        self.redis_client = redis_client

        if not self.enabled:
            logger.info("📦 Cache is disabled")
            return

        # Verify Redis connection
        try:
            if self.redis_client:
                self.redis_client.ping()
                logger.info("✅ Using existing Redis client for cache")
            else:
                logger.warning("❌ Redis client not initialized - cache disabled")
                self.enabled = False
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e} - cache disabled")
            self.enabled = False

        # Verify PV is mounted and writable
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.cache_dir / ".cache_test"
            test_file.touch()
            test_file.unlink()
            logger.info(f"✅ Cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"❌ Cache directory not writable: {e} - cache disabled")
            self.enabled = False

    def get_cache_key(
        self,
        bbox: List[float],
        date: str,
        collection_name: str,
        band_names: List[str],
        maxcc: float,
        modality_tag: str,
        transform: Optional[str] = None,
    ) -> str:
        """
        Generate deterministic cache key from query parameters.

        Args:
            bbox: Bounding box coordinates
            date: Data date
            collection_name: Data collection name
            band_names: List of band names
            maxcc: Maximum cloud cover
            modality_tag: Modality identifier
            transform: Optional transform applied (e.g., "to_decibels")

        Returns:
            Cache key string
        """
        key_data = {
            "bbox": bbox,
            "date": date,
            "collection": collection_name,
            "bands": sorted(band_names),
            "maxcc": maxcc,
            "modality": modality_tag,
            "transform": transform,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"terrakit:v1:{key_hash}"

    def _get_cache_file_paths(self, cache_key: str, date: str) -> tuple:
        """Get file paths in PV for cached files."""
        cache_hash = cache_key.split(":")[-1][:16]
        date_dir = self.cache_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)

        original_path = date_dir / f"{cache_hash}_original.tif"
        imputed_path = date_dir / f"{cache_hash}_imputed.tif"

        return str(original_path), str(imputed_path)

    def get_cached_files(self, cache_key: str) -> Optional[Dict]:
        """
        Check if cached files exist in PV and return metadata.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Dict with metadata including file paths, or None if not cached
        """
        if not self.enabled:
            return None

        try:
            # Check Redis for metadata
            cached_data_json = self.redis_client.get(cache_key)
            if not cached_data_json:
                logger.debug(f"🔍 Cache miss: {cache_key[:16]}...")
                return None

            metadata = json.loads(cached_data_json)

            # Verify files exist in PV
            original_path = metadata.get("original_pv_path")
            imputed_path = metadata.get("imputed_pv_path")

            if original_path and imputed_path:
                if Path(original_path).exists() and Path(imputed_path).exists():
                    logger.info(f"✅ Cache hit: {cache_key[:16]}...")
                    return metadata
                else:
                    logger.warning("⚠️ Cache metadata exists but files missing")
                    # Clean up stale cache entry
                    self.redis_client.delete(cache_key)
                    return None

            return None

        except (AttributeError, ConnectionError) as e:
            logger.warning(f"❌ Redis error during cache lookup: {e}")
        except Exception as e:
            logger.error(f"❌ Error checking cache: {e}")

        return None

    def copy_cached_file(self, cached_path: str, destination_path: str) -> bool:
        """
        Copy cached file from PV to destination.

        Args:
            cached_path: Path to cached file in PV
            destination_path: Destination path

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            # Copy file (or create hardlink for efficiency)
            try:
                # Try hardlink first (instant, no disk space)
                os.link(cached_path, destination_path)
                logger.info(f"🔗 Hardlinked: {os.path.basename(destination_path)}")
            except (OSError, PermissionError):
                # Fall back to copy if hardlink fails
                shutil.copy2(cached_path, destination_path)
                logger.info(f"📋 Copied: {os.path.basename(destination_path)}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to copy {cached_path}: {e}")
            return False

    def cache_files(
        self,
        cache_key: str,
        original_file_path: str,
        imputed_file_path: str,
        metadata: Dict,
    ) -> bool:
        """
        Copy files to PV cache and store metadata in Redis.

        Args:
            cache_key: Cache key for this query
            original_file_path: Local path to original file
            imputed_file_path: Local path to imputed file
            metadata: Additional metadata to store

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Check cache size limit
            if self.max_cache_size_bytes:
                cache_size = self._get_cache_size()
                if cache_size > self.max_cache_size_bytes:
                    logger.warning(
                        "⚠️ Cache size limit reached, cleaning old entries..."
                    )
                    self._cleanup_old_entries()

            # Get PV paths for cached files
            date = metadata.get("date", "unknown")
            original_pv_path, imputed_pv_path = self._get_cache_file_paths(
                cache_key, date
            )

            # Copy files to PV cache
            logger.info("📤 Caching files to PV...")
            shutil.copy2(original_file_path, original_pv_path)
            shutil.copy2(imputed_file_path, imputed_pv_path)

            # Store metadata in Redis
            cache_metadata = {
                "original_pv_path": original_pv_path,
                "imputed_pv_path": imputed_pv_path,
                "original_filename": os.path.basename(original_file_path),
                "imputed_filename": os.path.basename(imputed_file_path),
                "file_size_mb": (
                    Path(original_file_path).stat().st_size
                    + Path(imputed_file_path).stat().st_size
                )
                / (1024**2),
                **metadata,
            }

            try:
                self.redis_client.setex(
                    cache_key, self.cache_ttl_seconds, json.dumps(cache_metadata)
                )
                logger.info(
                    f"✅ Cached files in PV: {cache_key[:16]}... (TTL: {self.cache_ttl_seconds}s)"
                )
                return True
            except (AttributeError, ConnectionError) as e:
                logger.warning(f"❌ Redis caching failed: {e}")
                # Clean up PV files if Redis fails
                Path(original_pv_path).unlink(missing_ok=True)
                Path(imputed_pv_path).unlink(missing_ok=True)
                return False

        except Exception as e:
            logger.error(f"❌ Failed to cache files: {e}")
            return False

    def _get_cache_size(self) -> int:
        """Get total size of cache directory in bytes."""
        total_size = 0
        try:
            for path in self.cache_dir.rglob("*.tif"):
                total_size += path.stat().st_size
        except Exception as e:
            logger.warning(f"⚠️ Error calculating cache size: {e}")
        return total_size

    def _cleanup_old_entries(self, target_percent: float = 0.8):
        """Remove oldest cache entries to reach target size."""
        try:
            # Get all cache files with their access times
            files = []
            for path in self.cache_dir.rglob("*.tif"):
                try:
                    files.append((path, path.stat().st_atime))
                except Exception:
                    continue

            if not files:
                return

            # Sort by access time (oldest first)
            files.sort(key=lambda x: x[1])

            # Remove oldest 20% of files
            remove_count = int(len(files) * (1 - target_percent))
            for path, _ in files[:remove_count]:
                try:
                    path.unlink(missing_ok=True)
                    logger.info(f"🗑️ Removed old cache file: {path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {path.name}: {e}")

        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")

    def invalidate_cache(self, cache_key: str) -> bool:
        """
        Remove cache entry and associated PV files.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                metadata = json.loads(cached_data)

                # Delete PV files
                for key in ["original_pv_path", "imputed_pv_path"]:
                    pv_path = metadata.get(key)
                    if pv_path:
                        Path(pv_path).unlink(missing_ok=True)
                        logger.info(f"🗑️ Deleted PV file: {pv_path}")

                # Delete Redis entry
                self.redis_client.delete(cache_key)
                logger.info(f"✅ Invalidated cache: {cache_key[:16]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Failed to invalidate cache: {e}")
            return False
