"""Cache manager for storing Notion memories locally"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from charisma.utils.logger import get_logger

logger = get_logger()


class CacheManager:
    """Manages local cache of Notion memories"""

    def __init__(self, cache_dir: str = "./data/memories"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        logger.info(f"Cache manager initialized at: {self.cache_dir}")

    def save_memories(self, memories: List[Dict]) -> Dict[str, str]:
        """
        Save memories to local cache
        
        Args:
            memories: List of memory dictionaries from Notion
            
        Returns:
            Dictionary mapping memory IDs to file paths
        """
        memory_map = {}
        
        for memory in memories:
            memory_id = memory.get("id", "")
            if not memory_id:
                logger.warning("Skipping memory without ID")
                continue
            
            # Create filename from memory ID
            filename = f"{memory_id}.json"
            filepath = self.cache_dir / filename
            
            # Save memory to file
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(memory, f, indent=2, ensure_ascii=False, default=str)
                
                memory_map[memory_id] = str(filepath)
                logger.debug(f"Saved memory {memory_id} to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save memory {memory_id}: {e}")
        
        # Update index
        self._update_index(memory_map)
        
        logger.success(f"✅ Saved {len(memory_map)} memories to local cache")
        return memory_map

    def load_memories(self, memory_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Load memories from local cache
        
        Args:
            memory_ids: Optional list of specific memory IDs to load.
                       If None, loads all cached memories.
            
        Returns:
            List of memory dictionaries
        """
        memories = []
        
        # Load index
        index = self._load_index()
        
        if not index:
            logger.warning("No cached memories found")
            return memories
        
        # Determine which memories to load
        ids_to_load = memory_ids if memory_ids else list(index.keys())
        
        for memory_id in ids_to_load:
            if memory_id not in index:
                logger.warning(f"Memory {memory_id} not found in cache")
                continue
            
            filepath = Path(index[memory_id])
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                    memories.append(memory)
                    logger.debug(f"Loaded memory {memory_id} from {filepath}")
            except FileNotFoundError:
                logger.error(f"Memory file not found: {filepath}")
            except Exception as e:
                logger.error(f"Failed to load memory {memory_id}: {e}")
        
        logger.info(f"Loaded {len(memories)} memories from cache")
        return memories

    def get_cached_memory_ids(self) -> List[str]:
        """Get list of all cached memory IDs"""
        index = self._load_index()
        return list(index.keys())

    def is_cached(self, memory_id: str) -> bool:
        """Check if a memory is cached"""
        index = self._load_index()
        return memory_id in index

    def clear_cache(self) -> None:
        """Clear all cached memories"""
        try:
            for file in self.cache_dir.glob("*.json"):
                if file.name != "index.json":
                    file.unlink()
            
            if self.index_file.exists():
                self.index_file.unlink()
            
            logger.success("✅ Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def _update_index(self, memory_map: Dict[str, str]) -> None:
        """Update the cache index file"""
        # Load existing index
        index = self._load_index()
        
        # Update with new memories
        index.update(memory_map)
        
        # Add metadata
        metadata = {
            "memories": index,
            "last_updated": datetime.now().isoformat(),
            "total_memories": len(index)
        }
        
        # Save index
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug("Index updated successfully")
        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def _load_index(self) -> Dict[str, str]:
        """Load the cache index file"""
        if not self.index_file.exists():
            return {}
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("memories", {})
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return {}

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        index = self._load_index()
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            metadata = {}
        
        # Calculate cache size
        total_size = 0
        for filepath in index.values():
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
        
        return {
            "total_memories": len(index),
            "cache_size_bytes": total_size,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "last_updated": metadata.get("last_updated", "Unknown"),
            "cache_dir": str(self.cache_dir)
        }
