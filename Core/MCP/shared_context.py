"""
Shared Context Store for MCP Implementation

This module provides thread-safe shared context storage for cross-request state
management in the DIGIMON MCP system.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Represents a single context entry with metadata"""
    key: str
    value: Any
    created_at: float
    updated_at: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    session_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
        
    def access(self):
        """Record access to this entry"""
        self.access_count += 1
        self.updated_at = time.time()


@dataclass
class SessionContext:
    """Represents a session with its own context namespace"""
    session_id: str
    created_at: float
    last_accessed: float
    entries: Dict[str, ContextEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self):
        """Update last accessed time"""
        self.last_accessed = time.time()
        
    def is_expired(self, timeout_seconds: float = 3600) -> bool:
        """Check if session has expired"""
        return time.time() - self.last_accessed > timeout_seconds


class SharedContextStore:
    """Thread-safe shared context store with session management"""
    
    def __init__(self, gc_interval_seconds: float = 60.0):
        # Thread-safe storage
        self._lock = threading.RLock()
        self._global_context: Dict[str, ContextEntry] = {}
        self._sessions: Dict[str, SessionContext] = {}
        
        # Performance tracking
        self._operation_times: List[float] = []
        self._max_operation_samples = 1000
        
        # Garbage collection
        self._gc_interval = gc_interval_seconds
        self._gc_task = None
        self._running = False
        
        # Weak references for memory efficiency
        self._weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
    async def start(self):
        """Start the context store with background garbage collection"""
        self._running = True
        self._gc_task = asyncio.create_task(self._garbage_collector())
        logger.info("SharedContextStore started")
        
    async def stop(self):
        """Stop the context store"""
        self._running = False
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
        logger.info("SharedContextStore stopped")
        
    async def _garbage_collector(self):
        """Background task to clean up expired entries"""
        while self._running:
            try:
                await asyncio.sleep(self._gc_interval)
                removed_count = self._cleanup_expired()
                if removed_count > 0:
                    logger.debug(f"Garbage collector removed {removed_count} expired entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in garbage collector: {e}")
                
    def _cleanup_expired(self) -> int:
        """Remove expired entries and sessions"""
        start_time = time.time()
        removed_count = 0
        
        with self._lock:
            # Clean global context
            expired_keys = [
                key for key, entry in self._global_context.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._global_context[key]
                removed_count += 1
                
            # Clean sessions
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if session.is_expired()
            ]
            for sid in expired_sessions:
                # Clean session entries first
                removed_count += len(self._sessions[sid].entries)
                del self._sessions[sid]
                removed_count += 1
                
        self._record_operation_time(time.time() - start_time)
        return removed_count
        
    def _record_operation_time(self, duration: float):
        """Record operation duration for performance tracking"""
        self._operation_times.append(duration)
        if len(self._operation_times) > self._max_operation_samples:
            self._operation_times.pop(0)
            
    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None,
            session_id: Optional[str] = None):
        """Set a context value"""
        start_time = time.time()
        
        with self._lock:
            entry = ContextEntry(
                key=key,
                value=value,
                created_at=time.time(),
                updated_at=time.time(),
                ttl_seconds=ttl_seconds,
                session_id=session_id
            )
            
            if session_id:
                # Store in session context
                if session_id not in self._sessions:
                    self._sessions[session_id] = SessionContext(
                        session_id=session_id,
                        created_at=time.time(),
                        last_accessed=time.time()
                    )
                self._sessions[session_id].entries[key] = entry
                self._sessions[session_id].access()
            else:
                # Store in global context
                self._global_context[key] = entry
                
        self._record_operation_time(time.time() - start_time)
        
    def get(self, key: str, default: Any = None, session_id: Optional[str] = None) -> Any:
        """Get a context value"""
        start_time = time.time()
        
        with self._lock:
            entry = None
            
            if session_id and session_id in self._sessions:
                # Look in session context first
                session = self._sessions[session_id]
                session.access()
                entry = session.entries.get(key)
                
            if entry is None:
                # Fall back to global context
                entry = self._global_context.get(key)
                
            if entry is None:
                result = default
            elif entry.is_expired():
                # Remove expired entry
                if session_id and session_id in self._sessions:
                    self._sessions[session_id].entries.pop(key, None)
                else:
                    self._global_context.pop(key, None)
                result = default
            else:
                entry.access()
                result = entry.value
                
        self._record_operation_time(time.time() - start_time)
        return result
        
    def update(self, key: str, value: Any, session_id: Optional[str] = None) -> bool:
        """Update an existing context value"""
        start_time = time.time()
        
        with self._lock:
            entry = None
            
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.access()
                entry = session.entries.get(key)
            else:
                entry = self._global_context.get(key)
                
            if entry and not entry.is_expired():
                entry.value = value
                entry.updated_at = time.time()
                entry.access()
                success = True
            else:
                success = False
                
        self._record_operation_time(time.time() - start_time)
        return success
        
    def delete(self, key: str, session_id: Optional[str] = None) -> bool:
        """Delete a context value"""
        start_time = time.time()
        
        with self._lock:
            deleted = False
            
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.access()
                if key in session.entries:
                    del session.entries[key]
                    deleted = True
            elif key in self._global_context:
                del self._global_context[key]
                deleted = True
                
        self._record_operation_time(time.time() - start_time)
        return deleted
        
    def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> SessionContext:
        """Create a new session"""
        start_time = time.time()
        
        with self._lock:
            session = SessionContext(
                session_id=session_id,
                created_at=time.time(),
                last_accessed=time.time(),
                metadata=metadata or {}
            )
            self._sessions[session_id] = session
            
        self._record_operation_time(time.time() - start_time)
        return session
        
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get a session by ID"""
        start_time = time.time()
        
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired():
                session.access()
                result = session
            else:
                result = None
                
        self._record_operation_time(time.time() - start_time)
        return result
        
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its entries"""
        start_time = time.time()
        
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                deleted = True
            else:
                deleted = False
                
        self._record_operation_time(time.time() - start_time)
        return deleted
        
    def get_stats(self) -> Dict[str, Any]:
        """Get context store statistics"""
        with self._lock:
            global_count = len(self._global_context)
            session_count = len(self._sessions)
            total_entries = global_count + sum(
                len(s.entries) for s in self._sessions.values()
            )
            
            # Calculate average operation time
            if self._operation_times:
                avg_op_time_ms = sum(self._operation_times) * 1000 / len(self._operation_times)
                max_op_time_ms = max(self._operation_times) * 1000
            else:
                avg_op_time_ms = 0
                max_op_time_ms = 0
                
        return {
            "global_entries": global_count,
            "sessions": session_count,
            "total_entries": total_entries,
            "avg_operation_ms": round(avg_op_time_ms, 3),
            "max_operation_ms": round(max_op_time_ms, 3),
            "operation_samples": len(self._operation_times)
        }
        
    def export_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export context for debugging or persistence"""
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                return {
                    "session_id": session_id,
                    "created_at": session.created_at,
                    "entries": {
                        key: {
                            "value": entry.value,
                            "created_at": entry.created_at,
                            "access_count": entry.access_count
                        }
                        for key, entry in session.entries.items()
                    }
                }
            else:
                return {
                    "global": True,
                    "entries": {
                        key: {
                            "value": entry.value,
                            "created_at": entry.created_at,
                            "access_count": entry.access_count
                        }
                        for key, entry in self._global_context.items()
                    }
                }


# Global singleton instance
_shared_context_store: Optional[SharedContextStore] = None
_store_lock = threading.Lock()


def get_shared_context() -> SharedContextStore:
    """Get the global shared context store instance"""
    global _shared_context_store
    
    if _shared_context_store is None:
        with _store_lock:
            if _shared_context_store is None:
                _shared_context_store = SharedContextStore()
                
    return _shared_context_store