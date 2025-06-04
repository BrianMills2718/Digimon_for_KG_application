"""
Shared context store for MCP communication
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class ContextSession:
    """Context session for tracking state across requests"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0
        self._lock = asyncio.Lock()
        
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from session context"""
        async with self._lock:
            self.last_accessed = datetime.utcnow()
            self.access_count += 1
            return self.data.get(key, default)
    
    async def set(self, key: str, value: Any):
        """Set value in session context"""
        async with self._lock:
            self.last_accessed = datetime.utcnow()
            self.data[key] = value
    
    async def update(self, updates: Dict[str, Any]):
        """Update multiple values in session context"""
        async with self._lock:
            self.last_accessed = datetime.utcnow()
            self.data.update(updates)
    
    async def clear(self):
        """Clear session data"""
        async with self._lock:
            self.data.clear()
            self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export session data"""
        return {
            'session_id': self.session_id,
            'data': self.data.copy(),
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count
        }


class SharedContextStore:
    """
    Centralized context storage for cross-tool communication
    Supports TTL-based cleanup and persistence
    """
    
    def __init__(self, ttl_minutes: int = 60, cleanup_interval_minutes: int = 5):
        self.sessions: Dict[str, ContextSession] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval.total_seconds())
                    await self._cleanup_expired_sessions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}", exc_info=True)
        
        try:
            self._cleanup_task = asyncio.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, skip cleanup task
            logger.debug("No event loop available for cleanup task")
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.utcnow()
            expired = []
            
            for session_id, session in self.sessions.items():
                if now - session.last_accessed > self.ttl:
                    expired.append(session_id)
            
            for session_id in expired:
                logger.debug(f"Removing expired session: {session_id}")
                del self.sessions[session_id]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    async def get_or_create_session(self, session_id: str) -> ContextSession:
        """Get existing session or create new one"""
        async with self._lock:
            if session_id not in self.sessions:
                logger.debug(f"Creating new session: {session_id}")
                self.sessions[session_id] = ContextSession(session_id)
            return self.sessions[session_id]
    
    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get full context for a session"""
        session = await self.get_or_create_session(session_id)
        return session.data.copy()
    
    async def update_context(self, session_id: str, updates: Dict[str, Any]):
        """Update context for a session"""
        session = await self.get_or_create_session(session_id)
        await session.update(updates)
    
    async def get_value(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get specific value from session context"""
        session = await self.get_or_create_session(session_id)
        return await session.get(key, default)
    
    async def set_value(self, session_id: str, key: str, value: Any):
        """Set specific value in session context"""
        session = await self.get_or_create_session(session_id)
        await session.set(key, value)
    
    async def clear_session(self, session_id: str):
        """Clear all data for a session"""
        if session_id in self.sessions:
            await self.sessions[session_id].clear()
    
    async def delete_session(self, session_id: str):
        """Delete a session entirely"""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
    
    async def export_sessions(self) -> Dict[str, Any]:
        """Export all sessions for persistence"""
        async with self._lock:
            return {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
    
    async def import_sessions(self, data: Dict[str, Any]):
        """Import sessions from persisted data"""
        async with self._lock:
            for session_id, session_data in data.items():
                session = ContextSession(session_id)
                session.data = session_data['data']
                session.created_at = datetime.fromisoformat(session_data['created_at'])
                session.last_accessed = datetime.fromisoformat(session_data['last_accessed'])
                session.access_count = session_data['access_count']
                self.sessions[session_id] = session
            logger.info(f"Imported {len(data)} sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context store statistics"""
        total_size = sum(
            len(json.dumps(session.data))
            for session in self.sessions.values()
        )
        
        return {
            'session_count': len(self.sessions),
            'total_size_bytes': total_size,
            'ttl_minutes': self.ttl.total_seconds() / 60,
            'oldest_session': min(
                (s.created_at for s in self.sessions.values()),
                default=None
            ),
            'most_accessed': max(
                ((s.session_id, s.access_count) for s in self.sessions.values()),
                key=lambda x: x[1],
                default=(None, 0)
            )
        }
    
    async def close(self):
        """Clean up resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass