"""
MCP Agent Interface

Enables agents to communicate and coordinate via MCP protocol.
Provides agent registration, discovery, and messaging capabilities.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from Core.Common.Logger import logger
from Core.MCP.shared_context import get_shared_context


class AgentCapability(Enum):
    """Standard agent capabilities"""
    ENTITY_EXTRACTION = "entity_extraction"
    GRAPH_ANALYSIS = "graph_analysis"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    TEXT_PROCESSING = "text_processing"
    VECTOR_SEARCH = "vector_search"
    GRAPH_CONSTRUCTION = "graph_construction"
    QUERY_PLANNING = "query_planning"
    RESULT_AGGREGATION = "result_aggregation"


class MessageType(Enum):
    """Agent message types"""
    REGISTER = "register"
    DISCOVER = "discover"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_ACKNOWLEDGMENT = "task_acknowledgment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    CAPABILITY_QUERY = "capability_query"
    HEARTBEAT = "heartbeat"


@dataclass
class AgentInfo:
    """Agent metadata"""
    agent_id: str
    name: str
    capabilities: Set[AgentCapability]
    status: str = "active"
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentMessage:
    """Message between agents"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.STATUS_UPDATE
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None = broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    requires_acknowledgment: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "requires_acknowledgment": self.requires_acknowledgment
        }


class MCPAgentInterface:
    """
    MCP-based agent interface for multi-agent coordination.
    Provides registration, discovery, and communication services.
    """
    
    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._message_queue: Dict[str, List[AgentMessage]] = {}
        self._shared_context = get_shared_context()
        self._heartbeat_interval = 30.0  # seconds
        self._heartbeat_task = None
        
    async def start(self):
        """Start the agent interface"""
        await self._shared_context.start()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        logger.info("MCP Agent Interface started")
        
    async def stop(self):
        """Stop the agent interface"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        await self._shared_context.stop()
        logger.info("MCP Agent Interface stopped")
        
    async def register_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an agent with the system.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of capability strings
            metadata: Optional additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Convert string capabilities to enum
            cap_set = set()
            for cap_str in capabilities:
                try:
                    cap_set.add(AgentCapability(cap_str))
                except ValueError:
                    logger.warning(f"Unknown capability: {cap_str}")
                    
            # Create agent info
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=name,
                capabilities=cap_set,
                metadata=metadata or {}
            )
            
            # Register in local registry
            self._agents[agent_id] = agent_info
            
            # Initialize message queue
            self._message_queue[agent_id] = []
            
            # Store in shared context for cross-server visibility
            agents_dict = self._shared_context.get("agents", default={})
            agents_dict[agent_id] = agent_info.to_dict()
            self._shared_context.set("agents", agents_dict)
            
            logger.info(f"Agent registered: {agent_id} with capabilities: {capabilities}")
            
            # Broadcast registration
            await self._broadcast_message(AgentMessage(
                message_type=MessageType.REGISTER,
                sender_id=agent_id,
                payload=agent_info.to_dict()
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    async def discover_agents(
        self,
        capabilities: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover registered agents.
        
        Args:
            capabilities: Filter by required capabilities
            status: Filter by status (active, inactive)
            
        Returns:
            List of agent info dictionaries
        """
        # Get from shared context to see all agents
        agents_dict = self._shared_context.get("agents", default={})
        
        # Update local registry
        for agent_id, agent_data in agents_dict.items():
            if agent_id not in self._agents:
                # Reconstruct AgentInfo
                cap_set = set()
                for cap_str in agent_data.get("capabilities", []):
                    try:
                        cap_set.add(AgentCapability(cap_str))
                    except ValueError:
                        pass
                        
                self._agents[agent_id] = AgentInfo(
                    agent_id=agent_id,
                    name=agent_data.get("name", "Unknown"),
                    capabilities=cap_set,
                    status=agent_data.get("status", "unknown"),
                    metadata=agent_data.get("metadata", {})
                )
        
        # Filter agents
        results = []
        for agent in self._agents.values():
            # Check capabilities filter
            if capabilities:
                required_caps = {AgentCapability(c) for c in capabilities if c in AgentCapability._value2member_map_}
                if not required_caps.issubset(agent.capabilities):
                    continue
                    
            # Check status filter
            if status and agent.status != status:
                continue
                
            results.append(agent.to_dict())
            
        return results
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        requires_acknowledgment: bool = False
    ) -> str:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_type: Type of message
            payload: Message payload
            requires_acknowledgment: Whether to expect acknowledgment
            
        Returns:
            str: Message ID
        """
        message = AgentMessage(
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            requires_acknowledgment=requires_acknowledgment
        )
        
        # Add to recipient's queue
        if recipient_id in self._message_queue:
            self._message_queue[recipient_id].append(message)
        else:
            # Store in shared context for cross-server delivery
            messages_dict = self._shared_context.get(f"messages_{recipient_id}", default=[])
            messages_dict.append(message.to_dict())
            self._shared_context.set(f"messages_{recipient_id}", messages_dict)
        
        logger.debug(f"Message {message.message_id} sent from {sender_id} to {recipient_id}")
        
        # Call handler if registered
        if recipient_id in self._message_handlers:
            asyncio.create_task(self._message_handlers[recipient_id](message))
        
        return message.message_id
    
    async def receive_messages(
        self,
        agent_id: str,
        message_types: Optional[List[MessageType]] = None,
        limit: int = 10
    ) -> List[AgentMessage]:
        """
        Receive pending messages for an agent.
        
        Args:
            agent_id: Receiving agent ID
            message_types: Filter by message types
            limit: Maximum messages to return
            
        Returns:
            List of messages
        """
        # Check local queue
        local_messages = self._message_queue.get(agent_id, [])
        
        # Check shared context
        shared_messages = self._shared_context.get(f"messages_{agent_id}", default=[])
        for msg_dict in shared_messages:
            msg = AgentMessage(
                message_id=msg_dict["message_id"],
                message_type=MessageType(msg_dict["message_type"]),
                sender_id=msg_dict["sender_id"],
                recipient_id=msg_dict.get("recipient_id"),
                payload=msg_dict["payload"],
                requires_acknowledgment=msg_dict.get("requires_acknowledgment", False)
            )
            local_messages.append(msg)
        
        # Clear shared messages
        self._shared_context.set(f"messages_{agent_id}", [])
        
        # Filter by type if specified
        if message_types:
            filtered = [m for m in local_messages if m.message_type in message_types]
        else:
            filtered = local_messages
        
        # Get requested messages
        messages = filtered[:limit]
        
        # Update queue with remaining
        remaining = filtered[limit:]
        self._message_queue[agent_id] = remaining
        
        return messages
    
    def register_message_handler(self, agent_id: str, handler: Callable):
        """
        Register a callback for incoming messages.
        
        Args:
            agent_id: Agent to handle messages for
            handler: Async function to call with messages
        """
        self._message_handlers[agent_id] = handler
        
    async def update_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp"""
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = datetime.utcnow()
            
            # Update in shared context
            agents_dict = self._shared_context.get("agents", default={})
            if agent_id in agents_dict:
                agents_dict[agent_id]["last_heartbeat"] = datetime.utcnow().isoformat()
                self._shared_context.set("agents", agents_dict)
    
    async def _broadcast_message(self, message: AgentMessage):
        """Broadcast a message to all agents"""
        # Note: For now, we're not adding broadcast messages to queues
        # This could be enabled with a separate broadcast queue if needed
        pass
                    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and update status"""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                
                now = datetime.utcnow()
                for agent in self._agents.values():
                    time_since_heartbeat = (now - agent.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self._heartbeat_interval * 3:
                        if agent.status == "active":
                            agent.status = "inactive"
                            logger.warning(f"Agent {agent.agent_id} marked inactive")
                            
                            # Update shared context
                            agents_dict = self._shared_context.get("agents", default={})
                            if agent.agent_id in agents_dict:
                                agents_dict[agent.agent_id]["status"] = "inactive"
                                self._shared_context.set("agents", agents_dict)
                                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")


# Global instance
_agent_interface = None


def get_agent_interface() -> MCPAgentInterface:
    """Get the global agent interface instance"""
    global _agent_interface
    if _agent_interface is None:
        _agent_interface = MCPAgentInterface()
    return _agent_interface