"""
MCP Coordination Protocols

Implements Contract Net Protocol, blackboard synchronization, and task allocation
for multi-agent coordination in DIGIMON.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import heapq

from Core.Common.Logger import logger
from Core.MCP.mcp_agent_interface import (
    get_agent_interface, MessageType, AgentCapability
)
from Core.MCP.blackboard_system import BlackboardSystem


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ANNOUNCED = "announced"
    BIDDING = "bidding"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskSpec:
    """Task specification for contract net"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    description: str = ""
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher is more important
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority > other.priority  # Higher priority first


@dataclass
class Bid:
    """Agent bid for a task"""
    agent_id: str
    task_id: str
    score: float  # 0.0 to 1.0
    estimated_time: float  # seconds
    capabilities_match: float
    availability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAssignment:
    """Task assignment record"""
    task_id: str
    agent_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    status: TaskStatus = TaskStatus.ASSIGNED


class ContractNetProtocol:
    """
    Implements Contract Net Protocol for task allocation.
    Managers announce tasks, agents bid, best bidder is selected.
    """
    
    def __init__(self):
        self._tasks: Dict[str, TaskSpec] = {}
        self._bids: Dict[str, List[Bid]] = {}  # task_id -> list of bids
        self._assignments: Dict[str, TaskAssignment] = {}
        self._agent_interface = get_agent_interface()
        self._bid_timeout = 2.0  # seconds to wait for bids
        
    async def announce_task(
        self,
        task_type: str,
        description: str,
        required_capabilities: List[str],
        payload: Dict[str, Any],
        priority: int = 0,
        deadline: Optional[datetime] = None
    ) -> str:
        """
        Announce a task to all capable agents.
        
        Returns:
            str: Task ID
        """
        # Convert capabilities
        cap_set = set()
        for cap_str in required_capabilities:
            try:
                cap_set.add(AgentCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")
        
        # Create task
        task = TaskSpec(
            task_type=task_type,
            description=description,
            required_capabilities=cap_set,
            payload=payload,
            priority=priority,
            deadline=deadline
        )
        
        self._tasks[task.task_id] = task
        self._bids[task.task_id] = []
        
        # Find capable agents
        agents = await self._agent_interface.discover_agents()
        capable_agents = []
        
        for agent in agents:
            agent_caps = set(AgentCapability(c) for c in agent["capabilities"] 
                           if c in AgentCapability._value2member_map_)
            if cap_set.issubset(agent_caps):
                capable_agents.append(agent["agent_id"])
        
        logger.info(f"Announcing task {task.task_id} to {len(capable_agents)} agents")
        
        # Send task announcement
        for agent_id in capable_agents:
            await self._agent_interface.send_message(
                sender_id="contract_net_manager",
                recipient_id=agent_id,
                message_type=MessageType.TASK_ASSIGNMENT,
                payload={
                    "action": "call_for_bids",
                    "task": {
                        "task_id": task.task_id,
                        "task_type": task_type,
                        "description": description,
                        "required_capabilities": required_capabilities,
                        "priority": priority,
                        "deadline": deadline.isoformat() if deadline else None
                    }
                }
            )
        
        task.status = TaskStatus.ANNOUNCED
        return task.task_id
    
    async def submit_bid(
        self,
        agent_id: str,
        task_id: str,
        score: float,
        estimated_time: float,
        capabilities_match: float = 1.0,
        availability: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit a bid for a task.
        
        Args:
            agent_id: Bidding agent
            task_id: Task to bid on
            score: Overall bid score (0.0 to 1.0)
            estimated_time: Estimated completion time in seconds
            capabilities_match: How well capabilities match (0.0 to 1.0)
            availability: Agent availability (0.0 to 1.0)
            metadata: Additional bid information
            
        Returns:
            bool: Success status
        """
        if task_id not in self._tasks:
            logger.error(f"Unknown task: {task_id}")
            return False
            
        bid = Bid(
            agent_id=agent_id,
            task_id=task_id,
            score=score,
            estimated_time=estimated_time,
            capabilities_match=capabilities_match,
            availability=availability,
            metadata=metadata or {}
        )
        
        self._bids[task_id].append(bid)
        logger.debug(f"Received bid from {agent_id} for task {task_id}: score={score}")
        
        return True
    
    async def select_winner(
        self,
        task_id: str,
        selection_function: Optional[Callable[[List[Bid]], Optional[Bid]]] = None
    ) -> Optional[str]:
        """
        Select winning bid for a task.
        
        Args:
            task_id: Task to select winner for
            selection_function: Custom selection function (default: highest score)
            
        Returns:
            str: Winning agent ID or None
        """
        if task_id not in self._bids:
            return None
            
        bids = self._bids[task_id]
        if not bids:
            logger.warning(f"No bids received for task {task_id}")
            return None
        
        # Default selection: highest score
        if selection_function is None:
            selection_function = lambda bids: max(bids, key=lambda b: b.score)
        
        winner = selection_function(bids)
        if winner is None:
            return None
        
        # Create assignment
        assignment = TaskAssignment(
            task_id=task_id,
            agent_id=winner.agent_id,
            deadline=self._tasks[task_id].deadline
        )
        self._assignments[task_id] = assignment
        
        # Notify winner
        await self._agent_interface.send_message(
            sender_id="contract_net_manager",
            recipient_id=winner.agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={
                "action": "task_awarded",
                "task_id": task_id,
                "task": self._tasks[task_id].payload
            }
        )
        
        # Notify losers
        for bid in bids:
            if bid.agent_id != winner.agent_id:
                await self._agent_interface.send_message(
                    sender_id="contract_net_manager",
                    recipient_id=bid.agent_id,
                    message_type=MessageType.TASK_ASSIGNMENT,
                    payload={
                        "action": "bid_rejected",
                        "task_id": task_id
                    }
                )
        
        logger.info(f"Task {task_id} awarded to {winner.agent_id} (score: {winner.score})")
        return winner.agent_id
    
    async def wait_for_bids(self, task_id: str, timeout: Optional[float] = None) -> int:
        """
        Wait for bids on a task.
        
        Args:
            task_id: Task to wait for bids on
            timeout: Maximum wait time (default: self._bid_timeout)
            
        Returns:
            int: Number of bids received
        """
        timeout = timeout or self._bid_timeout
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(0.1)
            
        return len(self._bids.get(task_id, []))


class TaskAllocationSystem:
    """
    Manages task allocation across multiple agents.
    Supports priority queuing and load balancing.
    """
    
    def __init__(self):
        self._task_queue: List[TaskSpec] = []  # Priority queue
        self._agent_loads: Dict[str, int] = {}  # Agent -> active task count
        self._contract_net = ContractNetProtocol()
        self._blackboard = SimpleBlackboard()
        self._allocation_lock = asyncio.Lock()
        
    async def submit_task(
        self,
        task_type: str,
        description: str,
        required_capabilities: List[str],
        payload: Dict[str, Any],
        priority: int = 0,
        deadline: Optional[datetime] = None
    ) -> str:
        """
        Submit a task for allocation.
        
        Returns:
            str: Task ID
        """
        # Create task
        cap_set = set()
        for cap_str in required_capabilities:
            try:
                cap_set.add(AgentCapability(cap_str))
            except ValueError:
                pass
                
        task = TaskSpec(
            task_type=task_type,
            description=description,
            required_capabilities=cap_set,
            payload=payload,
            priority=priority,
            deadline=deadline
        )
        
        # Add to priority queue
        async with self._allocation_lock:
            heapq.heappush(self._task_queue, task)
        
        # Trigger allocation
        asyncio.create_task(self._allocate_task(task))
        
        return task.task_id
    
    async def _allocate_task(self, task: TaskSpec):
        """Allocate a single task"""
        # Use contract net for allocation
        task_id = await self._contract_net.announce_task(
            task_type=task.task_type,
            description=task.description,
            required_capabilities=[cap.value for cap in task.required_capabilities],
            payload=task.payload,
            priority=task.priority,
            deadline=task.deadline
        )
        
        # Wait for bids
        num_bids = await self._contract_net.wait_for_bids(task_id)
        
        if num_bids > 0:
            # Select winner considering load balancing
            winner = await self._contract_net.select_winner(
                task_id,
                selection_function=self._load_balanced_selection
            )
            
            if winner:
                # Update agent load
                self._agent_loads[winner] = self._agent_loads.get(winner, 0) + 1
                
                # Update blackboard
                await self._blackboard.write(
                    f"task_allocation/{task_id}",
                    {
                        "task_id": task_id,
                        "agent_id": winner,
                        "status": "assigned",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
    
    def _load_balanced_selection(self, bids: List[Bid]) -> Optional[Bid]:
        """Select bid considering agent load"""
        if not bids:
            return None
            
        # Score bids with load factor
        scored_bids = []
        for bid in bids:
            load = self._agent_loads.get(bid.agent_id, 0)
            load_factor = 1.0 / (1.0 + load * 0.2)  # Reduce score by 20% per active task
            adjusted_score = bid.score * load_factor
            scored_bids.append((adjusted_score, bid))
        
        # Return highest adjusted score
        scored_bids.sort(key=lambda x: x[0], reverse=True)
        return scored_bids[0][1] if scored_bids else None
    
    async def complete_task(self, task_id: str, agent_id: str):
        """Mark task as completed"""
        # Update agent load
        if agent_id in self._agent_loads:
            self._agent_loads[agent_id] = max(0, self._agent_loads[agent_id] - 1)
        
        # Update blackboard
        await self._blackboard.write(
            f"task_allocation/{task_id}",
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


class ParallelTaskExecutor:
    """
    Executes tasks in parallel across multiple agents.
    Handles dependencies and result aggregation.
    """
    
    def __init__(self):
        self._allocation_system = TaskAllocationSystem()
        self._blackboard = SimpleBlackboard()
        self._results: Dict[str, Any] = {}
        
    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]],
        aggregation_function: Optional[Callable[[List[Any]], Any]] = None
    ) -> Any:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of task specifications
            aggregation_function: Function to aggregate results
            
        Returns:
            Aggregated result
        """
        task_ids = []
        
        # Submit all tasks
        for task in tasks:
            task_id = await self._allocation_system.submit_task(
                task_type=task.get("type", "generic"),
                description=task.get("description", ""),
                required_capabilities=task.get("capabilities", []),
                payload=task.get("payload", {}),
                priority=task.get("priority", 0)
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = await self._wait_for_results(task_ids)
        
        # Aggregate results
        if aggregation_function:
            return aggregation_function(results)
        else:
            return results
    
    async def _wait_for_results(
        self,
        task_ids: List[str],
        timeout: float = 30.0
    ) -> List[Any]:
        """Wait for task results"""
        results = []
        start_time = asyncio.get_event_loop().time()
        
        while len(results) < len(task_ids):
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning("Timeout waiting for task results")
                break
                
            for task_id in task_ids:
                if task_id not in self._results:
                    # Check blackboard for result
                    result = await self._blackboard.read(f"task_results/{task_id}")
                    if result:
                        self._results[task_id] = result
                        results.append(result)
            
            await asyncio.sleep(0.1)
        
        return results


# Simple blackboard wrapper for coordination
class SimpleBlackboard:
    """Simple blackboard wrapper for coordination protocols"""
    
    def __init__(self):
        self._data = {}
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the blackboard"""
        pass
        
    async def stop(self):
        """Stop the blackboard"""
        pass
        
    async def write(self, key: str, value: Any):
        """Write to blackboard"""
        async with self._lock:
            self._data[key] = value
            
    async def read(self, key: str) -> Any:
        """Read from blackboard"""
        async with self._lock:
            return self._data.get(key)


# Global instances
_contract_net = None
_task_allocation = None
_parallel_executor = None
_blackboard = None


def get_contract_net() -> ContractNetProtocol:
    """Get global contract net instance"""
    global _contract_net
    if _contract_net is None:
        _contract_net = ContractNetProtocol()
    return _contract_net


def get_task_allocation() -> TaskAllocationSystem:
    """Get global task allocation instance"""
    global _task_allocation
    if _task_allocation is None:
        _task_allocation = TaskAllocationSystem()
    return _task_allocation


def get_parallel_executor() -> ParallelTaskExecutor:
    """Get global parallel executor instance"""
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelTaskExecutor()
    return _parallel_executor


def get_blackboard() -> SimpleBlackboard:
    """Get global blackboard instance"""
    global _blackboard
    if _blackboard is None:
        _blackboard = SimpleBlackboard()
    return _blackboard