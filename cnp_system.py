"""
Contract Net Protocol - Warehouse Logistics System
A real-world implementation showing CNP integration with detailed metrics tracking
"""

import time
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta

class MessageType(Enum):
    TASK_ANNOUNCEMENT = "task_announcement"
    BID = "bid"
    AWARD = "award"
    REJECT = "reject"
    COMPLETE = "complete"
    REFUSE = "refuse"

class TaskStatus(Enum):
    ANNOUNCED = "announced"
    BIDDING = "bidding"
    AWARDED = "awarded"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Message:
    """Message passed between agents"""
    msg_id: str
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    task_id: str
    content: Dict
    timestamp: float

@dataclass
class Task:
    """Represents a warehouse task"""
    task_id: str
    task_type: str  # 'pick', 'pack', 'transport', 'sort'
    priority: int  # 1-5, 5 being highest
    location: Tuple[int, int]  # warehouse coordinates
    required_skills: List[str]
    estimated_duration: float  # in seconds
    deadline: float  # timestamp
    created_at: float
    status: TaskStatus = TaskStatus.ANNOUNCED
    assigned_agent: Optional[str] = None
    completed_at: Optional[float] = None

@dataclass
class Bid:
    """Bid from an agent for a task"""
    bid_id: str
    task_id: str
    agent_id: str
    cost: float  # computed cost based on distance, workload, etc.
    estimated_time: float
    confidence: float  # 0-1, agent's confidence in completing the task
    timestamp: float

@dataclass
class PerformanceMetrics:
    """Detailed metrics for CNP execution"""
    task_id: str
    announcement_time: float
    bidding_start_time: float
    bidding_end_time: float
    evaluation_start_time: float
    evaluation_end_time: float
    award_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    num_bids_received: int = 0
    winning_bid_cost: Optional[float] = None
    average_bid_cost: Optional[float] = None
    
    # Decision processing times
    bid_generation_times: List[float] = field(default_factory=list)
    bid_evaluation_time: Optional[float] = None
    
    # Outcome
    task_completed: bool = False
    actual_completion_time: Optional[float] = None

class ContractorAgent:
    """Manager agent that announces tasks and awards contracts"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active_tasks: Dict[str, Task] = {}
        self.pending_bids: Dict[str, List[Bid]] = {}
        self.awarded_tasks: Dict[str, str] = {}  # task_id -> agent_id
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.message_log: List[Message] = []
        
    def announce_task(self, task: Task, agents: List['ContracteeAgent']) -> PerformanceMetrics:
        """Announce a task to all agents and collect bids"""
        print(f"\n{'='*60}")
        print(f"[CONTRACTOR] Announcing task {task.task_id}")
        print(f"  Type: {task.task_type}, Priority: {task.priority}")
        print(f"  Location: {task.location}, Skills: {task.required_skills}")
        
        # Initialize metrics
        metrics = PerformanceMetrics(
            task_id=task.task_id,
            announcement_time=time.time(),
            bidding_start_time=time.time(),
            bidding_end_time=0,
            evaluation_start_time=0,
            evaluation_end_time=0
        )
        
        self.active_tasks[task.task_id] = task
        self.pending_bids[task.task_id] = []
        self.metrics[task.task_id] = metrics
        
        # Send announcement to all agents
        for agent in agents:
            msg = Message(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.TASK_ANNOUNCEMENT,
                sender_id=self.agent_id,
                receiver_id=agent.agent_id,
                task_id=task.task_id,
                content={'task': task},
                timestamp=time.time()
            )
            self.message_log.append(msg)
            
        # Collect bids with realistic delays
        print(f"\n[BIDDING PHASE] Waiting for bids...")
        for agent in agents:
            bid_start = time.time()
            bid = agent.evaluate_and_bid(task)
            bid_time = time.time() - bid_start
            metrics.bid_generation_times.append(bid_time)
            
            if bid:
                self.pending_bids[task.task_id].append(bid)
                print(f"  ✓ Bid from {agent.agent_id}: cost={bid.cost:.2f}, time={bid.estimated_time:.2f}s, confidence={bid.confidence:.2f}")
            else:
                print(f"  ✗ {agent.agent_id} refused (skill/capacity mismatch)")
        
        metrics.bidding_end_time = time.time()
        metrics.num_bids_received = len(self.pending_bids[task.task_id])
        
        return metrics
    
    def evaluate_bids(self, task_id: str) -> Optional[Tuple[Bid, 'ContracteeAgent']]:
        """Evaluate all bids and select winner using multi-criteria decision making"""
        print(f"\n[EVALUATION PHASE] Evaluating {len(self.pending_bids[task_id])} bids...")
        
        metrics = self.metrics[task_id]
        metrics.evaluation_start_time = time.time()
        
        bids = self.pending_bids[task_id]
        if not bids:
            print("  No bids received!")
            metrics.evaluation_end_time = time.time()
            return None
        
        task = self.active_tasks[task_id]
        
        # Calculate average bid cost
        avg_cost = sum(b.cost for b in bids) / len(bids)
        metrics.average_bid_cost = avg_cost
        
        # Multi-criteria scoring
        scored_bids = []
        for bid in bids:
            # Normalize factors (lower is better for cost and time)
            cost_score = 1 - (bid.cost / (max(b.cost for b in bids) + 0.001))
            time_score = 1 - (bid.estimated_time / (max(b.estimated_time for b in bids) + 0.001))
            confidence_score = bid.confidence
            
            # Weighted scoring (adjustable parameters)
            if task.priority >= 4:
                # High priority: favor time and confidence
                total_score = 0.2 * cost_score + 0.4 * time_score + 0.4 * confidence_score
            else:
                # Normal priority: balanced
                total_score = 0.4 * cost_score + 0.3 * time_score + 0.3 * confidence_score
            
            scored_bids.append((total_score, bid))
            print(f"  {bid.agent_id}: score={total_score:.3f} (cost={cost_score:.2f}, time={time_score:.2f}, conf={confidence_score:.2f})")
        
        # Select winner
        scored_bids.sort(reverse=True, key=lambda x: x[0])
        winning_score, winning_bid = scored_bids[0]
        
        metrics.evaluation_end_time = time.time()
        metrics.bid_evaluation_time = metrics.evaluation_end_time - metrics.evaluation_start_time
        metrics.winning_bid_cost = winning_bid.cost
        
        print(f"\n  🏆 Winner: {winning_bid.agent_id} (score={winning_score:.3f})")
        
        return winning_bid
    
    def award_contract(self, task_id: str, winning_bid: Bid, agents: Dict[str, 'ContracteeAgent']):
        """Award the contract to the winning bidder"""
        metrics = self.metrics[task_id]
        metrics.award_time = time.time()
        
        task = self.active_tasks[task_id]
        task.assigned_agent = winning_bid.agent_id
        task.status = TaskStatus.AWARDED
        
        self.awarded_tasks[task_id] = winning_bid.agent_id
        
        # Send award message
        award_msg = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=MessageType.AWARD,
            sender_id=self.agent_id,
            receiver_id=winning_bid.agent_id,
            task_id=task_id,
            content={'bid': winning_bid},
            timestamp=time.time()
        )
        self.message_log.append(award_msg)
        
        # Send reject messages to others
        for bid in self.pending_bids[task_id]:
            if bid.agent_id != winning_bid.agent_id:
                reject_msg = Message(
                    msg_id=str(uuid.uuid4()),
                    msg_type=MessageType.REJECT,
                    sender_id=self.agent_id,
                    receiver_id=bid.agent_id,
                    task_id=task_id,
                    content={'reason': 'bid_not_selected'},
                    timestamp=time.time()
                )
                self.message_log.append(reject_msg)
        
        print(f"\n[AWARD] Contract awarded to {winning_bid.agent_id}")
        
        # Execute task
        agent = agents[winning_bid.agent_id]
        actual_time = agent.execute_task(task)
        
        metrics.completion_time = time.time()
        metrics.actual_completion_time = actual_time
        metrics.task_completed = True
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        
        print(f"[COMPLETE] Task {task_id} completed in {actual_time:.2f}s")

class ContracteeAgent:
    """Worker agent that bids on and executes tasks"""
    
    def __init__(self, agent_id: str, agent_type: str, skills: List[str], 
                 location: Tuple[int, int], max_capacity: int = 3):
        self.agent_id = agent_id
        self.agent_type = agent_type  # 'robot', 'forklift', 'drone', etc.
        self.skills = skills
        self.location = location
        self.max_capacity = max_capacity
        self.current_workload = 0
        self.completed_tasks = []
        self.efficiency_factor = random.uniform(0.8, 1.2)  # Individual performance variation
        
    def evaluate_and_bid(self, task: Task) -> Optional[Bid]:
        """Evaluate task and generate a bid if capable"""
        # Simulate thinking time
        time.sleep(random.uniform(0.01, 0.05))
        
        # Check if agent has required skills
        if not all(skill in self.skills for skill in task.required_skills):
            return None
        
        # Check capacity
        if self.current_workload >= self.max_capacity:
            return None
        
        # Calculate cost based on multiple factors
        distance = self._calculate_distance(task.location)
        travel_time = distance * 0.1  # 0.1 seconds per unit distance
        
        # Cost components
        distance_cost = distance * 2  # Cost per unit distance
        workload_penalty = self.current_workload * 10  # Prefer less loaded agents
        urgency_bonus = (5 - task.priority) * 5  # Lower cost for urgent tasks
        
        total_cost = distance_cost + workload_penalty + urgency_bonus
        
        # Estimated time considering agent's efficiency
        estimated_time = (travel_time + task.estimated_duration) * self.efficiency_factor
        
        # Confidence based on workload and skills
        skill_confidence = 0.9 if len(self.skills) > len(task.required_skills) else 0.7
        workload_confidence = 1.0 - (self.current_workload / self.max_capacity) * 0.3
        confidence = skill_confidence * workload_confidence
        
        bid = Bid(
            bid_id=str(uuid.uuid4()),
            task_id=task.task_id,
            agent_id=self.agent_id,
            cost=total_cost,
            estimated_time=estimated_time,
            confidence=confidence,
            timestamp=time.time()
        )
        
        return bid
    
    def execute_task(self, task: Task) -> float:
        """Execute the awarded task"""
        self.current_workload += 1
        
        # Simulate task execution with some variability
        execution_time = task.estimated_duration * self.efficiency_factor * random.uniform(0.9, 1.1)
        time.sleep(min(execution_time, 0.1))  # Simulate work (capped for demo)
        
        self.completed_tasks.append(task.task_id)
        self.current_workload -= 1
        self.location = task.location  # Agent moves to task location
        
        return execution_time
    
    def _calculate_distance(self, target: Tuple[int, int]) -> float:
        """Calculate Manhattan distance to target"""
        return abs(self.location[0] - target[0]) + abs(self.location[1] - target[1])

class CNPSystem:
    """Main system that orchestrates the Contract Net Protocol"""
    
    def __init__(self):
        self.contractor = ContractorAgent("warehouse_manager")
        self.agents: Dict[str, ContracteeAgent] = {}
        self.tasks: List[Task] = []
        self.all_metrics: List[PerformanceMetrics] = []
        
    def add_agent(self, agent: ContracteeAgent):
        """Add a worker agent to the system"""
        self.agents[agent.agent_id] = agent
        
    def create_task(self, task_type: str, priority: int, location: Tuple[int, int],
                   skills: List[str], duration: float) -> Task:
        """Create a new task"""
        task = Task(
            task_id=f"task_{len(self.tasks):03d}",
            task_type=task_type,
            priority=priority,
            location=location,
            required_skills=skills,
            estimated_duration=duration,
            deadline=time.time() + 3600,  # 1 hour from now
            created_at=time.time()
        )
        self.tasks.append(task)
        return task
    
    def run_cnp_cycle(self, task: Task):
        """Execute one complete CNP cycle for a task"""
        print(f"\n{'#'*60}")
        print(f"# Starting CNP Cycle for {task.task_id}")
        print(f"{'#'*60}")
        
        # Phase 1: Task Announcement & Bidding
        metrics = self.contractor.announce_task(task, list(self.agents.values()))
        
        # Phase 2: Bid Evaluation
        winning_bid = self.contractor.evaluate_bids(task.task_id)
        
        # Phase 3: Contract Award & Execution
        if winning_bid:
            self.contractor.award_contract(task.task_id, winning_bid, self.agents)
            self.all_metrics.append(metrics)
        else:
            print(f"\n[FAILED] No suitable agent found for {task.task_id}")
            metrics.task_completed = False
            self.all_metrics.append(metrics)
    
    def get_performance_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis"""
        data = []
        for metrics in self.all_metrics:
            task = next((t for t in self.tasks if t.task_id == metrics.task_id), None)
            
            data.append({
                'task_id': metrics.task_id,
                'task_type': task.task_type if task else 'unknown',
                'priority': task.priority if task else 0,
                'num_bids': metrics.num_bids_received,
                'bidding_duration': metrics.bidding_end_time - metrics.bidding_start_time,
                'avg_bid_generation_time': sum(metrics.bid_generation_times) / len(metrics.bid_generation_times) if metrics.bid_generation_times else 0,
                'evaluation_time': metrics.bid_evaluation_time or 0,
                'winning_cost': metrics.winning_bid_cost or 0,
                'average_cost': metrics.average_bid_cost or 0,
                'task_completed': metrics.task_completed,
                'actual_completion_time': metrics.actual_completion_time or 0,
                'total_cnp_time': (metrics.completion_time - metrics.announcement_time) if metrics.completion_time else 0,
                'assigned_agent': task.assigned_agent if task else None
            })
        
        return pd.DataFrame(data)
    
    def get_agent_performance_dataframe(self) -> pd.DataFrame:
        """Get performance metrics per agent"""
        data = []
        for agent_id, agent in self.agents.items():
            assigned_tasks = [t for t in self.tasks if t.assigned_agent == agent_id]
            data.append({
                'agent_id': agent_id,
                'agent_type': agent.agent_type,
                'skills': ', '.join(agent.skills),
                'tasks_completed': len(agent.completed_tasks),
                'efficiency_factor': agent.efficiency_factor,
                'avg_workload': sum(1 for t in assigned_tasks) / max(len(assigned_tasks), 1)
            })
        
        return pd.DataFrame(data)
