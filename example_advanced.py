"""
Advanced Example: Custom Evaluation Criteria and CNP Extensions

This example demonstrates:
1. Custom bid evaluation strategies
2. Dynamic weight adjustment based on system state
3. Agent learning from past performance
4. Task bundling for efficiency
"""

from cnp_system import (
    CNPSystem, ContracteeAgent, ContractorAgent, Task, Bid, PerformanceMetrics
)
import random
from typing import List, Dict

class AdaptiveContractorAgent(ContractorAgent):
    """
    Enhanced contractor that adapts evaluation weights based on:
    - System load
    - Historical performance
    - Time of day (simulated)
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.agent_history: Dict[str, List[float]] = {}  # Track agent performance
        self.system_load = 0.0
        
    def adaptive_evaluate_bids(self, task_id: str) -> Bid:
        """
        Enhanced evaluation with adaptive weights
        """
        print(f"\n[ADAPTIVE EVALUATION] Evaluating bids with context-aware weights...")
        
        bids = self.pending_bids[task_id]
        if not bids:
            return None
        
        task = self.active_tasks[task_id]
        
        # Calculate system load (0-1)
        total_capacity = sum(3 for _ in self.agent_history.keys())  # Simplified
        active_tasks = len([t for t in self.active_tasks.values() 
                          if t.status.value in ['awarded', 'in_progress']])
        self.system_load = active_tasks / max(total_capacity, 1)
        
        # Adapt weights based on context
        if task.priority >= 4:
            # High priority: favor reliability and speed
            cost_weight = 0.1
            time_weight = 0.4
            confidence_weight = 0.4
            history_weight = 0.1
        elif self.system_load > 0.7:
            # System under load: favor agents with capacity
            cost_weight = 0.2
            time_weight = 0.5
            confidence_weight = 0.2
            history_weight = 0.1
        else:
            # Normal conditions: balanced
            cost_weight = 0.3
            time_weight = 0.3
            confidence_weight = 0.3
            history_weight = 0.1
        
        print(f"  System load: {self.system_load:.2f}")
        print(f"  Weights: cost={cost_weight}, time={time_weight}, "
              f"confidence={confidence_weight}, history={history_weight}")
        
        # Score bids
        scored_bids = []
        for bid in bids:
            # Normalize criteria
            cost_score = 1 - (bid.cost / (max(b.cost for b in bids) + 0.001))
            time_score = 1 - (bid.estimated_time / (max(b.estimated_time for b in bids) + 0.001))
            confidence_score = bid.confidence
            
            # Historical performance score
            if bid.agent_id in self.agent_history:
                history = self.agent_history[bid.agent_id]
                # Average of past performance (1.0 = met expectations)
                history_score = sum(history) / len(history) if history else 0.5
            else:
                history_score = 0.5  # Neutral for new agents
            
            # Weighted total
            total_score = (cost_weight * cost_score + 
                          time_weight * time_score + 
                          confidence_weight * confidence_score +
                          history_weight * history_score)
            
            scored_bids.append((total_score, bid))
            print(f"  {bid.agent_id}: total={total_score:.3f} "
                  f"(cost={cost_score:.2f}, time={time_score:.2f}, "
                  f"conf={confidence_score:.2f}, hist={history_score:.2f})")
        
        # Select winner
        scored_bids.sort(reverse=True, key=lambda x: x[0])
        return scored_bids[0][1]
    
    def record_performance(self, task_id: str, agent_id: str, 
                          estimated_time: float, actual_time: float):
        """
        Record agent performance for learning
        """
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []
        
        # Performance ratio: 1.0 = perfect, <1.0 = better than expected
        performance = estimated_time / max(actual_time, 0.001)
        self.agent_history[agent_id].append(performance)
        
        # Keep only recent history (last 10 tasks)
        if len(self.agent_history[agent_id]) > 10:
            self.agent_history[agent_id] = self.agent_history[agent_id][-10:]
        
        print(f"[LEARNING] {agent_id} performance: {performance:.2f} "
              f"(avg: {sum(self.agent_history[agent_id])/len(self.agent_history[agent_id]):.2f})")

class SmartAgent(ContracteeAgent):
    """
    Enhanced agent with learning capability
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_history = []
        self.reliability = 1.0  # Increases with successful completions
    
    def evaluate_and_bid(self, task: Task) -> Bid:
        """
        Enhanced bidding with reliability factor
        """
        bid = super().evaluate_and_bid(task)
        
        if bid:
            # Adjust confidence based on past performance
            bid.confidence *= self.reliability
            
            # Adjust cost based on reliability (reliable agents can charge more)
            bid.cost *= (0.8 + 0.4 * self.reliability)
        
        return bid
    
    def complete_task(self, task: Task, success: bool):
        """
        Update reliability based on task outcome
        """
        self.success_history.append(1.0 if success else 0.0)
        
        # Keep last 20 tasks
        if len(self.success_history) > 20:
            self.success_history = self.success_history[-20:]
        
        # Update reliability (exponential moving average)
        recent_success_rate = sum(self.success_history) / len(self.success_history)
        self.reliability = 0.7 * self.reliability + 0.3 * recent_success_rate
        self.reliability = max(0.5, min(1.0, self.reliability))  # Clamp to [0.5, 1.0]

def demonstrate_task_bundling():
    """
    Example: Bundle nearby tasks for efficiency
    """
    print("\n" + "="*70)
    print("EXAMPLE: Task Bundling for Spatial Efficiency")
    print("="*70)
    
    system = CNPSystem()
    
    # Create agents
    for i in range(3):
        agent = ContracteeAgent(
            agent_id=f"robot_{i:02d}",
            agent_type="robot",
            skills=['pick', 'pack'],
            location=(random.randint(0, 100), random.randint(0, 100)),
            max_capacity=5
        )
        system.add_agent(agent)
    
    # Create clustered tasks
    print("\nCreating task clusters...")
    
    # Cluster 1: Bottom-left
    cluster1_center = (20, 20)
    for i in range(3):
        task = system.create_task(
            task_type='pick',
            priority=3,
            location=(cluster1_center[0] + random.randint(-5, 5),
                     cluster1_center[1] + random.randint(-5, 5)),
            skills=['pick'],
            duration=5.0
        )
        print(f"  Cluster 1 - {task.task_id} at {task.location}")
    
    # Cluster 2: Top-right
    cluster2_center = (80, 80)
    for i in range(3):
        task = system.create_task(
            task_type='pick',
            priority=3,
            location=(cluster2_center[0] + random.randint(-5, 5),
                     cluster2_center[1] + random.randint(-5, 5)),
            skills=['pick'],
            duration=5.0
        )
        print(f"  Cluster 2 - {task.task_id} at {task.location}")
    
    # Identify clusters (simple distance-based)
    def distance(loc1, loc2):
        return ((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)**0.5
    
    clusters = []
    for task in system.tasks:
        added = False
        for cluster in clusters:
            if any(distance(task.location, t.location) < 15 for t in cluster):
                cluster.append(task)
                added = True
                break
        if not added:
            clusters.append([task])
    
    print(f"\nIdentified {len(clusters)} task clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} tasks")
    
    # Award entire clusters to agents for efficiency
    print("\nAwarding clusters to agents...")
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        
        # Announce first task of cluster
        system.run_cnp_cycle(cluster[0])
        
        # Assign remaining tasks to same agent
        winner = cluster[0].assigned_agent
        if winner:
            print(f"\n  Bundling remaining tasks in cluster {i+1} to {winner}")
            for task in cluster[1:]:
                task.assigned_agent = winner
                task.status = task.status.AWARDED
                system.agents[winner].execute_task(task)
                print(f"    {task.task_id} → {winner}")

def demonstrate_adaptive_evaluation():
    """
    Example: Adaptive evaluation with learning
    """
    print("\n" + "="*70)
    print("EXAMPLE: Adaptive Evaluation with Agent Learning")
    print("="*70)
    
    # Create system with adaptive contractor
    contractor = AdaptiveContractorAgent("adaptive_manager")
    
    # Create smart agents
    agents = []
    for i in range(4):
        agent = SmartAgent(
            agent_id=f"smart_robot_{i:02d}",
            agent_type="robot",
            skills=['pick', 'pack', 'sort'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=3
        )
        agents.append(agent)
    
    # Simulate some agents being more reliable than others
    agents[0].reliability = 0.95  # Very reliable
    agents[1].reliability = 0.85  # Good
    agents[2].reliability = 0.75  # Average
    agents[3].reliability = 0.65  # Below average
    
    print("\nInitial agent reliabilities:")
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.reliability:.2f}")
    
    # Create and run tasks
    print("\nRunning tasks with adaptive evaluation...")
    for i in range(6):
        task = Task(
            task_id=f"task_{i:03d}",
            task_type='pick',
            priority=random.randint(3, 5),
            location=(random.randint(0, 50), random.randint(0, 50)),
            required_skills=['pick'],
            estimated_duration=random.uniform(5, 10),
            deadline=999999,
            created_at=0
        )
        
        print(f"\n--- Task {task.task_id} (priority {task.priority}) ---")
        
        # Collect bids
        contractor.active_tasks[task.task_id] = task
        contractor.pending_bids[task.task_id] = []
        
        for agent in agents:
            bid = agent.evaluate_and_bid(task)
            if bid:
                contractor.pending_bids[task.task_id].append(bid)
                print(f"  Bid from {agent.agent_id}: cost={bid.cost:.2f}, "
                      f"conf={bid.confidence:.2f}")
        
        # Adaptive evaluation
        winning_bid = contractor.adaptive_evaluate_bids(task.task_id)
        
        if winning_bid:
            print(f"\n  Winner: {winning_bid.agent_id}")
            
            # Simulate execution
            winner = next(a for a in agents if a.agent_id == winning_bid.agent_id)
            actual_time = winner.execute_task(task)
            
            # Record performance
            contractor.record_performance(
                task.task_id, 
                winning_bid.agent_id,
                winning_bid.estimated_time,
                actual_time
            )
            
            # Update agent learning
            success = actual_time <= winning_bid.estimated_time * 1.2
            winner.complete_task(task, success)

def main():
    print("="*70)
    print("Advanced CNP Examples: Custom Evaluation & Extensions")
    print("="*70)
    
    # Example 1: Task bundling
    demonstrate_task_bundling()
    
    # Example 2: Adaptive evaluation
    demonstrate_adaptive_evaluation()
    
    print("\n" + "="*70)
    print("Examples complete! Key takeaways:")
    print("  1. Task bundling reduces travel time and increases efficiency")
    print("  2. Adaptive evaluation weights improve system performance")
    print("  3. Agent learning creates trust-based allocation over time")
    print("  4. Context-aware decisions optimize for current system state")
    print("="*70)

if __name__ == "__main__":
    main()
