"""
Standalone Example: Using the CNP System Programmatically

This script demonstrates how to use the Contract Net Protocol system
directly in your own Python code without the Streamlit interface.
"""

from cnp_system import CNPSystem, ContracteeAgent
import random

def main():
    print("="*70)
    print("Contract Net Protocol - Programmatic Example")
    print("="*70)
    print()
    
    # Initialize the CNP system
    print("📦 Initializing CNP system...")
    system = CNPSystem()
    
    # Create a fleet of agents with different capabilities
    print("🤖 Creating agent fleet...")
    
    # Picking robots - specialized in picking and packing
    for i in range(3):
        agent = ContracteeAgent(
            agent_id=f"robot_{i:02d}",
            agent_type="robot",
            skills=['pick', 'pack', 'sort'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=3
        )
        system.add_agent(agent)
        print(f"  Added {agent.agent_id} at location {agent.location}")
    
    # Forklifts - specialized in transport
    for i in range(2):
        agent = ContracteeAgent(
            agent_id=f"forklift_{i:02d}",
            agent_type="forklift",
            skills=['transport', 'lift', 'move'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=2
        )
        system.add_agent(agent)
        print(f"  Added {agent.agent_id} at location {agent.location}")
    
    # Drones - fast but limited capacity
    agent = ContracteeAgent(
        agent_id="drone_01",
        agent_type="drone",
        skills=['pick', 'transport', 'inspect'],
        location=(25, 25),  # Central location
        max_capacity=1
    )
    system.add_agent(agent)
    print(f"  Added {agent.agent_id} at location {agent.location}")
    
    print(f"\n✅ Created {len(system.agents)} agents")
    print()
    
    # Create tasks
    print("📋 Creating tasks...")
    
    # Task 1: High priority picking task
    task1 = system.create_task(
        task_type='pick',
        priority=5,
        location=(10, 15),
        skills=['pick'],
        duration=8.0
    )
    print(f"  Created {task1.task_id}: {task1.task_type} (priority {task1.priority})")
    
    # Task 2: Normal priority transport task
    task2 = system.create_task(
        task_type='transport',
        priority=3,
        location=(45, 30),
        skills=['transport'],
        duration=12.0
    )
    print(f"  Created {task2.task_id}: {task2.task_type} (priority {task2.priority})")
    
    # Task 3: Packing task requiring multiple skills
    task3 = system.create_task(
        task_type='pack',
        priority=4,
        location=(20, 40),
        skills=['pick', 'pack'],
        duration=10.0
    )
    print(f"  Created {task3.task_id}: {task3.task_type} (priority {task3.priority})")
    
    print(f"\n✅ Created {len(system.tasks)} tasks")
    print()
    
    # Execute CNP cycles
    print("🔄 Executing Contract Net Protocol cycles...")
    print()
    
    for task in system.tasks:
        system.run_cnp_cycle(task)
        print()
    
    # Analyze results
    print("="*70)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*70)
    print()
    
    perf_df = system.get_performance_dataframe()
    agent_df = system.get_agent_performance_dataframe()
    
    # Summary statistics
    print("Task Completion Summary:")
    print(f"  Total tasks: {len(perf_df)}")
    print(f"  Completed: {perf_df['task_completed'].sum()}")
    print(f"  Failed: {len(perf_df) - perf_df['task_completed'].sum()}")
    print(f"  Success rate: {perf_df['task_completed'].sum() / len(perf_df) * 100:.1f}%")
    print()
    
    print("Timing Statistics:")
    print(f"  Avg bidding duration: {perf_df['bidding_duration'].mean():.3f}s")
    print(f"  Avg evaluation time: {perf_df['evaluation_time'].mean():.3f}s")
    print(f"  Avg total CNP time: {perf_df['total_cnp_time'].mean():.3f}s")
    print()
    
    print("Bidding Statistics:")
    print(f"  Avg bids per task: {perf_df['num_bids'].mean():.1f}")
    print(f"  Avg winning cost: {perf_df['winning_cost'].mean():.2f}")
    print(f"  Avg bid cost: {perf_df['average_cost'].mean():.2f}")
    cost_savings = perf_df['average_cost'].sum() - perf_df['winning_cost'].sum()
    print(f"  Total cost savings: {cost_savings:.2f}")
    print()
    
    print("Agent Performance:")
    for _, agent_row in agent_df.iterrows():
        print(f"  {agent_row['agent_id']}: {agent_row['tasks_completed']} tasks, "
              f"efficiency={agent_row['efficiency_factor']:.2f}")
    print()
    
    # Detailed task breakdown
    print("="*70)
    print("📋 DETAILED TASK BREAKDOWN")
    print("="*70)
    print()
    
    for _, row in perf_df.iterrows():
        print(f"Task: {row['task_id']}")
        print(f"  Type: {row['task_type']}, Priority: {row['priority']}")
        print(f"  Bids received: {row['num_bids']}")
        print(f"  Bidding time: {row['bidding_duration']:.3f}s")
        print(f"  Evaluation time: {row['evaluation_time']:.3f}s")
        if row['task_completed']:
            print(f"  ✅ Completed by {row['assigned_agent']} in {row['actual_completion_time']:.2f}s")
            print(f"  Winning cost: {row['winning_cost']:.2f}")
        else:
            print(f"  ❌ Failed to allocate")
        print()
    
    # Save results to CSV
    print("💾 Saving results to CSV files...")
    perf_df.to_csv('cnp_task_results.csv', index=False)
    agent_df.to_csv('cnp_agent_results.csv', index=False)
    print("  ✅ Saved to cnp_task_results.csv and cnp_agent_results.csv")
    print()
    
    print("="*70)
    print("✨ Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
