"""
Streamlit Application for Contract Net Protocol Visualization
Interactive dashboard to explore CNP performance with adjustable parameters
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from cnp_system import (
    CNPSystem, ContracteeAgent, Task, TaskStatus
)
import random
import time

# Page configuration
st.set_page_config(
    page_title="Contract Net Protocol Simulator",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Contract Net Protocol - Real-World Integration")
st.markdown("""
This interactive simulator demonstrates how Contract Net Protocol is integrated in a **warehouse logistics system**.
Adjust parameters below to see how they affect task allocation, processing times, and overall system performance.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ System Parameters")

# Agent configuration
st.sidebar.subheader("Agent Configuration")
num_robots = st.sidebar.slider("Number of Picking Robots", 1, 10, 4)
num_forklifts = st.sidebar.slider("Number of Forklifts", 1, 8, 3)
num_drones = st.sidebar.slider("Number of Drones", 1, 6, 2)

# Task configuration
st.sidebar.subheader("Task Configuration")
num_tasks = st.sidebar.slider("Number of Tasks to Simulate", 5, 50, 15)
task_priority_high = st.sidebar.slider("% High Priority Tasks", 0, 100, 30)
task_complexity_range = st.sidebar.slider("Task Duration Range (seconds)", 1, 30, (5, 15))

# CNP parameters
st.sidebar.subheader("CNP Behavior")
cost_weight = st.sidebar.slider("Cost Weight in Evaluation", 0.0, 1.0, 0.4, 0.1)
time_weight = st.sidebar.slider("Time Weight in Evaluation", 0.0, 1.0, 0.3, 0.1)
confidence_weight = 1.0 - cost_weight - time_weight
st.sidebar.text(f"Confidence Weight: {confidence_weight:.1f}")

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

def create_system(num_robots, num_forklifts, num_drones):
    """Initialize the CNP system with agents"""
    system = CNPSystem()
    
    # Create robots (specialized in picking and packing)
    for i in range(num_robots):
        agent = ContracteeAgent(
            agent_id=f"robot_{i:02d}",
            agent_type="robot",
            skills=['pick', 'pack', 'sort'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=3
        )
        system.add_agent(agent)
    
    # Create forklifts (specialized in transport)
    for i in range(num_forklifts):
        agent = ContracteeAgent(
            agent_id=f"forklift_{i:02d}",
            agent_type="forklift",
            skills=['transport', 'lift', 'move'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=2
        )
        system.add_agent(agent)
    
    # Create drones (fast but limited capacity)
    for i in range(num_drones):
        agent = ContracteeAgent(
            agent_id=f"drone_{i:02d}",
            agent_type="drone",
            skills=['pick', 'transport', 'inspect'],
            location=(random.randint(0, 50), random.randint(0, 50)),
            max_capacity=1
        )
        system.add_agent(agent)
    
    return system

def generate_tasks(system, num_tasks, priority_high_pct, duration_range):
    """Generate random tasks"""
    task_types = ['pick', 'pack', 'transport', 'sort']
    skill_requirements = {
        'pick': ['pick'],
        'pack': ['pick', 'pack'],
        'transport': ['transport'],
        'sort': ['sort', 'pick']
    }
    
    tasks = []
    for i in range(num_tasks):
        task_type = random.choice(task_types)
        is_high_priority = random.random() < (priority_high_pct / 100)
        
        task = system.create_task(
            task_type=task_type,
            priority=random.randint(4, 5) if is_high_priority else random.randint(1, 3),
            location=(random.randint(0, 50), random.randint(0, 50)),
            skills=skill_requirements[task_type],
            duration=random.uniform(duration_range[0], duration_range[1])
        )
        tasks.append(task)
    
    return tasks

def plot_cnp_timeline(df):
    """Create a timeline visualization of CNP phases"""
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        # Announcement to bidding
        fig.add_trace(go.Scatter(
            x=[0, row['bidding_duration']],
            y=[idx, idx],
            mode='lines',
            line=dict(color='lightblue', width=10),
            name='Bidding Phase',
            showlegend=(idx == 0),
            hovertemplate=f"Task: {row['task_id']}<br>Bidding: {row['bidding_duration']:.3f}s<extra></extra>"
        ))
        
        # Evaluation
        eval_start = row['bidding_duration']
        eval_end = eval_start + row['evaluation_time']
        fig.add_trace(go.Scatter(
            x=[eval_start, eval_end],
            y=[idx, idx],
            mode='lines',
            line=dict(color='orange', width=10),
            name='Evaluation Phase',
            showlegend=(idx == 0),
            hovertemplate=f"Task: {row['task_id']}<br>Evaluation: {row['evaluation_time']:.3f}s<extra></extra>"
        ))
        
        # Execution
        if row['task_completed']:
            exec_start = eval_end
            exec_end = exec_start + row['actual_completion_time']
            fig.add_trace(go.Scatter(
                x=[exec_start, exec_end],
                y=[idx, idx],
                mode='lines',
                line=dict(color='green', width=10),
                name='Execution Phase',
                showlegend=(idx == 0),
                hovertemplate=f"Task: {row['task_id']}<br>Execution: {row['actual_completion_time']:.3f}s<extra></extra>"
            ))
    
    fig.update_layout(
        title="CNP Lifecycle Timeline (All Tasks)",
        xaxis_title="Time (seconds)",
        yaxis_title="Task Index",
        height=400,
        hovermode='closest'
    )
    
    return fig

def plot_processing_times(df):
    """Create detailed processing time breakdown"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Bid Generation Times',
            'Bid Evaluation Times', 
            'Total CNP Processing Time',
            'Task Execution Times'
        )
    )
    
    # Bid generation times
    fig.add_trace(
        go.Box(y=df['avg_bid_generation_time'], name='Bid Generation', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Evaluation times
    fig.add_trace(
        go.Box(y=df['evaluation_time'], name='Evaluation', marker_color='orange'),
        row=1, col=2
    )
    
    # Total CNP time
    fig.add_trace(
        go.Histogram(x=df['total_cnp_time'], name='Total CNP Time', marker_color='purple', nbinsx=20),
        row=2, col=1
    )
    
    # Execution times
    if df['task_completed'].sum() > 0:
        completed_df = df[df['task_completed']]
        fig.add_trace(
            go.Histogram(x=completed_df['actual_completion_time'], name='Execution', marker_color='green', nbinsx=20),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Processing Time Analysis")
    
    return fig

def plot_bid_analysis(df):
    """Analyze bidding patterns"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bids Received per Task', 'Cost Analysis')
    )
    
    # Number of bids
    fig.add_trace(
        go.Bar(x=df['task_id'], y=df['num_bids'], marker_color='steelblue', name='Bids Received'),
        row=1, col=1
    )
    
    # Cost comparison
    fig.add_trace(
        go.Scatter(x=df.index, y=df['average_cost'], mode='lines+markers', 
                   name='Average Bid Cost', marker_color='gray'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['winning_cost'], mode='lines+markers',
                   name='Winning Bid Cost', marker_color='gold'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Task ID", row=1, col=1)
    fig.update_xaxes(title_text="Task Index", row=1, col=2)
    fig.update_yaxes(title_text="Number of Bids", row=1, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=2)
    
    fig.update_layout(height=400, title_text="Bidding Behavior Analysis")
    
    return fig

def plot_agent_performance(agent_df):
    """Visualize agent-level performance"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Tasks Completed by Agent', 'Agent Efficiency Factors'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Tasks completed
    colors = ['lightcoral' if 'robot' in aid else 'lightgreen' if 'forklift' in aid else 'lightblue' 
              for aid in agent_df['agent_id']]
    
    fig.add_trace(
        go.Bar(x=agent_df['agent_id'], y=agent_df['tasks_completed'], 
               marker_color=colors, name='Tasks Completed'),
        row=1, col=1
    )
    
    # Efficiency factors
    fig.add_trace(
        go.Scatter(x=agent_df['agent_id'], y=agent_df['efficiency_factor'],
                   mode='markers', marker=dict(size=12, color=colors),
                   name='Efficiency Factor'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Agent ID", row=1, col=1)
    fig.update_xaxes(title_text="Agent ID", row=1, col=2)
    fig.update_yaxes(title_text="Tasks Completed", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency Factor", row=1, col=2)
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    fig.update_layout(height=400, showlegend=False, title_text="Agent Performance Metrics")
    
    return fig

def plot_task_allocation_map(system, tasks):
    """Visualize task and agent locations in the warehouse"""
    fig = go.Figure()
    
    # Plot agents
    for agent_id, agent in system.agents.items():
        color = 'red' if 'robot' in agent_id else 'blue' if 'forklift' in agent_id else 'green'
        symbol = 'circle' if 'robot' in agent_id else 'square' if 'forklift' in agent_id else 'diamond'
        
        fig.add_trace(go.Scatter(
            x=[agent.location[0]],
            y=[agent.location[1]],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol=symbol),
            text=[agent_id.split('_')[0]],
            textposition='top center',
            name=agent.agent_type.capitalize(),
            showlegend=True,
            hovertemplate=f"{agent_id}<br>Skills: {', '.join(agent.skills)}<br>Tasks: {len(agent.completed_tasks)}<extra></extra>"
        ))
    
    # Plot tasks
    task_colors = {1: 'lightgray', 2: 'lightgray', 3: 'yellow', 4: 'orange', 5: 'red'}
    for task in tasks:
        if task.status == TaskStatus.COMPLETED:
            fig.add_trace(go.Scatter(
                x=[task.location[0]],
                y=[task.location[1]],
                mode='markers',
                marker=dict(size=10, color=task_colors[task.priority], 
                           line=dict(width=2, color='green')),
                name=f'Priority {task.priority}',
                showlegend=False,
                hovertemplate=f"Task: {task.task_id}<br>Type: {task.task_type}<br>Priority: {task.priority}<br>Assigned: {task.assigned_agent}<extra></extra>"
            ))
    
    fig.update_layout(
        title="Warehouse Task Allocation Map",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        height=500,
        hovermode='closest'
    )
    
    return fig

def display_metrics_summary(df, agent_df):
    """Display key summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tasks", 
            len(df),
            help="Total number of tasks processed"
        )
        st.metric(
            "Completion Rate",
            f"{df['task_completed'].sum() / len(df) * 100:.1f}%",
            help="Percentage of tasks successfully completed"
        )
    
    with col2:
        avg_cnp_time = df['total_cnp_time'].mean()
        st.metric(
            "Avg CNP Time",
            f"{avg_cnp_time:.3f}s",
            help="Average time for complete CNP cycle"
        )
        st.metric(
            "Avg Evaluation Time",
            f"{df['evaluation_time'].mean():.3f}s",
            help="Average bid evaluation time"
        )
    
    with col3:
        st.metric(
            "Avg Bids/Task",
            f"{df['num_bids'].mean():.1f}",
            help="Average number of bids received per task"
        )
        st.metric(
            "Avg Winning Cost",
            f"{df['winning_cost'].mean():.2f}",
            help="Average cost of winning bids"
        )
    
    with col4:
        st.metric(
            "Total Agents",
            len(agent_df),
            help="Total number of agents in system"
        )
        st.metric(
            "Avg Tasks/Agent",
            f"{agent_df['tasks_completed'].mean():.1f}",
            help="Average tasks completed per agent"
        )

# Main simulation logic
if run_simulation:
    with st.spinner('🔄 Initializing CNP system and running simulation...'):
        # Create system
        system = create_system(num_robots, num_forklifts, num_drones)
        
        # Generate tasks
        tasks = generate_tasks(system, num_tasks, task_priority_high, task_complexity_range)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run CNP for each task
        for i, task in enumerate(tasks):
            status_text.text(f"Processing task {i+1}/{num_tasks}: {task.task_id}")
            system.run_cnp_cycle(task)
            progress_bar.progress((i + 1) / num_tasks)
        
        status_text.text("✅ Simulation complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Get performance data
        perf_df = system.get_performance_dataframe()
        agent_df = system.get_agent_performance_dataframe()
        
        # Store in session state
        st.session_state['perf_df'] = perf_df
        st.session_state['agent_df'] = agent_df
        st.session_state['system'] = system
        st.session_state['tasks'] = tasks

# Display results if available
if 'perf_df' in st.session_state:
    perf_df = st.session_state['perf_df']
    agent_df = st.session_state['agent_df']
    system = st.session_state['system']
    tasks = st.session_state['tasks']
    
    st.success("📊 Analysis Results")
    
    # Summary metrics
    st.subheader("📈 Key Performance Indicators")
    display_metrics_summary(perf_df, agent_df)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏱️ Processing Times", 
        "📊 Bidding Analysis", 
        "🤖 Agent Performance",
        "🗺️ Task Allocation Map",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.plotly_chart(plot_cnp_timeline(perf_df), use_container_width=True)
        st.plotly_chart(plot_processing_times(perf_df), use_container_width=True)
        
        # Detailed breakdown
        st.subheader("Timing Breakdown Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Bidding Phase**")
            st.write(f"- Mean: {perf_df['bidding_duration'].mean():.3f}s")
            st.write(f"- Std: {perf_df['bidding_duration'].std():.3f}s")
            st.write(f"- Min: {perf_df['bidding_duration'].min():.3f}s")
            st.write(f"- Max: {perf_df['bidding_duration'].max():.3f}s")
        
        with col2:
            st.write("**Evaluation Phase**")
            st.write(f"- Mean: {perf_df['evaluation_time'].mean():.3f}s")
            st.write(f"- Std: {perf_df['evaluation_time'].std():.3f}s")
            st.write(f"- Min: {perf_df['evaluation_time'].min():.3f}s")
            st.write(f"- Max: {perf_df['evaluation_time'].max():.3f}s")
    
    with tab2:
        st.plotly_chart(plot_bid_analysis(perf_df), use_container_width=True)
        
        # Bidding statistics
        st.subheader("Bidding Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bids", perf_df['num_bids'].sum())
        with col2:
            st.metric("Tasks with No Bids", (perf_df['num_bids'] == 0).sum())
        with col3:
            cost_savings = perf_df['average_cost'].sum() - perf_df['winning_cost'].sum()
            st.metric("Total Cost Savings", f"{cost_savings:.2f}")
    
    with tab3:
        st.plotly_chart(plot_agent_performance(agent_df), use_container_width=True)
        
        # Agent breakdown table
        st.subheader("Detailed Agent Breakdown")
        st.dataframe(
            agent_df.style.background_gradient(subset=['tasks_completed'], cmap='Greens')
                         .background_gradient(subset=['efficiency_factor'], cmap='Blues'),
            use_container_width=True
        )
    
    with tab4:
        st.plotly_chart(plot_task_allocation_map(system, tasks), use_container_width=True)
        
        # Task type breakdown
        st.subheader("Task Distribution")
        task_type_counts = perf_df['task_type'].value_counts()
        fig = px.pie(values=task_type_counts.values, names=task_type_counts.index, 
                     title="Task Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Task Performance Data")
        st.dataframe(perf_df, use_container_width=True)
        
        st.subheader("Agent Performance Data")
        st.dataframe(agent_df, use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv_perf = perf_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Task Data (CSV)",
                data=csv_perf,
                file_name="cnp_task_performance.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_agent = agent_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Agent Data (CSV)",
                data=csv_agent,
                file_name="cnp_agent_performance.csv",
                mime="text/csv"
            )

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click '🚀 Run Simulation' to start!")
    
    st.subheader("What This Simulation Shows")
    st.markdown("""
    This simulator demonstrates the **real-world integration** of Contract Net Protocol in a warehouse logistics system:
    
    ### 🎯 CNP Phases Tracked:
    1. **Task Announcement** - Manager broadcasts task to all capable agents
    2. **Bidding** - Agents evaluate tasks and submit bids (with processing time tracking)
    3. **Evaluation** - Manager evaluates bids using multi-criteria decision making (timed)
    4. **Award & Execution** - Winner executes task (actual completion time measured)
    
    ### 📊 Metrics Collected:
    - **Bid generation time** per agent (decision-making speed)
    - **Bid evaluation time** (manager's processing time)
    - **Number of bids** received (system responsiveness)
    - **Cost analysis** (winning vs. average bid costs)
    - **Task completion times** (actual vs. estimated)
    - **Agent workload distribution**
    
    ### 🔧 What You Can Explore:
    - How agent quantity affects bid competition and processing time
    - Impact of task priority on allocation decisions
    - Trade-offs between cost, time, and confidence in bid evaluation
    - Agent specialization effects on task distribution
    - System scalability and bottlenecks
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust the **number and types of agents** (robots, forklifts, drones)
    2. Configure **task parameters** (quantity, priority distribution, complexity)
    3. Tune **CNP evaluation weights** (cost vs. time vs. confidence)
    4. Click **Run Simulation** and observe the results
    5. Explore different tabs to analyze timing, bidding patterns, and agent performance
    """)
