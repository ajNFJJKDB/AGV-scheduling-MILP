# AGV-scheduling-MILP

main file - updated8.py

## Project Description
This project aims to optimize the scheduling and management of Automated Guided Vehicles (AGVs) in a modern industrial setting. By implementing a Mixed-Integer Linear Programming (MILP) model, the system ensures efficient task allocation, workload balancing, and charging management for AGVs. The primary focus is to minimize idle times, reduce environmental impact, and maximize charge utilization while ensuring all tasks are completed within specified time windows.

## Key Features
- **Optimized Scheduling**: Uses MILP to ensure tasks are assigned efficiently, minimizing idle time and maximizing operational output.
- **Charge Management**: Ensures AGVs charge during idle periods to avoid running out of power during tasks.
- **Workload Balance**: Distributes tasks across AGVs evenly to prevent overuse or underutilization.
- **Environmental Impact**: The model emphasizes reducing carbon footprint by ensuring efficient energy usage.
- **Real-time Visualizations**: Outputs include interactive graphs to visualize AGV operations, task timelines, and charge levels.

## Objectives
- Develop an optimized AGV scheduling model to enhance operational efficiency.
- Maximize AGV fleet utilization while managing battery levels effectively.
- Achieve load balancing across the fleet by evenly distributing tasks.
- Minimize idle time while maximizing the charging time during breaks in task execution.

## Components
1. **AGV Scheduling Model**: 
    - Solves the AGV scheduling problem using MILP with the Gurobi optimizer.
    - Ensures tasks start and end within their respective time windows.
    - Allocates tasks to AGVs based on capability requirements.
  
2. **Charging Management**: 
    - Ensures AGVs charge during idle time, maximizing operational efficiency.
    - Prevents AGVs from running out of power during task execution.

3. **Visualization**: 
    - The project outputs multiple graphs using Plotly and NetworkX to visualize AGV operations, task distribution, and charge levels.

## Process Overview
1. **Data Collection**: 
    - Collect and format input data using Python’s Pandas library for optimal model processing.
2. **Model Development**: 
    - Build the MILP model using Gurobi to solve the AGV scheduling problem.
3. **Graph Generation**: 
    - Use Plotly and NetworkX to generate interactive charts, Gantt charts, and task distribution graphs.
4. **Result Evaluation**: 
    - Evaluate the model in various operational scenarios, ensuring optimal performance in each scenario.

## Requirements

### Software:
- **Programming Language**: Python
- **Libraries**: 
    - Gurobi Optimizer
    - Pandas
    - Plotly
    - NetworkX
- **IDE**: 
    - Jupyter Notebook, PyCharm, or Visual Studio Code
- **Version Control**: Git & GitHub

### Hardware:
- Standard computational resources to handle the optimization model and data visualization tasks.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/AGV-scheduling-optimization.git
    ```

2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

3. Run the model to solve the AGV scheduling problem:
    ```
    python updated8.py
    ```

4. Visualize the results using the generated output files:
    - View interactive Gantt charts and operational timelines for AGV tasks and charging levels.
    - Charge Percetage vs Time chart
    - AGV scehudle gantt chart (visualizing timeline)

## Usage
- **Input**: Provide the scheduling problem parameters, including task data, AGV capabilities, and charge constraints.
- **Output**: The model outputs optimized schedules, task assignments, and charging plans.
- **Graphs**: Interactive charts visualize task allocation, idle times, and charging behavior, ensuring a comprehensive understanding of the AGV fleet operations.

## Results
The MILP model successfully balanced task loads, managed AGV charging, and minimized idle time across various operational scenarios. The optimization framework significantly improved the operational efficiency of AGV fleets by scheduling tasks and charging events effectively, preventing battery depletion during tasks.

## Future Work
- Explore additional optimization algorithms, such as deep reinforcement learning, to further enhance scheduling efficiency.
- Integrate real-time dynamic scheduling to adapt to changing operational conditions.
- Expand the system to manage larger fleets and more complex task scenarios.

## Authors
- Balaji Gopal (3122213002015)
