import pandas as pd
from gurobipy import Model, GRB, quicksum

# Load the data
routes_df = pd.read_excel('C:/Users/balaj/Downloads/beifen_Ma0510/Table 1 - All possible routes.xlsx')
service_requests_df = pd.read_csv('C:/Users/balaj/Downloads/beifen_Ma0510/Table 2 -Service request time chart.csv')
agv_properties_df = pd.read_excel('C:/Users/balaj/Downloads/beifen_Ma0510/Table 3 - AGV properties.xlsx')

# Clean column names
routes_df.columns = routes_df.iloc[0]
routes_df = routes_df.drop(0).reset_index(drop=True)
routes_df.columns = routes_df.columns.str.strip()  # Remove any leading/trailing spaces

# Convert necessary columns to appropriate data types
routes_df['Total time'] = pd.to_numeric(routes_df['Total time'])
service_requests_df['min Proc. time'] = pd.to_numeric(service_requests_df['min Proc. time'])
service_requests_df['Earliest pick time'] = pd.to_numeric(service_requests_df['Earliest pick time'])
service_requests_df['Latest delivery time'] = pd.to_numeric(service_requests_df['Latest delivery time'])
service_requests_df['min Energy Requirement'] = pd.to_numeric(service_requests_df['min Energy Requirement'])
agv_properties_df['Initial charge'] = pd.to_numeric(agv_properties_df['Initial charge'])

# Charging and discharging rates
charging_rate = 1 / 10  # 1% per 10 seconds
discharging_rate = 0.5 / 10  # 0.5% per 10 seconds

# Extract relevant data
num_agvs = len(agv_properties_df)
num_requests = len(service_requests_df)
time_horizon = 8000  # Assumed total time horizon (can be adjusted)

# Initialize the model
m = Model("AGV_Scheduling")

# Define variables
# x[i, j] = 1 if AGV i is assigned to service request j
x = m.addVars(num_agvs, num_requests, vtype=GRB.BINARY, name="x")

# y[i, t] = 1 if AGV i is at charging station at time t
y = m.addVars(num_agvs, time_horizon, vtype=GRB.BINARY, name="y")

# z[j] = start time of service request j
z = m.addVars(num_requests, vtype=GRB.CONTINUOUS, name="z")

# Penalty for heavy AGVs doing normal tasks
penalty = m.addVars(num_agvs, num_requests, vtype=GRB.BINARY, name="penalty")

# Load balancing variables
task_count = m.addVars(num_agvs, vtype=GRB.INTEGER, name="task_count")
avg_task_count = service_requests_df.shape[0] // agv_properties_df[agv_properties_df['Capability'] == "Normal"].shape[0]

# Define objective function
# Minimize total run time and balance task load, with penalties for improper AGV usage
m.setObjective(
    quicksum(z[j] for j in range(num_requests)) +
    quicksum(y[i, t] for i in range(num_agvs) for t in range(time_horizon)) +
    quicksum(penalty[i, j] for i in range(num_agvs) for j in range(num_requests)) +
    quicksum((task_count[i] - avg_task_count) * (task_count[i] - avg_task_count) for i in range(num_agvs)),
    GRB.MINIMIZE
)

# Add constraints
# 1. Each service request is assigned to one AGV
for j in range(num_requests):
    m.addConstr(quicksum(x[i, j] for i in range(num_agvs)) == 1, name=f"assign_{j}")

# 2. Capability requirement and penalty assignment
for i in range(num_agvs):
    for j in range(num_requests):
        if service_requests_df.loc[j, "Capability Requirement"] == "Heavy" and agv_properties_df.loc[i, "Capability"] == "Normal":
            m.addConstr(x[i, j] == 0, name=f"capability_{i}_{j}")
        if service_requests_df.loc[j, "Capability Requirement"] == "Normal" and agv_properties_df.loc[i, "Capability"] == "Heavy":
            m.addConstr(penalty[i, j] == x[i, j], name=f"penalty_{i}_{j}")

# 3. Charging constraints (simplified, need to refine based on energy requirements)
for i in range(num_agvs):
    for j in range(num_requests):
        m.addConstr(z[j] >= quicksum(y[i, t] * 10 for t in range(time_horizon)), name=f"charging_{i}_{j}")

# 4. Time window constraints
for j in range(num_requests):
    m.addConstr(z[j] >= service_requests_df.loc[j, "Earliest pick time"], name=f"earliest_{j}")
    m.addConstr(z[j] + service_requests_df.loc[j, "min Proc. time"] <= service_requests_df.loc[j, "Latest delivery time"], name=f"latest_{j}")

# 5. Task count for load balancing
for i in range(num_agvs):
    m.addConstr(task_count[i] == quicksum(x[i, j] for j in range(num_requests)), name=f"task_count_{i}")

# Optimize the model
m.optimize()

# Extract and display the results
results = []
for i in range(num_agvs):
    agv_schedule = []
    current_time = 0
    current_charge = agv_properties_df.loc[i, "Initial charge"]
    for j in range(num_requests):
        if x[i, j].X > 0.5:
            task_start = z[j].X
            task_end = task_start + service_requests_df.loc[j, "min Proc. time"]
            energy_needed = service_requests_df.loc[j, "min Energy Requirement"]
            
            # Calculate the energy needed for the backhauling trip
            backhaul_time = routes_df.loc[
                (routes_df['Origin'] == service_requests_df.loc[j, 'Destination']) & 
                (routes_df['Destination'] == 'C1'), 'Total time'].values[0]
            backhaul_energy_needed = backhaul_time * discharging_rate * 10  # Convert time to energy
            
            total_energy_needed = energy_needed + backhaul_energy_needed
            
            # Check if AGV needs charging before starting the task
            if current_charge < total_energy_needed:
                charging_needed = total_energy_needed - current_charge
                charging_duration = (charging_needed / charging_rate) * 10  # Convert energy to time
                agv_schedule.append(f"Charging ({current_charge:.1f}% to {current_charge + charging_needed:.1f}%) from {current_time} to {current_time + charging_duration} (Duration: {charging_duration})")
                current_time += charging_duration
                current_charge += charging_needed  # Update the current charge
                task_start = current_time  # Update task start time after charging
            
            agv_schedule.append(f"Task {j} (Job ID: {service_requests_df.loc[j, 'Job ID']}) starts at {task_start} and ends at {task_end}")
            backhaul_start = task_end
            backhaul_end = backhaul_start + backhaul_time
            agv_schedule.append(f"Backhauling from {service_requests_df.loc[j, 'Destination']} to C1 starts at {backhaul_start} to ends at {backhaul_end}")
            current_time = backhaul_end
            current_charge -= energy_needed  # Update the current charge after completing the task
            
            # Charging after backhauling if needed
            if current_charge < 100:
                charging_needed = 100 - current_charge
                charging_duration = (charging_needed / charging_rate) * 10  # Convert energy to time
                agv_schedule.append(f"Charging ({current_charge:.1f}% to 100.0%) from {current_time} to {current_time + charging_duration} (Duration: {charging_duration})")
                current_time += charging_duration
                current_charge = 100  # Fully charge after backhauling
            
    results.append((i, agv_schedule))

# Print the detailed schedule for each AGV
for agv_id, schedule in results:
    print(f"AGV {agv_id}:")
    for event in schedule:
        print(f"  {event}")
