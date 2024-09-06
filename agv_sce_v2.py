import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# File paths
routes_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 1 - All possible routes.xlsx'
service_requests_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 2 -Service request time chart.csv'
agv_properties_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 3 - AGV properties.xlsx'

# Load data
routes_df = pd.read_excel(routes_file_path)
service_requests_df = pd.read_csv(service_requests_file_path)
agv_properties_df = pd.read_excel(agv_properties_file_path)

# Define AGV capabilities
agv_capabilities = {row['AGV_ID']: row['Capability'] for index, row in agv_properties_df.iterrows()}
initial_charge = {row['AGV_ID']: row['Initial charge'] for index, row in agv_properties_df.iterrows()}

# Extract travel times to charging station
travel_times_to_charging = {
    'L1': routes_df.loc[routes_df['All possible routes'] == 'R28', 'Unnamed: 5'].values[0],
    'L2': routes_df.loc[routes_df['All possible routes'] == 'R29', 'Unnamed: 5'].values[0],
    'U1': routes_df.loc[routes_df['All possible routes'] == 'R21', 'Unnamed: 5'].values[0],
    'U2': routes_df.loc[routes_df['All possible routes'] == 'R22', 'Unnamed: 5'].values[0],
    'U3': routes_df.loc[routes_df['All possible routes'] == 'R23', 'Unnamed: 5'].values[0],
    'U4': routes_df.loc[routes_df['All possible routes'] == 'R24', 'Unnamed: 5'].values[0],
    'U5': routes_df.loc[routes_df['All possible routes'] == 'R25', 'Unnamed: 5'].values[0],
}

# Parameters
n_tasks = len(service_requests_df)
n_agvs = len(agv_capabilities)
time_horizon = max(service_requests_df['Latest delivery time'])  # Total time available
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# File paths
routes_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 1 - All possible routes.xlsx'
service_requests_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 2 -Service request time chart.csv'
agv_properties_file_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/Table 3 - AGV properties.xlsx'

# Load data
routes_df = pd.read_excel(routes_file_path)
service_requests_df = pd.read_csv(service_requests_file_path)
agv_properties_df = pd.read_excel(agv_properties_file_path)

# Define AGV capabilities
agv_capabilities = {row['AGV_ID']: row['Capability'] for index, row in agv_properties_df.iterrows()}
initial_charge = {row['AGV_ID']: max(row['Initial charge'], 70) for index, row in agv_properties_df.iterrows()}  # Ensure minimum 70% charge

# Extract travel times to charging station
travel_times_to_charging = {
    'L1': routes_df.loc[routes_df['All possible routes'] == 'R28', 'Unnamed: 5'].values[0],
    'L2': routes_df.loc[routes_df['All possible routes'] == 'R29', 'Unnamed: 5'].values[0],
    'U1': routes_df.loc[routes_df['All possible routes'] == 'R21', 'Unnamed: 5'].values[0],
    'U2': routes_df.loc[routes_df['All possible routes'] == 'R22', 'Unnamed: 5'].values[0],
    'U3': routes_df.loc[routes_df['All possible routes'] == 'R23', 'Unnamed: 5'].values[0],
    'U4': routes_df.loc[routes_df['All possible routes'] == 'R24', 'Unnamed: 5'].values[0],
    'U5': routes_df.loc[routes_df['All possible routes'] == 'R25', 'Unnamed: 5'].values[0],
}

# Parameters
n_tasks = len(service_requests_df)
n_agvs = len(agv_capabilities)
time_horizon = max(service_requests_df['Latest delivery time'])  # Total time available

charging_rate = 1  # 1% per 10 seconds
discharging_rate = 0.5  # 0.5% per 10 seconds
charging_station_capacity = 1  # Only one charging station

# Create a new model
model = gp.Model("AGV_Scheduling")

# Create variables
x = model.addVars(n_tasks, n_agvs, vtype=GRB.BINARY, name="x")  # Task assignment
start_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="start_time")  # Start times
end_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="end_time")  # End times
agv_load = model.addVars(n_agvs, vtype=GRB.CONTINUOUS, name="agv_load")  # Load of each AGV
charge_level = model.addVars(n_tasks + 1, n_agvs, vtype=GRB.CONTINUOUS, name="charge_level")  # Charge level of each AGV at each task
charging_time = model.addVars(n_tasks, n_agvs, vtype=GRB.CONTINUOUS, name="charging_time")  # Charging time for each AGV
charging_station_usage = model.addVars(n_tasks, vtype=GRB.BINARY, name="charging_station_usage")  # Charging station usage

# Objective: Minimize the makespan
model.setObjective(gp.quicksum(end_time[i] for i in range(n_tasks)), GRB.MINIMIZE)

# Constraints
for i in range(n_tasks):
    # Each task must be assigned to exactly one AGV
    model.addConstr(gp.quicksum(x[i, j] for j in range(n_agvs)) == 1, f"Task_{i}_assignment")

    # Task must start within its time window
    model.addConstr(start_time[i] >= service_requests_df.loc[i, 'Earliest pick time'], f"Task_{i}_earliest_start")
    model.addConstr(end_time[i] == start_time[i] + service_requests_df.loc[i, 'min Proc. time'], f"Task_{i}_end_time")
    model.addConstr(end_time[i] <= service_requests_df.loc[i, 'Latest delivery time'], f"Task_{i}_latest_end")

    # Capability requirement constraint
    task_capability = service_requests_df.loc[i, 'Capability Requirement']
    for j in range(n_agvs):
        agv_capability = agv_capabilities[j]
        if task_capability != agv_capability and agv_capability != 'Heavy':
            model.addConstr(x[i, j] == 0, f"Task_{i}AGV{j}_capability")

# Charge constraints
for j in range(n_agvs):
    # Initial charge level
    model.addConstr(charge_level[0, j] == initial_charge[j], f"Initial_charge_{j}")

    for i in range(n_tasks):
        required_charge = service_requests_df.loc[i, 'min Proc. time'] * discharging_rate
        destination = service_requests_df.loc[i, 'Destination'].strip()
        return_charge = travel_times_to_charging[destination] * discharging_rate

        if i == 0:
            idle_time = start_time[i]
        else:
            idle_time = start_time[i] - end_time[i - 1]

        max_possible_charge = (100 - charge_level[i, j]) / charging_rate
        model.addConstr(charging_time[i, j] <= idle_time, f"Charging_time_idle_{i}_{j}")
        model.addConstr(charging_time[i, j] <= max_possible_charge, f"Charging_time_max_{i}_{j}")

        model.addConstr(
            charge_level[i + 1, j] == charge_level[i, j] - required_charge - return_charge + charging_time[i, j] * charging_rate,
            f"Charge_{i}_{j}"
        )

        # Ensure charge level is non-negative and not below 20%
        model.addConstr(charge_level[i + 1, j] >= 20, f"Non_negative_charge_{i}_{j}")

    # Include charging station task
    model.addConstr(charge_level[n_tasks, j] <= 100, f"Max_charge_{n_tasks}_{j}")

# Charging station capacity constraint
model.addConstr(gp.quicksum(charging_station_usage[i] for i in range(n_tasks)) <= charging_station_capacity, f"Charging_station_capacity")

# Sequential task constraints for each AGV
for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] - (1 - x[i, j]) * time_horizon - (1 - x[k, j]) * time_horizon, f"AGV_{j}task{i}sequential{k}")

# Load constraints: AGV load is the sum of the end times of its tasks
for j in range(n_agvs):
    model.addConstr(agv_load[j] == gp.quicksum(end_time[i] * x[i, j] for i in range(n_tasks)), f"AGV_{j}_load")

# Optimize the model
model.optimize()

# Diagnose infeasibility
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    model.computeIIS()
    model.write("model.ilp")
    with open("model.ilp", "r") as f:
        for line in f:
            print(line.strip())

# Ensure agv_tasks and charge_over_time are defined even if infeasible
agv_tasks = {j: [] for j in range(n_agvs)}
charge_over_time = {j: [] for j in range(n_agvs)}

# Extract the solution if feasible
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('x', x)
    start_times_solution = model.getAttr('x', start_time)
    end_times_solution = model.getAttr('x', end_time)
    charge_levels_solution = model.getAttr('x', charge_level)
    tasks = []
    current_charge_level = initial_charge.copy()
    for j in range(n_agvs):
        previous_end_time = 0
        charge_over_time[j].append((0, current_charge_level[j]))  # Initial charge level
        for i in range(n_tasks):
            if solution[i, j] > 0.5:
                start = start_times_solution[i]
                end = end_times_solution[i]
                job_id = service_requests_df.loc[i, 'Job ID']

                # If there is idle time

charging_rate = 2  # 1% per 10 seconds
discharging_rate = 0.2  # 0.1% per second
charging_station_capacity = 1  # Only one charging station

# Create a new model
model = gp.Model("AGV_Scheduling")

# Create variables
x = model.addVars(n_tasks, n_agvs, vtype=GRB.BINARY, name="x")  # Task assignment
start_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="start_time")  # Start times
end_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="end_time")  # End times
agv_load = model.addVars(n_agvs, vtype=GRB.CONTINUOUS, name="agv_load")  # Load of each AGV
charge_level = model.addVars(n_tasks + 1, n_agvs, vtype=GRB.CONTINUOUS, name="charge_level")  # Charge level of each AGV at each task
charging_time = model.addVars(n_tasks, n_agvs, vtype=GRB.CONTINUOUS, name="charging_time")  # Charging time for each AGV
charging_station_usage = model.addVars(n_tasks, vtype=GRB.BINARY, name="charging_station_usage")  # Charging station usage

# Objective: Minimize the makespan
model.setObjective(gp.quicksum(end_time[i] for i in range(n_tasks)), GRB.MINIMIZE)

# Constraints
for i in range(n_tasks):
    # Each task must be assigned to exactly one AGV
    model.addConstr(gp.quicksum(x[i, j] for j in range(n_agvs)) == 1, f"Task_{i}_assignment")

    # Task must start within its time window
    model.addConstr(start_time[i] >= service_requests_df.loc[i, 'Earliest pick time'], f"Task_{i}_earliest_start")
    model.addConstr(end_time[i] == start_time[i] + service_requests_df.loc[i, 'min Proc. time'], f"Task_{i}_end_time")
    model.addConstr(end_time[i] <= service_requests_df.loc[i, 'Latest delivery time'], f"Task_{i}_latest_end")

    # Capability requirement constraint
    task_capability = service_requests_df.loc[i, 'Capability Requirement']
    for j in range(n_agvs):
        agv_capability = agv_capabilities[j]
        if task_capability != agv_capability and agv_capability != 'Heavy':
            model.addConstr(x[i, j] == 0, f"Task_{i}AGV{j}_capability")

# Charge constraints
for j in range(n_agvs):
    # Initial charge level
    model.addConstr(charge_level[0, j] == initial_charge[j], f"Initial_charge_{j}")

    for i in range(n_tasks):
        required_charge = service_requests_df.loc[i, 'min Proc. time'] * discharging_rate
        destination = service_requests_df.loc[i, 'Destination'].strip()
        return_charge = travel_times_to_charging[destination] * discharging_rate

        if i == 0:
            idle_time = start_time[i]
        else:
            idle_time = start_time[i] - end_time[i - 1]

        max_possible_charge = (100 - charge_level[i, j]) / charging_rate
        model.addConstr(charging_time[i, j] <= idle_time, f"Charging_time_idle_{i}_{j}")
        model.addConstr(charging_time[i, j] <= max_possible_charge, f"Charging_time_max_{i}_{j}")

        model.addConstr(
            charge_level[i + 1, j] == charge_level[i, j] - required_charge - return_charge + charging_time[i, j] * charging_rate,
            f"Charge_{i}_{j}"
        )

        # Ensure charge level is non-negative
        model.addConstr(charge_level[i + 1, j] >= 0, f"Non_negative_charge_{i}_{j}")

    # Include charging station task
    model.addConstr(charge_level[n_tasks, j] <= 100, f"Max_charge_{n_tasks}_{j}")

# Charging station capacity constraint
for i in range(n_tasks):
    model.addConstr(gp.quicksum(charging_station_usage[i] for i in range(n_tasks)) <= charging_station_capacity, f"Charging_station_capacity_{i}")

# Sequential task constraints for each AGV
for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] - (1 - x[i, j]) * time_horizon - (1 - x[k, j]) * time_horizon, f"AGV_{j}task{i}sequential{k}")

# Load constraints: AGV load is the sum of the end times of its tasks
for j in range(n_agvs):
    model.addConstr(agv_load[j] == gp.quicksum(end_time[i] * x[i, j] for i in range(n_tasks)), f"AGV_{j}_load")

# Optimize the model
model.optimize()

# Diagnose infeasibility
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    model.computeIIS()
    model.write("model.ilp")
    with open("model.ilp", "r") as f:
        for line in f:
            print(line.strip())

# Ensure agv_tasks and charge_over_time are defined even if infeasible
agv_tasks = {j: [] for j in range(n_agvs)}
charge_over_time = {j: [] for j in range(n_agvs)}

# Extract the solution if feasible
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('x', x)
    start_times_solution = model.getAttr('x', start_time)
    end_times_solution = model.getAttr('x', end_time)
    charge_levels_solution = model.getAttr('x', charge_level)
    tasks = []
    current_charge_level = initial_charge.copy()
    for j in range(n_agvs):
        previous_end_time = 0
        charge_over_time[j].append((0, current_charge_level[j]))  # Initial charge level
        for i in range(n_tasks):
            if solution[i, j] > 0.5:
                start = start_times_solution[i]
                end = end_times_solution[i]
                job_id = service_requests_df.loc[i, 'Job ID']

                # If there is idle time between previous task and current task, schedule charging if needed
                idle_time = start - previous_end_time
                if idle_time > 0:
                    charge_time = min(idle_time, (100 - current_charge_level[j]) / charging_rate)
                    if charge_time > 0:
                        charging_end_time = previous_end_time + charge_time
                        charge_before = current_charge_level[j]
                        current_charge_level[j] += charge_time * charging_rate
                        charge_after = current_charge_level[j]
                        agv_tasks[j].append((previous_end_time, charging_end_time, f"Charging ({charge_before:.1f}% to {charge_after:.1f}%)", 'Charging'))
                        charge_over_time[j].append((charging_end_time, charge_after))
                        previous_end_time = charging_end_time

                discharge_time = (end - start) * discharging_rate
                return_discharge_time = travel_times_to_charging[destination] * discharging_rate
                charge_before = current_charge_level[j]
                current_charge_level[j] -= (discharge_time + return_discharge_time)

                # Ensure the charge level does not go below zero
                if current_charge_level[j] < 0:
                    print(f"Error: AGV {j} has negative charge at task {i}.")
                    current_charge_level[j] = 0

                charge_after = current_charge_level[j]
                tasks.append((i, j, start, end, f"Task {i} (Job ID: {job_id})"))
                agv_tasks[j].append((start, end, f"Task {i} (Job ID: {job_id})", 'Task'))
                charge_over_time[j].append((end, charge_after))
                previous_end_time = end

# Calculate and display idle times and merge with tasks
for j in range(n_agvs):
    agv_tasks[j].sort()  # Sort tasks based on start time
    previous_end_time = 0
    agv_output = []
    for start, end, label, typ in agv_tasks[j]:
        if start > previous_end_time:
            idle_time = start - previous_end_time
            agv_output.append((previous_end_time, start, f"Idle {idle_time:.1f}", 'Idle', idle_time))
        agv_output.append((start, end, label, typ, end - start))
        previous_end_time = end

    print(f"\nAGV {j}:")
    for start, end, label, typ, duration in agv_output:
        if typ == 'Idle':
            print(f"  {label} from {start} to {end} (Duration: {duration})")
        elif typ == 'Charging':
            print(f"  {label} from {start} to {end} (Duration: {duration})")
        else:
            print(f"  {label} starts at {start} and ends at {end}")
    agv_tasks[j] = agv_output

# Plot Gantt Chart
fig, ax1 = plt.subplots(figsize=(15, 10))
fig2, axs = plt.subplots(len(agv_tasks), 1, figsize=(15, 10), sharex=True)

# Plot the Gantt chart
colors = {'Task': 'tab:blue', 'Idle': 'tab:red', 'Charging': 'tab:green'}
labels = [f'AGV {j}' for j in agv_tasks.keys()]

for j, tasks in agv_tasks.items():
    for (start, end, label, typ, duration) in tasks:
        ax1.barh(j, end - start, left=start, color=colors[typ], edgecolor='black', height=0.5)
        if typ == 'Task':
            ax1.text((start + end) / 2, j, label, ha='center', va='center', color='white', fontsize=10, rotation=90)
        elif typ == 'Idle':
            ax1.text((start + end) / 2, j - 0.4, label, ha='center', va='center', color='black', fontsize=10)

ax1.set_yticks(range(len(agv_tasks)))
ax1.set_yticklabels(labels)
ax1.set_xlabel('Time')
ax1.set_title('AGV Task Scheduling Gantt Chart')
patches = [mpatches.Patch(color=colors[typ], label=typ) for typ in colors]
ax1.legend(handles=patches)
plt.tight_layout()

# Save the Gantt chart
gantt_chart_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/AGV_Task_Scheduling_Gantt_Chart.png'
fig.savefig(gantt_chart_path)

# Plot the charge percentage over time
for j in agv_tasks.keys():
    if charge_over_time[j]:  # Ensure there are charge data points
        times, charges = zip(*charge_over_time[j])
        axs[j].plot(times, charges, marker='o', linestyle='-', color='tab:green', label=f'AGV {j}')
        axs[j].set_ylabel('Charge Percentage')
        axs[j].set_ylim(0, 110)
        axs[j].legend()
        axs[j].grid(True)

axs[-1].set_xlabel('Time')
axs[0].set_title('AGV Charge Percentage Over Time')
plt.tight_layout()

# Save the charge percentage chart
charge_chart_path = 'C:/Users/balaj/Downloads/beifen_Ma0510/AGV_Charge_Percentage_Over_Time.png'
fig2.savefig(charge_chart_path)

plt.show()
