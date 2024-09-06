import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
routes_df = pd.read_excel('C:/Users/balaj/Downloads/Table 1 - All possible routes_y(1).xlsx')
service_requests_df = pd.read_csv('C:/Users/balaj/Downloads/Table 2 -Service request time chart(1).csv')
agv_properties_df = pd.read_excel('C:/Users/balaj/Downloads/Table 3 - AGV properties.xlsx')

# Define AGV capabilities and initial charges
agv_capabilities = agv_properties_df.set_index('AGV_ID')['Capability'].to_dict()
initial_charge = agv_properties_df.set_index('AGV_ID')['Initial charge'].to_dict()

# Parameters
n_tasks = len(service_requests_df)
n_agvs = len(agv_capabilities)
charging_rate = 0.1  # 1% per 10 seconds
discharging_rate_per_sec = 0.05  # 0.5% per 10 seconds

# Create a new model
model = gp.Model("AGV_Scheduling")

# Create variables
x = model.addVars(n_tasks, n_agvs, vtype=GRB.BINARY, name="x")
start_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="start_time")
end_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="end_time")
agv_load = model.addVars(n_agvs, vtype=GRB.CONTINUOUS, name="agv_load")
charge_level = model.addVars(n_tasks + 1, n_agvs, vtype=GRB.CONTINUOUS, name="charge_level")
idle_time = model.addVars(n_tasks, n_agvs, vtype=GRB.CONTINUOUS, name="idle_time")

# Objective
model.setObjective(
    gp.quicksum(agv_load[j] for j in range(n_agvs)) - 
    gp.quicksum(charge_level[n_tasks, j] for j in range(n_agvs)) + 
    gp.quicksum(idle_time[i, j] for i in range(n_tasks) for j in range(n_agvs)), 
    GRB.MINIMIZE
)

# Helper function to find the route ID
def find_route_id(origin, loading_station, unloading_station, destination):
    route = routes_df[(routes_df['Origin'] == origin) & 
                      (routes_df['Loading Station'] == loading_station) & 
                      (routes_df['Unloading Station'] == unloading_station) & 
                      (routes_df['Destination'] == destination)]
    return route.index[0] if not route.empty else None

# Constraints
for i in range(n_tasks):
    model.addConstr(gp.quicksum(x[i, j] for j in range(n_agvs)) == 1, f"Task_{i}_assignment")
    model.addConstr(start_time[i] >= service_requests_df.loc[i, 'Earliest pick time'], f"Task_{i}_earliest_start")
    task_duration = routes_df.loc[find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                                service_requests_df.loc[i, 'Unloading Station'], 'P0'), 'Total time']
    model.addConstr(end_time[i] == start_time[i] + task_duration, f"Task_{i}_end_time")
    model.addConstr(end_time[i] <= service_requests_df.loc[i, 'Latest delivery time'], f"Task_{i}_latest_end")
    
    task_capability = service_requests_df.loc[i, 'Capability Requirement']
    for j in range(n_agvs):
        if task_capability != agv_capabilities[j]:
            model.addConstr(x[i, j] == 0, f"Task_{i}AGV{j}_capability")

for j in range(n_agvs):
    model.addConstr(charge_level[0, j] == initial_charge[j], f"Initial_charge_{j}")

    for i in range(n_tasks):
        direct_route_id = find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                        service_requests_df.loc[i, 'Unloading Station'], 'P0')
        direct_route_time = routes_df.loc[direct_route_id, 'Total time']
        direct_route_charge = direct_route_time * discharging_rate_per_sec

        charging_route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                          service_requests_df.loc[i, 'Unloading Station'], 'P0')
        charging_route_time = routes_df.loc[charging_route_id, 'Total time']
        charging_route_charge = charging_route_time * discharging_rate_per_sec

        buffer_charge = 1

        if initial_charge[j] < (direct_route_charge + buffer_charge):
            required_charge = charging_route_charge + buffer_charge
            charge_time_needed = required_charge / charging_rate

            p0_to_c1_trip_time = 20
            p0_to_c1_trip_charge = p0_to_c1_trip_time * discharging_rate_per_sec

            model.addConstr(charge_level[i + 1, j] >= (charge_time_needed * charging_rate) + p0_to_c1_trip_charge, f"Charge_{i}_{j}_after_charging")

        discharge_time = (end_time[i] - start_time[i]) * discharging_rate_per_sec
        if i == 0:
            previous_end_time = 0
        else:
            previous_end_time = end_time[i-1]

        model.addConstr(idle_time[i, j] >= start_time[i] - previous_end_time)
        model.addConstr(idle_time[i, j] >= 0)

        model.addConstr(charge_level[i + 1, j] == charge_level[i, j] - discharge_time + idle_time[i, j] * charging_rate, f"Charge_{i}_{j}")
        model.addConstr(charge_level[i + 1, j] >= 0, f"No_negative_charge_{i}_{j}")

    model.addConstr(charge_level[n_tasks, j] <= 100, f"Max_charge_{n_tasks}_{j}")

for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] - (1 - x[i, j]) * 10000 - (1 - x[k, j]) * 10000, f"AGV_{j}task{i}sequential{k}")

for j in range(n_agvs):
    model.addConstr(agv_load[j] == gp.quicksum(end_time[i] * x[i, j] for i in range(n_tasks)), f"AGV_{j}_load")

# Optimize the model
model.optimize()

# Check for infeasibility
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible. Computing IIS.")
    model.computeIIS()
    model.write("model.ilp")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"Infeasible constraint: {c.constrName}")

# Extract the solution
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('x', x)
    start_times_solution = model.getAttr('x', start_time)
    end_times_solution = model.getAttr('x', end_time)
    agv_tasks = {j: [] for j in range(n_agvs)}
    charge_over_time = {j: [] for j in range(n_agvs)}
    current_charge_level = initial_charge.copy()
    agv_check_task = {x:[] for x in range(n_agvs)}
    
    for j in range(n_agvs):
        for i in range(n_tasks):
            if solution[i,j] > 0.5:
                agv_check_task[j].append(i)

    current_position = {i:'P0' for i in range(n_agvs)}
    initial_flag = 0

    for j in range(n_agvs):
        previous_end_time = 0
        k = 0
        for i in range(n_tasks):
            if solution[i, j] > 0.5:
                start = start_times_solution[i]
                end = end_times_solution[i]
                job_id = service_requests_df.loc[i, 'Job ID']
                    
                if current_position[j] == 'P0':
                    route_id = find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                             service_requests_df.loc[i, 'Unloading Station'], 'P0')
                else:
                    route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                             service_requests_df.loc[i, 'Unloading Station'], 'P0')
                    
                route_time = routes_df.loc[route_id, 'Total time']
                total_required_charge = (route_time * discharging_rate_per_sec) + 0.5

                if current_charge_level[j] < total_required_charge:
                    if current_position[j] != 'C1':
                        trip_to_c1_start = previous_end_time
                        trip_to_c1_end = trip_to_c1_start + 20
                        current_charge_level[j] -= 20 * discharging_rate_per_sec
                        charge_over_time[j].append((trip_to_c1_end, current_charge_level[j]))
                        agv_tasks[j].append((trip_to_c1_start, trip_to_c1_end, "Trip to C1", 'Trip'))
                        previous_end_time = trip_to_c1_end
                        current_position[j] = 'C1'

                    # Charging at C1
                    required_charge = total_required_charge - current_charge_level[j]
                    charge_time_needed = required_charge / charging_rate
                    charging_end_time = previous_end_time + charge_time_needed
                    charge_before = current_charge_level[j]
                    current_charge_level[j] += charge_time_needed * charging_rate
                    charge_after = current_charge_level[j]
                    agv_tasks[j].append((previous_end_time, charging_end_time, "Charging", 'Charging'))
                    charge_over_time[j].append((charging_end_time, charge_after))
                    previous_end_time = charging_end_time

                    # Update start and end time for the task from C1
                    start = max(previous_end_time, service_requests_df.loc[i, 'Earliest pick time'])
                    route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                             service_requests_df.loc[i, 'Unloading Station'], 'P0')
                    route_time = routes_df.loc[route_id, 'Total time']
                    end = start + route_time

                discharge_time = (end - start) * discharging_rate_per_sec
                charge_before = current_charge_level[j]
                current_charge_level[j] -= discharge_time
                charge_after = current_charge_level[j]
                route_id_str = routes_df.loc[route_id, 'Route ID']
                agv_tasks[j].append((start, end, job_id, 'Task'))
                charge_over_time[j].append((end, charge_after))
                previous_end_time = end
                current_position[j] = 'P0'

                # After completing a task, check if the AGV needs to charge before the next task
                if i < n_tasks - 1:
                    if k < (len(agv_check_task[j]) - 1):
                        next_task_start = service_requests_df.loc[agv_check_task[j][k+1], 'Earliest pick time']
                        k += 1
                    else:
                        next_task_start = 0  

                    idle_duration = next_task_start - previous_end_time
                    if idle_duration > 0:                            
                        if current_charge_level[j] < 100:
                            if current_position[j] != 'C1':
                                trip_to_c1_start = previous_end_time
                                trip_to_c1_end = trip_to_c1_start + 20
                                current_charge_level[j] -= 20 * discharging_rate_per_sec
                                charge_over_time[j].append((trip_to_c1_end, current_charge_level[j]))
                                agv_tasks[j].append((trip_to_c1_start, trip_to_c1_end, "Trip to C1", 'Trip'))
                                previous_end_time = trip_to_c1_end
                                current_position[j] = 'C1'                               

                            idle_dummy = next_task_start - previous_end_time   

                            if idle_dummy > 0:
                                if current_charge_level[j] < 100:
                                    max_charge_time = (100 - current_charge_level[j]) / charging_rate
                                    actual_charge_time = min(idle_dummy, max_charge_time)
                                    charging_end_time = previous_end_time + actual_charge_time
                                    charge_before = current_charge_level[j]
                                    current_charge_level[j] += actual_charge_time * charging_rate
                                    charge_after = min(current_charge_level[j], 100)
                                    if actual_charge_time > 0:
                                        agv_tasks[j].append((previous_end_time, charging_end_time, "Charging", 'Charging'))
                                        charge_over_time[j].append((charging_end_time, charge_after))
                                        previous_end_time = charging_end_time
                                else:
                                    idle_start = previous_end_time
                                    idle_end = idle_start + idle_dummy
                                    agv_tasks[j].append((idle_start, idle_end, "Idle", 'Idle'))
                                    previous_end_time = idle_end
                                    charge_over_time[j].append((idle_end, current_charge_level[j]))

            if initial_flag == 0:
                for b in range(n_agvs):
                    if b != j:
                        idle_duration = service_requests_df.loc[agv_check_task[b][0], 'Earliest pick time']                    
                        if idle_duration > 0:                            
                            if current_charge_level[b] < 100:
                                if current_position[b] != 'C1':
                                    trip_to_c1_start = 0
                                    trip_to_c1_end = trip_to_c1_start + 20
                                    current_charge_level[b] -= 20 * discharging_rate_per_sec
                                    charge_over_time[b].append((trip_to_c1_end, current_charge_level[b]))
                                    agv_tasks[b].append((trip_to_c1_start, trip_to_c1_end, "Trip to C1", 'Trip'))
                                    current_position[b] = 'C1'                               

                            idle_dummy = service_requests_df.loc[agv_check_task[b][0], 'Earliest pick time']   

                            if idle_dummy > 0:
                                if current_charge_level[b] < 100:
                                    max_charge_time = (100 - current_charge_level[b]) / charging_rate
                                    charging_end_time = min(idle_dummy, max_charge_time)
                                    charge_before = current_charge_level[b]
                                    current_charge_level[b] += charging_end_time * charging_rate
                                    charge_after = min(current_charge_level[b], 100)
                                    if charging_end_time > 0:
                                        agv_tasks[b].append((trip_to_c1_end, charging_end_time, "Charging", 'Charging'))
                                        charge_over_time[b].append((charging_end_time, charge_after))
                                else:
                                    idle_start = trip_to_c1_end
                                    idle_end = idle_start + idle_dummy
                                    agv_tasks[b].append((idle_start, idle_end, "Idle", 'Idle'))
                                    charge_over_time[b].append((idle_end, current_charge_level[b]))

            initial_flag = 1

    # Consolidate charging periods to avoid redundant statements
    for j in range(n_agvs):
        consolidated_agv_tasks = []
        previous_task = None
        for task in agv_tasks[j]:
            if previous_task and previous_task[3] == 'Charging' and task[3] == 'Charging':
                consolidated_agv_tasks[-1] = (
                    previous_task[0], task[1], 
                    f"Charging ({previous_task[2].split('(')[1].split(')')[0].split(' to ')[0]}% to {task[2].split('(')[1].split(')')[0].split(' to ')[1]}%)", 
                    'Charging'
                )
            else:
                consolidated_agv_tasks.append(task)
            previous_task = task
        agv_tasks[j] = consolidated_agv_tasks

    # Plot Gantt Chart
    fig, ax1 = plt.subplots(figsize=(15, 10))
    fig2, axs = plt.subplots(len(agv_tasks), 1, figsize=(15, 10), sharex=True)

    # Plot the Gantt chart
    colors = {'Task': 'tab:blue', 'Idle': 'tab:red', 'Charging': 'tab:green', 'Trip': 'tab:orange'}
    labels = [f'AGV {j}' for j in agv_tasks.keys()]

    for j, tasks in agv_tasks.items():
        for (start, end, label, typ) in tasks:  # Adjusted to unpack only 4 values
            ax1.barh(j, end - start, left=start, color=colors[typ], edgecolor='black', height=0.5)
            if typ == 'Task':
                ax1.text((start + end) / 2, j, label, ha='center', va='center', color='white', fontsize=6, rotation=90)

    ax1.set_yticks(range(len(agv_tasks)))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Time')
    ax1.set_title('AGV Task Scheduling Gantt Chart')
    patches = [mpatches.Patch(color=colors[typ], label=typ) for typ in colors]
    ax1.legend(handles=patches)

    # Plot the charge percentage over time
    for j in agv_tasks.keys():
        times, charges = zip(*charge_over_time[j])
        axs[j].plot(times, charges, marker='o', linestyle='-', color='tab:green', label=f'AGV {j}')
        axs[j].set_ylabel('Charge Percentage')
        axs[j].set_ylim(0, 110)
        axs[j].legend()
        axs[j].grid(True)

    axs[-1].set_xlabel('Time')
    axs[0].set_title('AGV Charge Percentage Over Time')

    plt.tight_layout()
    plt.show()

    # Export the task schedule to CSV
    output_data = []

    for j in range(n_agvs):
        previous_charge_level = initial_charge[j]
        for start, end, label, typ in agv_tasks[j]:
            if typ == 'Task':
                task_type = label
            else:
                task_type = typ

            duration = end - start
            start_charge = previous_charge_level
            if typ == 'Charging':
                end_charge = min(start_charge + duration * charging_rate, 100)
            else:
                end_charge = max(start_charge - duration * discharging_rate_per_sec, 0)
            
            output_data.append({
                "AGV_ID": j,
                "Task Type": task_type,
                "Start time": start,
                "End time": end,
                "Start charge": start_charge,
                "End charge": end_charge,
                "Duration": duration
            })
            
            previous_charge_level = end_charge

    output_df = pd.DataFrame(output_data)
    output_file_path = "C:/Users/balaj/Downloads/agv_scheduling_output.csv"
    output_df.to_csv(output_file_path, index=False)

    print(f"Output exported to {output_file_path}")