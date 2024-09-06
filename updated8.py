import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np

# Load data
routes_df = pd.read_excel("C:/Users/balaj/Downloads/Table 1 - All possible routes_y(2).xlsx")
service_requests_df = pd.read_csv('C:/Users/balaj/Downloads/Table 2 -Service request time chart(2).csv')
agv_properties_df = pd.read_excel('C:/Users/balaj/Downloads/Table 3 - AGV properties(1).xlsx')

# Define AGV capabilities and initial charges
agv_capabilities = agv_properties_df.set_index('AGV_ID')['Capability'].to_dict()

# Adjust initial charges to handle negative values (set to 1%) and cap at 100%
initial_charge = agv_properties_df.set_index('AGV_ID')['Initial charge'].to_dict()

# Parameters
n_tasks = len(service_requests_df)
n_agvs = len(agv_capabilities)
time_horizon = max(service_requests_df['Latest delivery time'])  # Total time available
charging_rate = 0.1  # 1% per 10 seconds
discharging_rate_per_sec = 0.05  # 0.5% per 10 seconds means 0.05% per second

# Create a new model
model = gp.Model("AGV_Scheduling")

# Create variables
x = model.addVars(n_tasks, n_agvs, vtype=GRB.BINARY, name="x")  # Task assignment
start_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="start_time")  # Start times
end_time = model.addVars(n_tasks, vtype=GRB.CONTINUOUS, name="end_time")  # End times
agv_load = model.addVars(n_agvs, vtype=GRB.CONTINUOUS, name="agv_load")  # Load of each AGV
charge_level = model.addVars(n_tasks + 1, n_agvs, vtype=GRB.CONTINUOUS, name="charge_level")  # Charge level of each AGV at each task
idle_time = model.addVars(n_tasks, n_agvs, vtype=GRB.CONTINUOUS, name="idle_time")  # Idle time for each AGV

# Additional variables for task balancing
task_count = model.addVars(n_agvs, vtype=GRB.INTEGER, name="task_count")  # Number of tasks assigned to each AGV
max_task_count = model.addVar(vtype=GRB.INTEGER, name="max_task_count")  # Maximum number of tasks assigned to any AGV
min_task_count = model.addVar(vtype=GRB.INTEGER, name="min_task_count")  # Minimum number of tasks assigned to any AGV

# Objective: Minimize the difference in task counts, balance the load across AGVs, and maximize final charge levels
model.setObjective(
    (max_task_count - min_task_count) + 
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
    # Each task must be assigned to exactly one AGV
    model.addConstr(gp.quicksum(x[i, j] for j in range(n_agvs)) == 1, f"Task_{i}_assignment")

    # Task must start within its time window
    model.addConstr(start_time[i] >= service_requests_df.loc[i, 'Earliest pick time'], f"Task_{i}_earliest_start")
    task_duration = routes_df.loc[find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                                service_requests_df.loc[i, 'Unloading Station'], 'P0'), 'Total time']
    model.addConstr(end_time[i] == start_time[i] + task_duration, f"Task_{i}_end_time")
    model.addConstr(end_time[i] <= service_requests_df.loc[i, 'Latest delivery time'], f"Task_{i}_latest_end")

    # Capability requirement constraint
    task_capability = service_requests_df.loc[i, 'Capability Requirement']
    for j in range(n_agvs):
        agv_capability = agv_capabilities[j]
        if task_capability == 'Normal' and agv_capability == 'Heavy':
            continue  # AGV with Heavy capability can do both Heavy and Normal tasks
        elif task_capability != agv_capability:
            model.addConstr(x[i, j] == 0, f"Task_{i}_AGV{j}_capability")

# Task count constraints
for j in range(n_agvs):
    model.addConstr(task_count[j] == gp.quicksum(x[i, j] for i in range(n_tasks)), f"Task_count_{j}")
    model.addConstr(task_count[j] <= max_task_count, f"Max_task_count_{j}")
    model.addConstr(task_count[j] >= min_task_count, f"Min_task_count_{j}")

# Load constraints: AGV load is the sum of the end times of its tasks
for j in range(n_agvs):
    model.addConstr(agv_load[j] == gp.quicksum(end_time[i] * x[i, j] for i in range(n_tasks)), f"AGV_{j}_load")

# Charge constraints
for j in range(n_agvs):
    # Initial charge level (ensure no charge below 1%)
    model.addConstr(charge_level[0, j] <= 100, f"No_over_charge_{j}")
    model.addConstr(charge_level[0, j] >= 1, f"No_negative_charge_{j}")
    model.addConstr(charge_level[0, j] == initial_charge[j], f"Initial_charge_{j}")

    for i in range(n_tasks):
        # Direct route from P0
        direct_route_id = find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                        service_requests_df.loc[i, 'Unloading Station'], 'P0')
        direct_route_time = routes_df.loc[direct_route_id, 'Total time']
        direct_route_charge = direct_route_time * discharging_rate_per_sec

        # Charging route from C1
        charging_route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                          service_requests_df.loc[i, 'Unloading Station'], 'P0')
        charging_route_time = routes_df.loc[charging_route_id, 'Total time']
        charging_route_charge = charging_route_time * discharging_rate_per_sec

        # Adding 1% buffer charge for trip from P0 to C1
        buffer_charge = 1

        # If not enough charge for direct route, AGV must go to C1 first
        if initial_charge[j] < (direct_route_charge + buffer_charge):
            required_charge = charging_route_charge + buffer_charge
            charge_time_needed = required_charge / charging_rate

            # Include trip from P0 to C1 in the output
            p0_to_c1_trip_time = 12.5  # 20 seconds
            p0_to_c1_trip_charge = p0_to_c1_trip_time * discharging_rate_per_sec

            # Log the trip from P0 to C1
            model.addConstr(charge_level[i + 1, j] >= (charge_time_needed * charging_rate) + p0_to_c1_trip_charge, f"Charge_{i}_{j}_after_charging")

        discharge_time = (end_time[i] - start_time[i]) * discharging_rate_per_sec
        if i == 0:
            previous_end_time = 0
        else:
            previous_end_time = end_time[i-1]

        # Idle time calculation
        model.addConstr(idle_time[i, j] >= start_time[i] - previous_end_time)
        model.addConstr(idle_time[i, j] >= 0)

        # Charge level update
        model.addConstr(charge_level[i + 1, j] == charge_level[i, j] - discharge_time + idle_time[i, j] * charging_rate, f"Charge_{i}_{j}")
        model.addConstr(charge_level[i + 1, j] >= 1, f"No_negative_charge_{i}_{j}")  # Ensure AGV charge level does not drop below 1%

    # Ensure charge level does not exceed 100%
    model.addConstr(charge_level[n_tasks, j] <= 100, f"Max_charge_{n_tasks}_{j}")

# Sequential task constraints for each AGV
for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] - (1 - x[i, j]) * time_horizon - (1 - x[k, j]) * time_horizon, f"AGV_{j}task{i}sequential{k}")

# No Loop Constraint
for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] - (1 - x[i, j]) * time_horizon - (1 - x[k, j]) * time_horizon, f"No_Loop_{j}_task{i}_task{k}")

# Headway Constraint
headway_time = 20  # 20s => for fast-paced systems, 60s => for complex systems, less than 20s => buffer will be too small
for j in range(n_agvs):
    for i in range(n_tasks):
        for k in range(i + 1, n_tasks):
            model.addConstr(start_time[k] >= end_time[i] + headway_time - (1 - x[i, j]) * time_horizon - (1 - x[k, j]) * time_horizon, f"Headway_{j}_task{i}_task{k}")

# Determine which AGV can complete Task 0 based on initial charge
task_0_capability = service_requests_df.loc[0, 'Capability Requirement']
task_0_route_id = find_route_id('P0', service_requests_df.loc[0, 'Loading Station'], 
                                service_requests_df.loc[0, 'Unloading Station'], 'P0')
task_0_duration = routes_df.loc[task_0_route_id, 'Total time']
task_0_required_charge = task_0_duration * discharging_rate_per_sec + 1.0  # 1.0% buffer

# Create a constraint that assigns Task 0 based on initial charge
assigned = False
for j in range(n_agvs):
    if agv_capabilities[j] == task_0_capability or (task_0_capability == 'Normal' and agv_capabilities[j] == 'Heavy'):
        if initial_charge[j] >= task_0_required_charge:
            # Assign Task 0 to this AGV directly
            model.addConstr(x[0, j] == 1, f"Force_Assignment_Task_0_to_AGV_{j}")
            assigned = True
            break

# If no AGV has enough initial charge, plan charging first for the most charged AGV
if not assigned:
    best_agv = None
    max_charge = -1
    for j in range(n_agvs):
        if agv_capabilities[j] == task_0_capability or (task_0_capability == 'Normal' and agv_capabilities[j] == 'Heavy'):
            if initial_charge[j] > max_charge:
                max_charge = initial_charge[j]
                best_agv = j

    if best_agv is not None:
        # Plan the charging sequence before assigning Task 0
        required_charge = task_0_required_charge - initial_charge[best_agv]
        charge_time_needed = required_charge / charging_rate
        model.addConstr(charge_level[1, best_agv] >= task_0_required_charge, f"Charge_AGV_{best_agv}_for_Task_0")
        model.addConstr(x[0, best_agv] == 1, f"Force_Assignment_Task_0_to_AGV_{best_agv}")

# Optimize the model
model.optimize()

# Check for infeasibility or unboundedness
if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
    print("Model is infeasible or unbounded. Computing IIS...")
    model.computeIIS()
    model.write("model.ilp")  # Save the IIS to a file for inspection
    print("IIS written to model.ilp")

    # Print out the infeasible constraints
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"Infeasible constraint: {c.constrName}")

# Extract the solution
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('x', x)
    start_times_solution = model.getAttr('x', start_time)
    end_times_solution = model.getAttr('x', end_time)
    charge_levels_solution = model.getAttr('x', charge_level)
    tasks = []
    agv_tasks = {j: [] for j in range(n_agvs)}
    
    charge_over_time = {j: [] for j in range(n_agvs)}
    current_charge_level = initial_charge.copy()

    agv_check_task = {x:[] for x in range(n_agvs)}
    
    for j in range(n_agvs):
        for i in range(n_tasks):
            if solution[i,j] > 0.5:
                agv_check_task[j].append(i)

    ls_task_list={1:[],2:[]}

    current_position = {i:'P0' for i in range(n_agvs)}

    initial_flag = 0
    final_flag = 0

    b = 0
    task_assigned = set()
    
    charge_over_time[j].append((0, current_charge_level[j]))  # Initial charge level
    for j in range(n_agvs):
        previous_end_time = 0
        final_flag = 0
        k = 0
        for i in range(n_tasks):
            final_flag += 1
            if solution[i, j] > 0.5:
                if i not in task_assigned:
                    task_assigned.add(i)
                    start = start_times_solution[i]
                    end = end_times_solution[i]
                    job_id = service_requests_df.loc[i, 'Job ID']
                    
                    # Determine the correct route based on the current position
                    if current_position[j] == 'P0':
                        route_id = find_route_id('P0', service_requests_df.loc[i, 'Loading Station'], 
                                                 service_requests_df.loc[i, 'Unloading Station'], 'P0')
                    else:
                        route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                                 service_requests_df.loc[i, 'Unloading Station'], 'P0')
                        end = end_times_solution[i] - 2.5
                    
                    route_time = routes_df.loc[route_id, 'Total time']
                    total_required_charge = (route_time * discharging_rate_per_sec) + 0.625  # 1% buffer

                    if (service_requests_df.loc[i,'Loading Station']=='L1'):
                        ls_task_list[1].append(i)
                    else:
                        ls_task_list[2].append(i)

                    if current_charge_level[j] < total_required_charge:
                        # Trip to C1 from P0 if not already there
                        if current_position[j] != 'C1':
                            trip_to_c1_start = previous_end_time
                            trip_to_c1_end = trip_to_c1_start + 12.5  # 20 seconds for trip to C1
                            current_charge_level[j] = current_charge_level[j] - (12.5 * discharging_rate_per_sec)  # 20 seconds trip to C1
                            charge_over_time[j].append((trip_to_c1_end, current_charge_level[j]))
                            agv_tasks[j].append((trip_to_c1_start, trip_to_c1_end, f"Trip to C1 from P0 (12.5s buffer)", 'Trip'))
                            previous_end_time = trip_to_c1_end
                            current_position[j] = 'C1'


                        # Charging at C1
                        required_charge = total_required_charge - current_charge_level[j]
                        charge_time_needed = required_charge / charging_rate
                        task_start = service_requests_df.loc[i, 'Earliest pick time']
                        charging_end_time = previous_end_time + charge_time_needed
                        charge_before = current_charge_level[j]
                        current_charge_level[j] += charge_time_needed * charging_rate
                        charge_after = current_charge_level[j]
                        agv_tasks[j].append((previous_end_time, charging_end_time, f"Charging at C1 ({charge_before:.2f}% to {charge_after:.2f}%)", 'Charging'))
                        charge_over_time[j].append((charging_end_time, charge_after))
                        previous_end_time = charging_end_time

                        charge_duration = task_start - previous_end_time

                        if current_charge_level[j] < 100:
                            if charge_duration > 0:
                                max_charge_time = (100 - current_charge_level[j]) / charging_rate
                                actual_charge_time = min(charge_duration, max_charge_time)
                                charging_end_time = previous_end_time + actual_charge_time
                                charge_before = current_charge_level[j]
                                current_charge_level[j] += actual_charge_time * charging_rate
                                charge_after = min(current_charge_level[j], 100)  # Ensure charge level does not exceed 100%
                                if actual_charge_time > 0:
                                    agv_tasks[j].append((previous_end_time, charging_end_time, f"Charging at C1 ({charge_before:.2f}% to {charge_after:.2f}%)", 'Charging'))
                                    charge_over_time[j].append((charging_end_time, charge_after))
                                    previous_end_time = charging_end_time


                        # Update start and end time for the task from C1
                        start = max(previous_end_time, service_requests_df.loc[i, 'Earliest pick time'])  # Ensure start time respects earliest pickup time
                        route_id = find_route_id('C1', service_requests_df.loc[i, 'Loading Station'], 
                                                 service_requests_df.loc[i, 'Unloading Station'], 'P0')
                        route_time = routes_df.loc[route_id, 'Total time']
                        end = start + route_time  # Adjusting for the route from C1

                    discharge_time = (end - start) * discharging_rate_per_sec
                    charge_before = current_charge_level[j]
                    current_charge_level[j] = current_charge_level[j] - discharge_time
                    charge_after = current_charge_level[j]
                    route_id_str = routes_df.loc[route_id, 'Route ID']  # Extracting the route ID string
                    tasks.append((i, j, start, end, f"Task {i} (Job ID: {job_id}, Route ID: {route_id_str}) from {current_position[j]}"))
                    agv_tasks[j].append((start, end, f"Task {i} (Job ID: {job_id}, Route ID: {route_id_str}) from {current_position[j]}", 'Task'))
                    charge_over_time[j].append((end, charge_after))
                    previous_end_time = end
                    current_position[j] = 'P0'  # After task, AGV returns to P0

                    
                    # After completing a task, check if the AGV needs to charge before the next task
                    if i < n_tasks - 1:
                        
                        if k < (len(agv_check_task[j]) - 1):
                            next_task_start = service_requests_df.loc[agv_check_task[j][k+1], 'Earliest pick time']
                            k+=1
                        else: 
                            next_task_start = 0  

                        idle_duration = next_task_start - previous_end_time
                    
                        if idle_duration > 0:                            
                            # Trip to C1 for charging if not already there
                            if current_charge_level[j] < 100:
                                if current_position[j] != 'C1':
                                    trip_to_c1_start = previous_end_time
                                    trip_to_c1_end = trip_to_c1_start + 12.5  # 20 seconds for trip to C1
                                    current_charge_level[j] = current_charge_level[j] - 12.5 * discharging_rate_per_sec  # 20 seconds trip to C1
                                    charge_over_time[j].append((trip_to_c1_end, current_charge_level[j]))
                                    agv_tasks[j].append((trip_to_c1_start, trip_to_c1_end, f"Trip to C1 from P0 (12.5s buffer)", 'Trip'))
                                    previous_end_time = trip_to_c1_end
                                    current_position[j] = 'C1'                               

                            idle_dummy = next_task_start - previous_end_time   

                            if idle_dummy > 0:
                                if current_charge_level[j] < 100:
                                    # Charge to maximum capacity during idle time
                                    max_charge_time = (100 - current_charge_level[j]) / charging_rate
                                    actual_charge_time = min(idle_dummy, max_charge_time)
                                    charging_end_time = previous_end_time + actual_charge_time
                                    charge_before = current_charge_level[j]
                                    current_charge_level[j] += actual_charge_time * charging_rate
                                    charge_after = min(current_charge_level[j], 100)  # Ensure charge level does not exceed 100%
                                    if actual_charge_time > 0:
                                        agv_tasks[j].append((previous_end_time, charging_end_time, f"Charging at C1 ({charge_before:.2f}% to {charge_after:.2f}%)", 'Charging'))
                                        charge_over_time[j].append((charging_end_time, charge_after))
                                        previous_end_time = charging_end_time
                                else:
                                    idle_start = previous_end_time
                                    idle_end = idle_start + idle_dummy
                                    agv_tasks[j].append((idle_start, idle_end, f"Idle from {idle_start:.2f} to {idle_end:.2f}", 'Idle'))
                                    previous_end_time = idle_end
                                    charge_over_time[j].append((idle_end, current_charge_level[j]))

            if initial_flag == 0:
            
                for b in range(n_agvs):

                    if b != j:

                        idle_duration = service_requests_df.loc[agv_check_task[b][0], 'Earliest pick time']                    
                        if idle_duration > 0:
                            trip_to_c1_end = 0                            
                            # Trip to C1 for charging if not already there
                            if current_charge_level[b] < 100:
                                if current_position[b] != 'C1':
                                    trip_to_c1_start = 0
                                    trip_to_c1_end = trip_to_c1_start + 12.5  # 20 seconds for trip to C1
                                    current_charge_level[b] = current_charge_level[b] - 12.5 * discharging_rate_per_sec  # 12.5 seconds trip to C1
                                    charge_over_time[b].append((trip_to_c1_end, current_charge_level[b]))
                                    agv_tasks[b].append((trip_to_c1_start, trip_to_c1_end, f"Trip to C1 from P0 (12.5s buffer)", 'Trip'))
                                    current_position[b] = 'C1'                               

                            idle_dummy = service_requests_df.loc[agv_check_task[b][0], 'Earliest pick time']   

                            if idle_dummy > 0:
                                if current_charge_level[b] < 100:
                                    # Charge to maximum capacity during idle time
                                    max_charge_time = (100 - current_charge_level[b]) / charging_rate
                                    charging_end_time = min(idle_dummy, max_charge_time)
                                    charge_before = current_charge_level[b]
                                    current_charge_level[b] += charging_end_time * charging_rate
                                    charge_after = min(current_charge_level[b], 100)  # Ensure charge level does not exceed 100%
                                    if charging_end_time > 0:
                                        agv_tasks[b].append((trip_to_c1_end, charging_end_time, f"Charging at C1 ({charge_before:.2f}% to {charge_after:.2f}%)", 'Charging'))
                                        charge_over_time[b].append((charging_end_time, charge_after))
                                else:
                                    idle_start = trip_to_c1_end
                                    idle_end = idle_start + idle_dummy
                                    agv_tasks[b].append((idle_start, idle_end, f"Idle from {idle_start:.2f} to {idle_end:.2f}", 'Idle'))
                                    charge_over_time[b].append((idle_end, current_charge_level[b]))

                    initial_flag = 1


    if final_flag >= 19:
        for b in range(n_agvs):
            trip_to_c1_start = agv_tasks[b][-1][1]
            if current_charge_level[b] < 100:
                if current_position[b] != 'C1':
                    trip_to_c1_end = trip_to_c1_start + 12.5  # 20 seconds for trip to C1
                    current_charge_level[b] = current_charge_level[b] - 12.5 * discharging_rate_per_sec  # 20 seconds trip to C1
                    charge_over_time[b].append((trip_to_c1_end, current_charge_level[b]))
                    agv_tasks[b].append((trip_to_c1_start, trip_to_c1_end, f"Trip to C1 from P0 (12.5s buffer)", 'Trip'))
                    current_position[b] = 'C1'   
                    
                    charge_end_limit =  trip_to_c1_end + 167.5
                    
                    # Charge to maximum capacity during idle time
                    max_charge_time = (100 - current_charge_level[b]) / charging_rate
                    charging_duration = min(167.5, max_charge_time)
                    charging_end_time = trip_to_c1_end + charging_duration
                    charge_before = current_charge_level[b]
                    current_charge_level[b] += charging_duration * charging_rate
                    charge_after = min(current_charge_level[b], 100)  # Ensure charge level does not exceed 100%
                    if charging_duration > 0:
                        agv_tasks[b].append((trip_to_c1_end, charging_end_time , f"Charging at C1 ({charge_before:.2f}% to {charge_after:.2f}%)", 'Charging'))
                        charge_over_time[b].append((charging_end_time, charge_after))      

                    if charging_end_time < charge_end_limit:
                        idle_start = charging_end_time
                        idle_end = charge_end_limit
                        agv_tasks[b].append((idle_start, idle_end, f"Idle from {idle_start:.2f} to {idle_end:.2f}", 'Idle'))
                        charge_over_time[b].append((idle_end, current_charge_level[b]))    

            else:
                idle_start = trip_to_c1_start
                idle_end = idle_start + 180
                agv_tasks[b].append((idle_start, idle_end, f"Idle from {idle_start:.2f} to {idle_end:.2f}", 'Idle'))
                charge_over_time[b].append((idle_end, current_charge_level[b]))                            

    # Consolidate charging periods to avoid redundant statements
    for j in range(n_agvs):
        consolidated_agv_tasks = []
        previous_task = None
        for task in agv_tasks[j]:
            if previous_task and previous_task[3] == 'Charging' and task[3] == 'Charging':
                consolidated_agv_tasks[-1] = (previous_task[0], task[1], f"Charging at C1 ({previous_task[2].split('(')[1].split(')')[0].split(' to ')[0]} to {task[2].split('(')[1].split(')')[0].split(' to ')[1]})", 'Charging')
            else:
                consolidated_agv_tasks.append(task)
            previous_task = task
        agv_tasks[j] = consolidated_agv_tasks 

    # Calculate and display idle times and merge with tasks
    for j in range(n_agvs):
        agv_tasks[j].sort()  # Sort tasks based on start time
        previous_end_time = 0
        agv_output = []
        for start, end, label, typ in agv_tasks[j]:  # Adjusted to unpack only 4 values
            if start > previous_end_time:
                idle_duration = start - previous_end_time
                agv_output.append((previous_end_time, start, f"Idle from {previous_end_time:.2f} to {start:.2f}", 'Idle'))
            agv_output.append((start, end, label, typ))
            previous_end_time = end

        print(f"\nAGV {j}:")
        for start, end, label, typ in agv_output:  # Adjusted to unpack only 4 values
            if typ == 'Idle':
                print(f"  {label} (Duration: {end - start:.2f})")
            elif typ == 'Charging' or typ == 'Trip':
                print(f"  {label} from {start:.2f} to {end:.2f} (Duration: {end - start:.2f})")
            else:
                print(f"  {label} starts at {start:.2f} and ends at {end:.2f}")
        agv_tasks[j] = agv_output

# After model optimization and processing, we will generate the Gantt chart data
output_data = []

now = datetime.datetime.now()

for j in range(n_agvs):
    previous_charge_level = initial_charge[j]
    for start, end, label, typ in agv_tasks[j]:
        if typ == 'Task':
            task_type = "Task"
            job_id = label.split('Job ID: ')[1].split(',')[0]  # Extract Job ID for tasks
            task_type = task_type + " - " + job_id
        elif typ == "Trip":
            task_type = "Trip to C1"
        else:
            task_type = typ
        
        category = typ

        start_time = now + datetime.timedelta(seconds=int(start))
        end_time = now + datetime.timedelta(seconds=int(end))

        duration = end - start
        start_charge = previous_charge_level
        if typ == 'Charging':
            end_charge = min(start_charge + duration * charging_rate, 100)
            if end_charge > 100:
                #print(f"For AGV_{j}, overcharge level is set to 100%")
                end_charge = 100
        elif typ == 'Idle':
            end_charge = start_charge
        else:
            end_charge = start_charge - duration * discharging_rate_per_sec
            if end_charge < 1:
                #print(f"For AGV_{j}, negative charge level is set to 1%")
                end_charge = 1

        # Add data to output
        output_data.append({
            "AGV_ID": f"AGV {j}",
            "Task Type": task_type,
            "Start time": start_time,
            "End time": end_time,
            "Start charge": start_charge,
            "End charge": end_charge,
            "Duration": duration,
            "Category": typ
        })
        
        previous_charge_level = end_charge  # Update for the next task

# Convert the output data to a DataFrame for Plotly
plotly_df = pd.DataFrame(output_data)

# Define custom colors for different task types
color_discrete_map = {
    "Task": 'blue',
    "Charging": 'green',
    "Idle": 'red',
    "Trip": 'orange'
}

# Create the interactive Gantt chart using Plotly
fig = px.timeline(
    plotly_df, 
    x_start="Start time", 
    x_end="End time", 
    y="AGV_ID", 
    color="Category",  # Use Task Type for visual differentiation
    title="AGV Scheduling Gantt Chart",
    hover_data={
        "Task Type": True,
        "Start time": True,
        "End time": True,
        "Start charge": True,
        "End charge": True,
        "Duration": True,
        "Category": False
    },
    color_discrete_map=color_discrete_map  # Apply the custom colors
)

# Update layout to enhance clarity
fig.update_yaxes(categoryorder="total ascending")
fig.update_layout(xaxis_title="Time", yaxis_title="AGV", showlegend=True)

fig.update_layout(
    title="AGV Scheduling Gantt Chart",
    xaxis=dict(
        title="Time",
        showgrid=True,
        zeroline=False,
        showline=True,
        showticklabels=True,
    ),
    yaxis=dict(
        title="AGV",
        showgrid=True,
        zeroline=False,
        showline=True,
        showticklabels=True,
    ),
    margin=dict(l=40, r=40, t=40, b=40),
    height=600
)

# Define a list of colors for different AGVs
colors = ['blue', 'green', 'red']  # Extend this list if you have more AGVs

# Initialize a Plotly figure
fig1 = go.Figure()

# Plot charge percentage over time for each AGV with separate colors
for i, j in enumerate(agv_tasks.keys()):
    times, charges = zip(*charge_over_time[j])
    fig1.add_trace(go.Scatter(
        x=times,
        y=charges,
        mode='lines+markers',
        name=f'AGV {j}',
        marker=dict(color=colors[i % len(colors)]),  # Assign color from the list
        line=dict(color=colors[i % len(colors)]),
        hoverinfo='x+y+name',
        hovertemplate='%{y:.2f}% charge at %{x}<extra></extra>',  # Custom hover template
        opacity=0.8  # Slight transparency for non-hovered lines
    ))

# Update layout to enhance clarity and aesthetics
fig1.update_layout(
    title="AGV Charge Percentage Over Time",
    xaxis_title="Time",
    yaxis_title="Charge Percentage",
    yaxis=dict(range=[0, 110]),  # Set Y-axis limits from 0 to 110%
    legend=dict(x=0.01, y=0.99),  # Position the legend
    template="plotly_white",
    margin=dict(l=40, r=40, t=40, b=40),
    height=600,
    hovermode='x unified'  # Unified hover mode to highlight the line
)

# Save the figure as an HTML file
fig.write_html("C:/Users/balaj/Downloads/agv_schedule_gantt_chart_clean.html")
fig1.write_html("C:/Users/balaj/Downloads/agv_charge_percentage_over_time.html")

# Export the output data to CSV
output_file_path = "C:/Users/balaj/Downloads/agv_scheduling_output_clean.csv"
plotly_df.to_csv(output_file_path, index=False)

# Function to assign positions based on type
def assign_positions(G):
    pos = {}
    charging_y, idle_y, tasks_y, trip_y = 0.8, 0.6, 0.4, 0.2
    x_offset = 0
    for node in G.nodes():
        if 'Charging' in node:
            pos[node] = (x_offset, charging_y)
        elif 'Idle' in node:
            pos[node] = (x_offset, idle_y)
        elif 'Task' in node:
            pos[node] = (x_offset, tasks_y)
        elif 'Trip' in node:
            pos[node] = (x_offset, trip_y)
        x_offset += 0.2  # Increase x position offset for more spacing
    return pos

# Function to create a graph for an individual AGV with arrow directions
def create_agv_graph(G, agv_id):
    plt.figure(figsize=(10, 6))  # Create a new figure for each AGV

    pos = assign_positions(G)  # Assign custom positions based on type
    
    # Draw the nodes with adjusted size and colors
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9)
    
    # Draw the edges with arrow marks for directions
    for edge in G.edges:
        style = 'solid'
        if 'Charging' in edge[0] or 'Charging' in edge[1]:
            style = 'dashed'
        elif 'Idle' in edge[0] or 'Idle' in edge[1]:
            style = 'dotted'
        nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='-|>', arrowsize=20, edge_color='black', style=style)
    
    # Draw the labels with improved clarity
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=8, font_weight='bold', verticalalignment='center', horizontalalignment='center')
    
    plt.title(f"AGV {agv_id} Task Flow")
    plt.axis('off')  # Turn off the axes for a cleaner look

    plt.savefig(f'C:/Users/balaj/Downloads/AGV {agv_id} Task Flow.png')  # Display the graph for the current AGV

# Initialize a list of graphs, one for each AGV
graphs = {}

for agv, tasks in agv_tasks.items():
    G = nx.DiGraph()
    previous_node = None
    
    for start, end, label, typ in tasks:
        # Create a concise node label with line breaks
        task_node = f"{typ}\n[{start:.1f},\n{end:.1f}]"
        
        # Add nodes to the graph
        G.add_node(task_node, color=('lightgreen' if 'Task' in typ else 'lightblue') if 'Charging' not in typ else 'yellow')
        
        # Add an edge from the previous node to the current node
        if previous_node:
            G.add_edge(previous_node, task_node)
        
        previous_node = task_node  # Update the previous node
    
    graphs[agv] = G  # Store the graph for this AGV

# Create and display separate graphs for each AGV with directional arrows
for agv_id, G in graphs.items():
    create_agv_graph(G, agv_id)


ls_task_list[1] =sorted(ls_task_list[1])
ls_task_list[2] =sorted(ls_task_list[2])

# Initialize the directed graph
G = nx.DiGraph()

# Define positions and properties
positions = {}
node_size = 1000
x_gap = 2  # Gap between tasks
y_gap = 0.5  # Reduce gap between AGV rows to bring loading stations closer

# Adjust the positions for Loading Stations to bring them closer
ls_y_positions = {1: 6, 2: 4}  # LS1 on top, LS2 below but closer to LS1

# Define different colors for different loading stations
color_map = {1: 'lightblue', 2: 'lightcoral'}

# Place tasks in a single horizontal line per loading station
for ls, tasks in ls_task_list.items():
    for i, task in enumerate(tasks):
        task_node = f"SR{task}"
        G.add_node(task_node)
        x_pos = x_gap * i
        y_pos = ls_y_positions[ls]
        positions[task_node] = (x_pos, y_pos)

# Place AGVs and connect to their first task
for agv, tasks in agv_check_task.items():
    agv_node = f"AGV{agv}"
    G.add_node(agv_node)
    positions[agv_node] = (-x_gap, ls_y_positions[1] - y_gap * (agv + 1))  # Align AGVs to the left

    # Connect AGV to its tasks directly
    prev_node = agv_node
    for task in tasks:
        task_node = f"SR{task}"
        G.add_edge(prev_node, task_node)
        prev_node = task_node



# Draw the graph using Plotly
edge_traces = []
colors = px.colors.qualitative.Set1  # Use a predefined color set from Plotly

def create_curve(x0, y0, x1, y1, offset=0.2):
    # Calculate the control points for a Bezier curve
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    control_x = xm + offset * (y1 - y0)
    control_y = ym + offset * (x0 - x1)
    t = np.linspace(0, 1, 100)
    bezier_x = (1-t)*2 * x0 + 2*(1-t)*t * control_x + t*2 * x1
    bezier_y = (1-t)*2 * y0 + 2*(1-t)*t * control_y + t*2 * y1
    return bezier_x, bezier_y

for i, (agv, tasks) in enumerate(agv_check_task.items()):
    edge_x = []
    edge_y = []
    prev_node = f"AGV{agv}"
    for task in tasks:
        task_node = f"SR{task}"
        x0, y0 = positions[prev_node]
        x1, y1 = positions[task_node]
        bezier_x, bezier_y = create_curve(x0, y0, x1, y1, offset=0.2)
        edge_x.extend(bezier_x)
        edge_y.extend(bezier_y)
        edge_x.append(None)  # Add None to break the line between segments
        edge_y.append(None)
        prev_node = task_node

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color=colors[i % len(colors)]),
        hoverinfo='none',
        mode='lines',
        name=f'AGV{agv}'
    )
    edge_traces.append(edge_trace)

node_x = []
node_y = []
for node in G.nodes():
    x, y = positions[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[f'{node}' for node in G.nodes()],
    textposition="top center",
    marker=dict(
        showscale=False,
        color=[color_map[1] if 'SR' in node and int(node[2:]) in ls_task_list[1] else
               color_map[2] if 'SR' in node and int(node[2:]) in ls_task_list[2] else
               'lightgreen'
               for node in G.nodes()],
        size=20,
        line_width=2))

fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title='Interactive Directed Graph with AGV Tasks',
                    titlefont_size=16,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)))

fig.write_html("C:/Users/balaj/Downloads/Interactive Directed Graph with AGV Tasks.html")

print(f"Output exported to {output_file_path}")
