# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/9/4 11:06 

import pandas as pd
import numpy as np
import copy
import pickle
import os

# Normalize function
def normalize(data):
    normalized_data = copy.deepcopy(data)
    min_val = np.min(normalized_data,axis=0)
    max_val = np.max(normalized_data,axis=0)
    normalized_data = (normalized_data - min_val) / (max_val - min_val)
    return normalized_data

def read_vrptw_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables to store the extracted data
    capacity = None

    # Flags to determine which part of the file we are currently processing
    vehicle_section = False
    customer_section = False

    customer = {
        'cust_no': [],
        'xycoord': [],
        'demand': [],
        'ready_time': [],
        'due_date': [],
        'service_time': []
    }

    for line in lines:
        # Remove any extraneous whitespace
        line = line.strip()

        if 'VEHICLE' in line:
            vehicle_section = True
            customer_section = False
            continue

        if 'CUSTOMER' in line:
            customer_section = True
            vehicle_section = False
            continue

        if vehicle_section:
            if 'CAPACITY' in line:
                continue  # Skip the header
            parts = line.split()
            if len(parts) >= 2:
                capacity = int(parts[1])  # Assuming capacity is the second element

        if customer_section:
            if 'CUST NO.' in line:
                continue  # Skip the header
            parts = line.split()
            if len(parts) >= 7:
                customer['xycoord'].append([int(parts[1]),int(parts[2])])
                customer['demand'].append(int(parts[3]))
                customer['ready_time'].append(int(parts[4]))
                customer['due_date'].append(int(parts[5]))
                customer['service_time'].append(int(parts[6]))

    # 对坐标和需求进行归一化
    depot_xy = normalize(np.array(customer['xycoord']))[0,:].reshape(1,2)
    node_xy = normalize(np.array(customer['xycoord']))[1:,:].reshape(-1,2)

    # depot_xy = np.array(customer['xycoord'])[0, :].reshape(1, 2)
    # node_xy = np.array(customer['xycoord'])[1:, :].reshape(-1, 2)

    node_demand = np.array([item/capacity for item in customer['demand']][1:]).reshape(-1,1)
    node_serviceTime = np.array(customer['service_time'][1:]).reshape(-1,1)
    node_earlyTW = np.array(customer['ready_time'][1:]).reshape(-1,1)
    node_lateTW = np.array(customer['due_date'][1:]).reshape(-1,1)

    data = np.concatenate((node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW),axis=1)

    path = r'D:\Paper Data\paper 3\instance\VRPTW_Benchmark'
    for file in [f for f in os.listdir(path) if f.endswith('.txt')]:
        file_path = path + f'/{file}'
        print(file_path)
        with open(path + f"/{file.split('.')[0]}_real.pkl", 'wb') as f:
            pickle.dump([depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW],f)
        print('修改完毕!')

    return depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW


# Example usage
file_path = r"D:\Paper Data\paper 3\instance\VRPTW_Benchmark\C101.txt"
depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW = read_vrptw_instance(file_path)




