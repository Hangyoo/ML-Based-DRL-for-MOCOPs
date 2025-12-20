# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou 
# @Time: 2024/9/3 11:18

import pandas as pd
import os


# Normalize function
def normalize(df):
    normalized_df = df.copy()
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
    return normalized_df


def read_vrp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists and variables
    coordinates = []
    demands = []
    vehicle_capacity = None
    node_section = False
    demand_section = False
    depot_section = False

    for line in lines:
        line = line.strip()

        if line.startswith('CAPACITY'):
            # Extract vehicle capacity
            vehicle_capacity = int(line.split()[2])
            continue
        elif line.startswith('NODE_COORD_SECTION'):
            node_section = True
            demand_section = False
            continue
        elif line.startswith('DEMAND_SECTION'):
            node_section = False
            demand_section = True
            continue
        elif line.startswith('DEPOT_SECTION'):
            depot_section = True
            break
        if line.startswith('EOF'):
            break

        if node_section and line:
            parts = line.split()
            coordinates.append({
                'x': float(parts[1]),
                'y': float(parts[2])
            })

        if demand_section and line:
            parts = line.split()
            demands.append({
                'demand': int(parts[1])
            })

    # Convert to DataFrames
    coordinates_df = pd.DataFrame(coordinates)
    demands_df = pd.DataFrame(demands)

    print(vehicle_capacity/vehicle_capacity, normalize(coordinates_df), demands_df/vehicle_capacity)
    return vehicle_capacity/vehicle_capacity, normalize(coordinates_df), demands_df/vehicle_capacity   # 点坐标归一化
    # return vehicle_capacity/vehicle_capacity, coordinates_df, demands_df/vehicle_capacity  # 点坐标未归一化


# # Example usage
# file_path = r"C:\Users\Hangyu\Desktop\OVRP\C1-C8\CMT1.vrp"
# vehicle_capacity, coordinates_df, demands_df = read_vrp_file(file_path)
# merged_data = pd.concat([coordinates_df, demands_df], axis=1)
# output_file_path = file_path[:-4] + '.csv'
# # merged_data.to_csv(output_file_path, index=False)

# 批量修改
num = 1
dirpsth = r'D:\Paper Data\paper 3\instance\OVRP_Benchmark\C1-C8'
for file_path in os.listdir(dirpsth):
    file_path = dirpsth + '/' + file_path
    if file_path[-1] == 'p':
        vehicle_capacity, coordinates_df, demands_df = read_vrp_file(file_path)
        merged_data = pd.concat([coordinates_df, demands_df], axis=1)
        # Save to CSV
        # output_file_path = file_path[:-4] + '_real' + '.csv'  # 坐标未归一化
        output_file_path = file_path[:-4] + '.csv'  # 坐标归一化
        merged_data.to_csv(output_file_path, index=False, header=False)
        print(num, file_path, '修改完毕!')
        num += 1




