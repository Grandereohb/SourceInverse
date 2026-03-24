#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random
import re
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Step 1: Load site data (station coordinates)
site_file_path = r"C:\Document\博士\高老师项目\SourceInverse\data_sumitomo\sites.xlsx"
site_data = pd.read_excel(site_file_path)


# Convert DMS (degrees, minutes, seconds) to decimal degrees
def manual_dms_to_decimal(dms_str):
    degrees, minutes, seconds = re.split("[°'\"]", dms_str)[:3]
    return float(degrees) + float(minutes) / 60 + float(seconds) / 3600


# Process the site coordinates
lon_values = ["120°56'38.23\"", "120°56'42.87\"", "120°56'58.88\""]
lat_values = ["31°45'5.47\"", "31°44'51.77\"", "31°45'0.69\""]
site_lon_decimals = [manual_dms_to_decimal(lon) for lon in lon_values]
site_lat_decimals = [manual_dms_to_decimal(lat) for lat in lat_values]

site_coordinates_cleaned = pd.DataFrame(
    {
        "Station": ["N", "S", "E"],
        "Longitude": site_lon_decimals,
        "Latitude": site_lat_decimals,
    }
)

# Step 2: Convert the site coordinates to the research area (X, Y coordinates)
boundary_coords = {
    "top_left": (120.9418, 31.7520),
    "top_right": (120.9500, 31.7520),
    "bottom_left": (120.9418, 31.7478),
    "bottom_right": (120.9500, 31.7478),
}

lat_range = boundary_coords["top_left"][1] - boundary_coords["bottom_left"][1]
lon_range = boundary_coords["top_right"][0] - boundary_coords["top_left"][0]

site_coordinates_cleaned["X"] = (
    (site_coordinates_cleaned["Longitude"] - boundary_coords["top_left"][0])
    / lon_range
    * 1000
)
site_coordinates_cleaned["Y"] = (
    (site_coordinates_cleaned["Latitude"] - boundary_coords["bottom_left"][1])
    / lat_range
    * 1000
)

# Step 3: Load pollution monitoring data for the three stations
monitoring_data_file_path = (
    r"C:\Document\博士\高老师项目\SourceInverse\data_sumitomo\厂界\test3\3.xlsx"
)
monitoring_data = pd.read_excel(monitoring_data_file_path)
monitoring_data.columns = ["时间", "N", "S", "E"]

# Step 4: Load wind data
wind_file_path = (
    r"C:\Document\博士\高老师项目\SourceInverse\data_sumitomo\厂界\test3\wind.xlsx"
)
wind_data = pd.read_excel(wind_file_path)
# Assuming wind_data has columns ['Hour', 'Wind_Direction', 'Wind_Speed']


# Step 5: Gaussian plume model to calculate pollution concentration
def gaussian_plume(x, y, Q, u, sigma_y, sigma_z, H=0):
    C = (
        (Q / (2 * np.pi * u * sigma_y * sigma_z))
        * np.exp(-(y**2) / (2 * sigma_y**2))
        * np.exp(-(H**2) / (2 * sigma_z**2))
    )
    return C


def calculate_concentration(x_source, y_source, u, wind_dir, station_x, station_y):
    wind_dir_rad = np.deg2rad(wind_dir)
    x_downwind = (station_x - x_source) * np.cos(wind_dir_rad) + (
        station_y - y_source
    ) * np.sin(wind_dir_rad)
    y_crosswind = -(station_x - x_source) * np.sin(wind_dir_rad) + (
        station_y - y_source
    ) * np.cos(wind_dir_rad)
    return gaussian_plume(x_downwind, y_crosswind, Q=1.0, u=u, sigma_y=50, sigma_z=50)


# Step 6: Sparrow Search Algorithm to find best pollution source location
population_size = 30
max_iter = 100
area_bounds = [0, 1000]


def initialize_population(pop_size, bounds):
    population = []
    for _ in range(pop_size):
        x_pos = random.uniform(bounds[0], bounds[1])
        y_pos = random.uniform(bounds[0], bounds[1])
        population.append((x_pos, y_pos))
    return population


def fitness_function(
    source_pos, wind_data, site_coordinates_cleaned, observed_concentrations, Q=1.0
):
    total_error = 0
    for i in range(len(wind_data)):
        for idx, row in site_coordinates_cleaned.iterrows():
            predicted_concentration = calculate_concentration(
                source_pos[0],
                source_pos[1],
                wind_data.iloc[i]["sp"],
                wind_data.iloc[i]["dir"],
                row["X"],
                row["Y"],
            )
            observed_concentration = observed_concentrations.iloc[i][row["Station"]]
            total_error += (predicted_concentration - observed_concentration) ** 2
    return total_error / (len(wind_data) * len(site_coordinates_cleaned))


population = initialize_population(population_size, area_bounds)


def ssa_optimize(max_iter, population, fitness_function):
    for iteration in range(max_iter):
        fitness_scores = [
            fitness_function(pos, wind_data, site_coordinates_cleaned, monitoring_data)
            for pos in population
        ]
        best_idx = np.argmin(fitness_scores)
        best_position = population[best_idx]
        new_population = []
        for pos in population:
            x_new = pos[0] + random.uniform(-50, 50)
            y_new = pos[1] + random.uniform(-50, 50)
            x_new = np.clip(x_new, area_bounds[0], area_bounds[1])
            y_new = np.clip(y_new, area_bounds[0], area_bounds[1])
            new_population.append((x_new, y_new))
        population = new_population
    return best_position


best_source_position = ssa_optimize(max_iter, population, fitness_function)


# Step 7: Visualize the results
def xy_to_latlon(x, y, boundary_top_left, boundary_top_right, boundary_bottom_left):
    lon_range = boundary_top_right[0] - boundary_top_left[0]
    lat_range = boundary_top_left[1] - boundary_bottom_left[1]
    lon = x / 1000 * lon_range + boundary_top_left[0]
    lat = y / 1000 * lat_range + boundary_bottom_left[1]
    return lat, lon


(
    site_coordinates_cleaned["Latitude_plot"],
    site_coordinates_cleaned["Longitude_plot"],
) = zip(
    *site_coordinates_cleaned.apply(
        lambda row: xy_to_latlon(
            row["X"],
            row["Y"],
            boundary_coords["top_left"],
            boundary_coords["top_right"],
            boundary_coords["bottom_left"],
        ),
        axis=1,
    )
)

best_source_lat, best_source_lon = xy_to_latlon(
    best_source_position[0],
    best_source_position[1],
    boundary_coords["top_left"],
    boundary_coords["top_right"],
    boundary_coords["bottom_left"],
)

# 打印推论的污染源经纬度坐标
print(
    f"Estimated Pollution Source Location: Latitude = {best_source_lat}, Longitude = {best_source_lon}"
)

plt.figure(figsize=(8, 8))
plt.plot(
    [
        boundary_coords["top_left"][0],
        boundary_coords["top_right"][0],
        boundary_coords["top_right"][0],
        boundary_coords["top_left"][0],
        boundary_coords["top_left"][0],
    ],
    [
        boundary_coords["top_left"][1],
        boundary_coords["top_left"][1],
        boundary_coords["bottom_left"][1],
        boundary_coords["bottom_left"][1],
        boundary_coords["top_left"][1],
    ],
    "k-",
    label="Research Area",
)

for idx, row in site_coordinates_cleaned.iterrows():
    plt.scatter(
        row["Longitude_plot"],
        row["Latitude_plot"],
        c="blue",
        label=f"Station {row['Station']}" if idx == 0 else "",
        s=100,
    )

plt.scatter(
    best_source_lon, best_source_lat, c="red", label="Estimated Pollution Source", s=200
)

# 绘制半径50米的区域
circle = Circle(
    (best_source_lon, best_source_lat),
    radius=50 / 111000,
    color="red",
    fill=False,
    linestyle="--",
    label="Source Area (50m radius)",
)
plt.gca().add_patch(circle)

plt.title("Research Area and Estimated Pollution Source with Latitude/Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
