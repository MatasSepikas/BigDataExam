import pandas as pd
import numpy as np
from scipy.spatial import distance
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


# Calculate the distance between two points in kilometers when longitude and latitude are given
def haversine(latitude1, longitude1, latitude2, longitude2):
    r = 6371.0
    latitude1, longitude1, latitude2, longitude2 = map(np.radians, [latitude1, longitude1, latitude2, longitude2])
    d_latitude = latitude2 - latitude1
    d_longitude = longitude2 - longitude1

    a = np.sin(d_latitude / 2.0) ** 2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(d_longitude / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c


# Use Euclidean distance to find the closest unique vessels at different time points
def find_closest_vessels(df):
    df = df.drop_duplicates(subset=['MMSI'])
    coordinates = df[['Latitude', 'Longitude']].values
    mmsi = df['MMSI'].values
    names = df['Name'].values

    distance_matrix = distance.cdist(coordinates, coordinates, metric='euclidean')
    np.fill_diagonal(distance_matrix, np.inf)
    min_distance_idx = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)

    vessel_1_mmsi = mmsi[min_distance_idx[0]]
    vessel_2_mmsi = mmsi[min_distance_idx[1]]
    vessel_1_name = names[min_distance_idx[0]]
    vessel_2_name = names[min_distance_idx[1]]
    min_distance = distance_matrix[min_distance_idx]

    return vessel_1_mmsi, vessel_2_mmsi, vessel_1_name, vessel_2_name, min_distance


# Find the closest unique vessels at dataset in the area of a 50 km circle area, where the circle center coordinate
# is Latitude: 55.225000, Longitude: 14.245000
def process_file(file_path):
    center_latitude, center_longitude = 55.225000, 14.245000
    radius_km = 50
    data = pd.read_csv(file_path, usecols=['# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
       'Navigational status', 'ROT', 'SOG', 'COG', 'Heading', 'Name'])
    data.dropna(inplace=True)
    distances = haversine(data['Latitude'].values, data['Longitude'].values, center_latitude, center_longitude)
    data_in_circle = data[distances <= radius_km]
    grouped = data_in_circle.groupby('# Timestamp')
    closest_results = []
    lowest_distance = float('inf')
    lowest_distance_info = None

    for timestamp_value, group in grouped:
        if len(group) > 1:
            closest_pair = find_closest_vessels(group)
            if closest_pair:
                closest_results.append((timestamp_value, closest_pair[0], closest_pair[1], closest_pair[2], closest_pair[3], closest_pair[4]))
                if closest_pair[4] < lowest_distance:
                    lowest_distance = closest_pair[4]
                    lowest_distance_info = (timestamp_value, closest_pair[0], closest_pair[1], closest_pair[2], closest_pair[3], closest_pair[4])

    closest_results_df = pd.DataFrame(closest_results, columns=['Timestamp', 'Vessel1_MMSI', 'Vessel2_MMSI', 'Vessel1_Name', 'Vessel2_Name', 'Distance'])

    if lowest_distance_info:
        return lowest_distance, closest_results_df, data_in_circle, lowest_distance_info

    return float('inf'), None, None, None


# 20-minute trajectory visualization around the rendezvous moment for the closest vessels in all datasets
def plot_trajectory(data, mmsi1, mmsi2, rendezvous_time, save_path='trajectory_plot.png'):
    data['# Timestamp'] = pd.to_datetime(data['# Timestamp'])
    rendezvous_time = pd.to_datetime(rendezvous_time)
    time_window = pd.Timedelta(minutes=10)
    start_timestamp = rendezvous_time - time_window
    end_timestamp = rendezvous_time + time_window

    traj1 = data[(data['MMSI'] == mmsi1) &
                 (data['# Timestamp'] >= start_timestamp) &
                 (data['# Timestamp'] <= end_timestamp)]
    traj2 = data[(data['MMSI'] == mmsi2) &
                 (data['# Timestamp'] >= start_timestamp) &
                 (data['# Timestamp'] <= end_timestamp)]

    plt.figure(figsize=(14, 10))
    plt.plot(traj1['Longitude'], traj1['Latitude'], 'b-o', label=f'Vessel {mmsi1}')
    plt.plot(traj2['Longitude'], traj2['Latitude'], 'r-s', label=f'Vessel {mmsi2}')

    plt.plot(traj1['Longitude'].iloc[0], traj1['Latitude'].iloc[0], 'go', label=f'Start vessel {mmsi1}')
    plt.plot(traj1['Longitude'].iloc[-1], traj1['Latitude'].iloc[-1], 'mo', label=f'End vessel {mmsi1}')
    plt.plot(traj2['Longitude'].iloc[0], traj2['Latitude'].iloc[0], 'go', label=f'Start vessel {mmsi2}')
    plt.plot(traj2['Longitude'].iloc[-1], traj2['Latitude'].iloc[-1], 'mo', label=f'End vessel {mmsi2}')

    if not traj1[traj1['# Timestamp'] == rendezvous_time].empty:
        rendezvous_point = traj1.loc[traj1['# Timestamp'] == rendezvous_time, ['Latitude', 'Longitude']].values[0]
        plt.plot(rendezvous_point[1], rendezvous_point[0], 'yo', label='Rendezvous point')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('10-Minute trajectories around rendezvous time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    folder_path = "C:/Users/matas/Desktop/aisdk-2021-12/"
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".csv")]

    overall_lowest_distance = float('inf')
    overall_results_df = None
    overall_data_within_radius = None
    overall_lowest_distance_details = None

    num_processes = max(1, cpu_count() - 2)

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_paths)

    for distance_km, results_df, data_within_radius, lowest_distance_details in results:
        if distance_km < overall_lowest_distance:
            overall_lowest_distance = distance_km
            overall_results_df = results_df
            overall_data_within_radius = data_within_radius
            overall_lowest_distance_details = lowest_distance_details

    if overall_lowest_distance_details:
        timestamp, vessel1_mmsi, vessel2_mmsi, vessel1_name, vessel2_name = overall_lowest_distance_details[:5]
        print(f"Closest vessels were {vessel1_name} (MMSI: {vessel1_mmsi}) and {vessel2_name} (MMSI: {vessel2_mmsi}) at {timestamp}")
        plot_trajectory(overall_data_within_radius, vessel1_mmsi, vessel2_mmsi, timestamp)
