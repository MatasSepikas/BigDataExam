### Description
The Python script processes AIS (Automatic Identification System) data to identify the closest vessels within a specified area and time window. It calculates the distance between vessels, determines the closest pairs, and visualizes their trajectories 10 minutes before and 10 minutes after the rendezvous time. The date range for this project is from December 1, 2021, to December 31, 2021. The code outputs the MMSI and names of the vessels that were closest in this area, along with their 20-minute trajectory visualization around the rendezvous moment.

### Program

- **Process files**: read and process multiple CSV files containing AIS data.
- **Remove noise**: remove rows with missing values in the selected  MMSI, timestamp, latitude, longitude, and vessel name columns.
- **Filter data**: use the Haversine formula to filter data within a 50 km radius around a central point defined by Latitude: 55.225000, Longitude: 14.245000.
- **Find closest vessels**: identify the closest unique pairs of vessels at different times
- tamps by calculating distances using the Euclidean distance method. The program collects the lowest distances at each timestamp, daily, and overall for all given datasets.
- **Parallel processing**: utilize multiprocessing to process files in parallel for efficiency.
- **Visualization of trajectories**: plot the 20-minute trajectories around the rendezvous moment for the closest vessels. 

### Output
The script outputs the closest vessels were LATTE (MMSI: 219020332) and SILLE BOB (MMSI: 219017554) at 28/12/2021 17:58:28.

![Trajectory Plot](https://github.com/MatasSepikas/BigDataExam/blob/main/trajectory_plot.png))

Trajectory visualization 10 minutes before the rendezvous moment indicates that it's possible the ships collided.
