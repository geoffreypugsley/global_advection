import numpy as np
import datetime as dt
import xarray as xr
from advection_functions import advection_funcs
from csat2 import ECMWF
import warnings
import os
from collections import defaultdict

# Initial domain
resolution = 1
lat_domain = np.arange(-60, 60+resolution, resolution)
lon_domain = np.arange(0, 360, resolution)
lon_grid, lat_grid = np.meshgrid(lon_domain, lat_domain)

# Time parameters
t0 = dt.datetime(2015, 1, 1, 0, 0)
tf = dt.datetime(2016, 1, 1, 0, 0)
dt_step = dt.timedelta(minutes=60)
lst_threshold = dt_step.total_seconds() / 3600

steps_total = int((tf - t0) / dt_step)
advect_duration = dt.timedelta(hours=24)
steps_per_traj = int(advect_duration / dt_step)

# Initialize wind data
winddata = ECMWF.ERA5WindData(level="1000hPa", res="1grid", linear_interp="both")

def calculate_lst(time_utc, lon):
    return np.mod(time_utc.hour + time_utc.minute / 60 + lon / 15, 24)

# Pre-allocate storage - estimate max trajectories needed
max_trajectories = len(lat_domain) * len(lon_domain) * steps_total // 24  # rough estimate
trajectory_positions = np.full((max_trajectories, steps_per_traj + 1, 2), np.nan)  # [traj, step, lon/lat]
trajectory_start_times = np.full(max_trajectories, dt.datetime(1900, 1, 1), dtype='datetime64[s]')
trajectory_count = 0

# Use deque for better performance with active parcels
from collections import deque
active_parcels = deque()

# Main loop
for step in range(steps_total):
    current_utc = t0 + step * dt_step
    print(f"Step {step}/{steps_total}: {current_utc}")

    # Initialize new trajectories
    lst_grid = calculate_lst(current_utc, lon_grid)
    init_mask = (lst_grid >= 6 - lst_threshold/2) & (lst_grid < 6 + lst_threshold/2)
    
    if np.any(init_mask):
        init_lons = lon_grid[init_mask].flatten()
        init_lats = lat_grid[init_mask].flatten()
        
        n_new_traj = len(init_lons)
        if trajectory_count + n_new_traj > max_trajectories:
            # Resize if needed
            new_size = max(max_trajectories * 2, trajectory_count + n_new_traj)
            new_positions = np.full((new_size, steps_per_traj + 1, 2), np.nan)
            new_positions[:trajectory_count] = trajectory_positions[:trajectory_count]
            trajectory_positions = new_positions
            
            new_start_times = np.full(new_size, dt.datetime(1900, 1, 1), dtype='datetime64[s]')
            new_start_times[:trajectory_count] = trajectory_start_times[:trajectory_count]
            trajectory_start_times = new_start_times
            max_trajectories = new_size

        # Add new trajectories
        traj_indices = np.arange(trajectory_count, trajectory_count + n_new_traj)
        trajectory_positions[traj_indices, 0, 0] = init_lons  # longitude
        trajectory_positions[traj_indices, 0, 1] = init_lats  # latitude
        trajectory_start_times[traj_indices] = np.datetime64(current_utc)
        
        # Add to active parcels
        for idx in traj_indices:
            active_parcels.append((idx, 0))
        
        trajectory_count += n_new_traj

    # Process active parcels - group by advancement step for efficiency
    adv_step_groups = defaultdict(list)
    new_active_parcels = deque()
    
    # Group parcels by their advancement step
    while active_parcels:
        traj_idx, adv_step = active_parcels.popleft()
        
        if adv_step >= steps_per_traj:
            continue  # Trajectory completed
            
        traj_start = trajectory_start_times[traj_idx].astype('datetime64[s]').astype(dt.datetime)
        expected_time = traj_start + adv_step * dt_step
        
        if expected_time == current_utc:
            adv_step_groups[adv_step].append(traj_idx)
        else:
            new_active_parcels.append((traj_idx, adv_step))

    # Advect parcels for each advancement step
    for adv_step, traj_indices in adv_step_groups.items():
        if not traj_indices:
            continue
            
        traj_indices = np.array(traj_indices)
        
        # Get current positions
        current_lons = trajectory_positions[traj_indices, adv_step, 0]
        current_lats = trajectory_positions[traj_indices, adv_step, 1]
        
        # Filter valid positions
        valid_mask = ~np.isnan(current_lons) & ~np.isnan(current_lats)
        if not np.any(valid_mask):
            # All invalid, keep in active list
            for idx in traj_indices:
                new_active_parcels.append((idx, adv_step))
            continue
            
        valid_indices = traj_indices[valid_mask]
        valid_lons = current_lons[valid_mask]
        valid_lats = current_lats[valid_mask]
        
        try:
            # Get wind data
            u_vals, v_vals = winddata.get_data(valid_lons, valid_lats, current_utc)
            
            # Calculate displacement
            dt_seconds = dt_step.total_seconds()
            dist_x = u_vals * dt_seconds / 1000.0
            dist_y = v_vals * dt_seconds / 1000.0
            
            # Calculate new positions
            new_lons, new_lats = advection_funcs.xy_offset_to_ll(valid_lons, valid_lats, dist_x, dist_y)
            
            # Update positions
            trajectory_positions[valid_indices, adv_step + 1, 0] = new_lons
            trajectory_positions[valid_indices, adv_step + 1, 1] = new_lats
            
            # Add to next step
            for idx in valid_indices:
                new_active_parcels.append((idx, adv_step + 1))
                
        except IndexError as e:
            # Download missing data
            ECMWF.download(current_utc.year, current_utc.month, 'V-wind-component', level='1000hPa', resolution='1grid')
            ECMWF.download(current_utc.year, current_utc.month, 'U-wind-component', level='1000hPa', resolution='1grid')
            # Retry or keep in active list
            for idx in traj_indices:
                new_active_parcels.append((idx, adv_step))
                
        except Exception as e:
            warnings.warn(f"Wind data load failed at {current_utc}: {e}")
            # Keep parcels active for retry
            for idx in traj_indices:
                new_active_parcels.append((idx, adv_step))
    
    active_parcels = new_active_parcels

# Trim to actual size and create xarray dataset
trajectory_positions = trajectory_positions[:trajectory_count]
trajectory_start_times = trajectory_start_times[:trajectory_count]

# Create time coordinates
start_times_dt = [dt.datetime.utctimetuple(t.astype('datetime64[s]').astype(dt.datetime)) for t in trajectory_start_times]
times_2d = np.array([
    pd.date_range(start=start, periods=steps_per_traj + 1, freq=dt_step) 
    for start in start_times_dt
])

# Build xarray Dataset
ds = xr.Dataset(
    data_vars=dict(
        lon=(["trajectory", "step"], trajectory_positions[:, :, 0]),
        lat=(["trajectory", "step"], trajectory_positions[:, :, 1]),
        time=(["trajectory", "step"], times_2d)
    ),
    coords=dict(
        trajectory=np.arange(trajectory_count),
        step=np.arange(steps_per_traj + 1),
        start_time=("trajectory", trajectory_start_times)
    )
)

# Save to netCDF
output_dir = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis"
output_path = os.path.join(output_dir, f"trajectories_{t0:%Y%m%d}_{tf:%Y%m%d}.nc")
ds.to_netcdf(output_path)
print(f"Saved {output_path}")