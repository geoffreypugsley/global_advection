import numpy as np
import datetime as dt
import xarray as xr
from advection_functions import advection_funcs
from csat2 import ECMWF
import warnings
import os

# initial domain
lat_domain = np.arange(-60, 60, 0.25)
lon_domain = np.arange(0, 360, 0.25)
lon_grid, lat_grid = np.meshgrid(lon_domain, lat_domain)

# initial and final times
t0 = dt.datetime(2015, 1, 1, 0, 0)
tf = dt.datetime(2016, 1, 1, 0, 0)
dt_step = dt.timedelta(minutes=30)
steps_total = int((tf - t0) / dt_step)

# Parcel advect duration (24h)
advect_duration = dt.timedelta(hours=24)
steps_per_traj = int(advect_duration / dt_step)

# Initialize wind data 
winddata = ECMWF.ERA5WindData(level="1000hPa", res="1grid", linear_interp="both")

# function to calc lst
def calculate_lst(time_utc, lon):
    return np.mod(time_utc.hour + time_utc.minute / 60 + lon / 15, 24)

# initialize some empty lists
trajectory_start_times = []   # datetime list for trajectory init time
trajectory_ids = []           # same as start times but repeated per step
traj_lons = []                # list of arrays shape (steps_per_traj + 1,)
traj_lats = []                # same as above

# Active parcels tracking: each entry is (traj_index, adv_step_index)
active_parcels = []

# Main loop over all UTC time steps
for step in range(steps_total):
    
    current_utc = t0 + step * dt_step
    print(current_utc)

    # Identify parcels to initialize: only those where LST == ~6am plus/ minus 15 mins at their lon
    lst_grid = calculate_lst(current_utc, lon_grid)
    init_mask = (lst_grid >= 5.75) & (lst_grid < 6.25)

    init_lons = lon_grid[init_mask].flatten()
    init_lats = lat_grid[init_mask].flatten()

    # Initialize new trajectories for these parcels
    for lon_init, lat_init in zip(init_lons, init_lats):
        # Create arrays for lon/lat filled with NaNs initially
        traj_lons.append(np.full(steps_per_traj + 1, np.nan)) # generate empty array of lons
        traj_lats.append(np.full(steps_per_traj + 1, np.nan))

        traj_idx = len(traj_lons) - 1

        # Set initial position at adv_step=0
        traj_lons[traj_idx][0] = lon_init
        traj_lats[traj_idx][0] = lat_init

        # Save start time
        trajectory_start_times.append(current_utc)

        # Mark parcel as active for adv_step=0
        active_parcels.append((traj_idx, 0))

    # Convert to numpy arrays for vectorized operations
    traj_lons_arr = np.array(traj_lons)
    traj_lats_arr = np.array(traj_lats)
    start_times_arr = np.array(trajectory_start_times)

    # Advect all active parcels for their next step if current_utc matches their trajectory start + adv_step*dt_step
    # Group active parcels by adv_step for current UTC step

    # We want to advect parcels at their correct adv_step only if current_utc == start_time + adv_step*dt_step

    # Find parcels that need to advect at current_utc
    new_active_parcels = []
    # We can batch advect parcels that share the same adv_step (= elapsed steps from start)

    # Create dictionary adv_step -> list of parcel indices
    adv_step_groups = {}

    for (traj_idx, adv_step) in active_parcels:
        traj_start = start_times_arr[traj_idx]
        expected_time = traj_start + adv_step * dt_step
        if expected_time == current_utc and adv_step < steps_per_traj:
            adv_step_groups.setdefault(adv_step, []).append(traj_idx)
        else:
            # Parcel not to advect this time
            new_active_parcels.append((traj_idx, adv_step))

    # For each adv_step group at this time, advect parcels together
    for adv_step, traj_indices in adv_step_groups.items():
        # Extract current lon/lat for parcels at adv_step
        current_lons = traj_lons_arr[traj_indices, adv_step]
        current_lats = traj_lats_arr[traj_indices, adv_step]

        # Skip parcels with NaN position
        valid_mask = ~np.isnan(current_lons) & ~np.isnan(current_lats)
        if not np.any(valid_mask):
            # All invalid, skip
            new_active_parcels.extend([(idx, adv_step) for idx in traj_indices])
            continue

        valid_indices = np.array(traj_indices)[valid_mask]
        valid_lons = current_lons[valid_mask]
        valid_lats = current_lats[valid_mask]

        try:
            # Load wind data once for all parcels at this time
            u_vals, v_vals = winddata.get_data(valid_lons, valid_lats, current_utc)
        except IndexError as e:
            #ECMWF.download(current_utc.year, current_utc.timetuple().tm_yday, ['Temperature', 'Relative_humidity', 'U-wind-component'], level='1000hPa', resolution='1grid')
            ECMWF.download(current_utc.year, current_utc.month, 'V-wind-component', level='1000hPa', resolution='1grid')
            ECMWF.download(current_utc.year, current_utc.month, 'U-wind-component', level='1000hPa', resolution='1grid')
            u_vals, v_vals = winddata.get_data(valid_lons, valid_lats, current_utc)
        except Exception as e:
            warnings.warn(f"Wind data load failed at {current_utc}: {e}, skipping adv_step {adv_step}")

            # Keep these parcels active at this adv_step for next try
            new_active_parcels.extend([(idx, adv_step) for idx in traj_indices])
            continue

        dt_seconds = dt_step.total_seconds()

        # Calculate displacement in km
        dist_x = u_vals * dt_seconds / 1000.0
        dist_y = v_vals * dt_seconds / 1000.0

        # Calculate new lon/lat
        new_lons, new_lats = advection_funcs.xy_offset_to_ll(valid_lons, valid_lats, dist_x, dist_y)

        # Update numpy arrays
        traj_lons_arr[valid_indices, adv_step + 1] = new_lons
        traj_lats_arr[valid_indices, adv_step + 1] = new_lats

        # Parcels with invalid positions keep NaNs for next step

        # Mark parcels active for next adv_step
        new_active_parcels.extend([(idx, adv_step + 1) for idx in valid_indices])

    active_parcels = new_active_parcels

    # Update Python lists from numpy arrays for next loop iteration
    traj_lons = list(traj_lons_arr)
    traj_lats = list(traj_lats_arr)

# Finalize arrays
traj_lons_arr = np.array(traj_lons)
traj_lats_arr = np.array(traj_lats)
start_times_arr = np.array(trajectory_start_times)

# Create time coordinates for each step per trajectory
times_2d = np.array([start + np.arange(steps_per_traj + 1) * dt_step for start in start_times_arr])

# Build xarray Dataset
ds = xr.Dataset(
    data_vars=dict(
        lon=(["trajectory", "step"], traj_lons_arr),
        lat=(["trajectory", "step"], traj_lats_arr),
        time=(["trajectory", "step"], times_2d)
    ),
    coords=dict(
        trajectory=np.arange(len(traj_lons_arr)),
        step=np.arange(steps_per_traj + 1),
        start_time=("trajectory", start_times_arr)
    )
)

# Save to netCDF
output_dir = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis"
output_path = os.path.join(output_dir, f"trajectories_{t0:%Y%m%d}_{tf:%Y%m%d}.nc")

# Save to netCDF
ds.to_netcdf(output_path)
print(f"Saved {output_path}")