#%%


from csat2.ISCCP import Granule
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import calendar

#%%

def colocate_trajectories_with_isccp_utc_loop(trajectories_file, collection='isccp-basic', 
                                            product='hgg', varname = 'cldamt_irtypes', dt_step_minutes=30):
    """
    Colocate trajectory data with ISCCP data by looping over UTC times in 3-hourly steps.
    For each UTC time, load ISCCP data once and colocate all active parcels.
    
    Parameters:
    -----------
    trajectories_file : str
        Path to the trajectories netCDF file
    collection : str
        ISCCP collection ('isccp-basic' or 'isccp')
    product : str
        ISCCP product ('hgg', 'hgh', 'hgm')
    varname : str
        Variable name to extract from ISCCP data
    dt_step_minutes : int
        Time step in minutes for trajectory data
        
    Returns:
    --------
    ds_colocated : xarray.Dataset
        Dataset with trajectory positions and colocated ISCCP data
    """
    
    # Load trajectory data
    ds_traj = xr.load_dataset(trajectories_file)
    n_trajectories, n_steps = ds_traj.lon.shape
    dt_step = timedelta(minutes=dt_step_minutes)
    
    # Initialize output array
    isccp_data = np.full((n_trajectories, n_steps), np.nan)
    
    # Determine time range from trajectory data
    start_times = ds_traj.start_time.values
    earliest_start = np.min(start_times)
    latest_start = np.max(start_times)
    
    # Convert to datetime objects
    earliest_start_dt = np.datetime64(earliest_start).astype('datetime64[s]').astype(datetime)
    latest_start_dt = np.datetime64(latest_start).astype('datetime64[s]').astype(datetime)
    
    # Calculate the full time range (latest start + max trajectory length)
    max_trajectory_duration = timedelta(minutes=dt_step_minutes * (n_steps - 1))
    end_time = latest_start_dt + max_trajectory_duration
    
    # Round start time down to nearest 3-hour boundary
    start_hour = (earliest_start_dt.hour // 3) * 3
    utc_start = earliest_start_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    
    # Round end time up to next 3-hour boundary
    end_hour = ((end_time.hour // 3) + 1) * 3
    if end_hour >= 24:
        end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        end_time = end_time.replace(hour=end_hour, minute=0, second=0, microsecond=0)
    
    print(f"Processing ISCCP data from {utc_start} to {end_time}")
    
    # Main loop: iterate over UTC times in 3-hourly steps
    current_utc = utc_start
    utc_step = 0
    
    while current_utc <= end_time:
        print(f"Processing UTC time: {current_utc} (step {utc_step})")
        
        # Convert to day of year for ISCCP granule
        doy = current_utc.timetuple().tm_yday
        
        # Create ISCCP granule
        granule = Granule(current_utc.year, doy, current_utc.hour)
        
        try:
            # Download ISCCP data if necessary
            if not granule.check(collection, product):
                print(f"  Downloading ISCCP data for {current_utc}")
                granule.download(collection, product)
            
            # Find all trajectory points that are active at this UTC time
            active_parcels = find_active_parcels_at_utc(ds_traj, current_utc)
            
            if len(active_parcels) > 0:
                print(f"  Found {len(active_parcels)} active parcels")
                
                # Extract positions
                traj_indices = [p['traj_idx'] for p in active_parcels]
                step_indices = [p['step_idx'] for p in active_parcels]
                lons = np.array([p['lon'] for p in active_parcels])
                lats = np.array([p['lat'] for p in active_parcels])
                
                # Convert longitudes to [0, 360] range for ISCCP
                lons = lons % 360
                
                # Colocate all active parcels at once using the granule's geolocate method
                isccp_values = granule.geolocate(collection, product, varname,lons, lats).isel(cloud_irtype = 0) # select liquid water clouds
                
                # Store results back in the output array
                for i, (traj_idx, step_idx) in enumerate(zip(traj_indices, step_indices)):
                    if hasattr(isccp_values, 'values'):
                        isccp_data[traj_idx, step_idx] = isccp_values.values[i]
                    else:
                        isccp_data[traj_idx, step_idx] = isccp_values[i]
                
                print(f"  Colocated {len(active_parcels)} parcels successfully")
            else:
                print(f"  No active parcels found for this UTC time")
                
        except Exception as e:
            print(f"  Error processing ISCCP data for {current_utc}: {e}")
        
        # Move to next 3-hour step
        current_utc += timedelta(hours=3)
        utc_step += 1
    
    # Create output dataset
    ds_colocated = xr.Dataset(
        data_vars=dict(
            lon=(["trajectory", "step"], ds_traj.lon.values),
            lat=(["trajectory", "step"], ds_traj.lat.values),
            isccp_data=(["trajectory", "step"], isccp_data)
        ),
        coords=dict(
            trajectory=np.arange(n_trajectories),
            step=np.arange(n_steps),
            start_time=("trajectory", ds_traj.start_time.values)
        ),
        attrs=dict(
            isccp_collection=collection,
            isccp_product=product,
            isccp_variable=varname,
            description=f"Trajectory data colocated with ISCCP {varname} data",
            time_step_minutes=dt_step_minutes
        )
    )
    
    # Add variable attributes
    ds_colocated.lon.attrs = {'units': 'degrees_east', 'long_name': 'longitude'}
    ds_colocated.lat.attrs = {'units': 'degrees_north', 'long_name': 'latitude'}
    ds_colocated.isccp_data.attrs = {'long_name': f'ISCCP {varname}', 'source': 'ISCCP'}
    
    return ds_colocated


def find_active_parcels_at_utc(ds_traj, utc_time):
    """
    Find all trajectory points that are active at a given UTC time.
    Active means: parcels initialized within 24 hours of the current UTC time,
    at their current position at that UTC time.
    
    Parameters:
    -----------
    ds_traj : xarray.Dataset
        Trajectory dataset with 'time' variable giving UTC time for each step
    utc_time : datetime
        UTC time to check
        
    Returns:
    --------
    active_parcels : list of dict
        List of active parcel information with keys: 'traj_idx', 'step_idx', 'lon', 'lat'
    """
    
    active_parcels = []
    n_trajectories, n_steps = ds_traj.lon.shape
    
    # Convert utc_time to numpy datetime64 for comparison
    utc_time_np = np.datetime64(utc_time)
    time_24h_ago = utc_time_np - np.timedelta64(24, 'h')
    
    for traj_idx in range(n_trajectories):
        # Check if this trajectory was initialized within the last 24 hours
        start_time = ds_traj.start_time.values[traj_idx]
        
        if start_time >= time_24h_ago and start_time <= utc_time_np:
            # This trajectory is "active" (initialized within last 24h)
            # Now find the step that corresponds to the current UTC time
            
            for step_idx in range(n_steps):
                # Check if this step has valid time and position data
                step_time = ds_traj.time.values[traj_idx, step_idx]
                
                if not np.isnat(step_time):  # Check if time is not NaT (Not a Time)
                    # Check if this step time matches our target UTC time
                    # (allowing for some small tolerance due to discrete time steps)
                    time_diff = abs((step_time - utc_time_np) / np.timedelta64(1, 'm'))  # difference in minutes
                    
                    if time_diff <= 90:  # Within 1.5 hours (for 3-hourly ISCCP matching)
                        lon = ds_traj.lon.values[traj_idx, step_idx]
                        lat = ds_traj.lat.values[traj_idx, step_idx]
                        
                        # Only include if position data is valid (not NaN)
                        if not (np.isnan(lon) or np.isnan(lat)):
                            active_parcels.append({
                                'traj_idx': traj_idx,
                                'step_idx': step_idx,
                                'lon': lon,
                                'lat': lat
                            })
                            break  # Found the right time step for this trajectory
    
    return active_parcels


def analyze_colocated_data(ds_colocated, output_file=None):
    """
    Basic analysis of colocated trajectory-ISCCP data.
    
    Parameters:
    -----------
    ds_colocated : xarray.Dataset
        Output from colocate_trajectories_with_isccp_utc_loop
    output_file : str, optional
        Path to save the colocated dataset
    """
    
    if output_file:
        ds_colocated.to_netcdf(output_file)
        print(f"Saved colocated data to {output_file}")
    
    # Basic statistics
    isccp_data = ds_colocated.isccp_data
    valid_data = isccp_data.values[~np.isnan(isccp_data.values)]
    
    print("\n=== Colocation Summary ===")
    print(f"Total trajectory points: {isccp_data.size}")
    print(f"Valid ISCCP matches: {len(valid_data)}")
    print(f"Match rate: {len(valid_data)/isccp_data.size*100:.1f}%")
    
    if len(valid_data) > 0:
        print(f"ISCCP data range: {valid_data.min():.2f} to {valid_data.max():.2f}")
        print(f"ISCCP data mean: {valid_data.mean():.2f}")
        print(f"ISCCP data std: {valid_data.std():.2f}")
    
    # Show distribution by trajectory step
    print(f"\nValid matches by trajectory step:")
    for step in range(min(10, ds_colocated.dims['step'])):  # Show first 10 steps
        step_data = isccp_data[:, step].values
        valid_count = np.sum(~np.isnan(step_data))
        total_count = np.sum(~np.isnan(ds_colocated.lon[:, step].values))
        if total_count > 0:
            print(f"  Step {step}: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return ds_colocated
#%%

# Example usage
if __name__ == "__main__":
    # Define file paths
    trajectory_file = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis/trajectories_20160101_20160102.nc"
    output_file = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis/trajectories_isccp_colocated_one_days_2016.nc"
    
    ds_colocated = colocate_trajectories_with_isccp_utc_loop(
            trajectories_file=trajectory_file,
            collection='isccp-basic',
            product='hgg',
            varname='cldamt_irtype',  # 
            dt_step_minutes=30
        )
    
    # Analyze and save results
    analyze_colocated_data(ds_colocated, output_file)
    
    # Example: Extract data for trajectories that have good ISCCP coverage
    valid_matches = np.sum(~np.isnan(ds_colocated.isccp_data.values), axis=1)
    good_trajectories = np.where(valid_matches >= 10)[0]  # trajectories with at least 10 matches
    
    print(f"\nFound {len(good_trajectories)} trajectories with >= 10 ISCCP matches")
    if len(good_trajectories) > 0:
        print(f"Example trajectory {good_trajectories[0]}:")
        traj_data = ds_colocated.isel(trajectory=good_trajectories[0])
        valid_points = ~np.isnan(traj_data.isccp_data.values)
        print(f"  Start time: {traj_data.start_time.values}")
        print(f"  ISCCP matches: {np.sum(valid_points)}")
        print(f"  ISCCP values: {traj_data.isccp_data.values[valid_points][:5]}...")  # First 5 values

# %%
