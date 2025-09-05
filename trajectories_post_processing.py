#%%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import scipy.stats
import cartopy.feature as cfeature

import csat2.MODIS
import seaborn as sns


# %%
# Define bins (1°x1° grid)]
lat_bins = np.arange(-90, 91, 1)
lon_bins = np.arange(0, 360, 1)

lon_grid,lat_grid = np.meshgrid(lon_bins,lat_bins)

path_to_file = '/disk1/Users/gjp23/outputs/traj_positions/global_analysis/trajectories_isccp_optimized_20150101_20160101.nc'

ds = xr.open_dataset(path_to_file)
ds['lon'] = ds['lon'] % 360 ## ensure all lons are in the [0, 360) range


#%%

def create_global_cf_change_grids_enhanced(ds, grid_resolution=1.0):
    """
    Enhanced version that properly calculates and stores initial CF values.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Your trajectory dataset containing isccp_data, lon, lat, start_time
    grid_resolution : float
        Grid resolution in degrees (default: 1.0 for 1° grid)
    
    Returns:
    --------
    cf_change_grids : xarray.Dataset
        Dataset containing gridded delta_day, delta_night, init_day, init_night for each date
    """
    
    # Calculate day and night cloud fraction changes for all trajectories
    isccp_avg_am_one = ds['isccp_data'].isel(step=slice(0, 3)).mean(dim='step', skipna=True)
    lon_avg_am_one = ds['lon'].isel(step=0)
    lat_avg_am_one = ds['lat'].isel(step=0)

    isccp_avg_pm_one = ds['isccp_data'].isel(step=slice(10, 13)).mean(dim='step', skipna=True)
    lon_avg_pm_one = ds['lon'].isel(step=slice(10, 13)).mean(dim='step', skipna=True)
    lat_avg_pm_one = ds['lat'].isel(step=slice(10, 13)).mean(dim='step', skipna=True)

    isccp_avg_am_two = ds['isccp_data'].isel(step=slice(21, 24)).mean(dim='step', skipna=True)

    # Calculate changes and initial values
    delta_day = isccp_avg_pm_one - isccp_avg_am_one  # Day change (PM - AM)
    delta_night = isccp_avg_am_two - isccp_avg_pm_one  # Night change (Next AM - PM)
    init_day = isccp_avg_am_one  # Initial day CF (AM)
    init_night = isccp_avg_pm_one  # Initial night CF (PM)
    
    # Get unique dates from start_time
    start_times = pd.to_datetime(ds['start_time'].values)
    unique_dates = pd.to_datetime(start_times.date).unique()
    
    # Define global grid
    lon_grid = np.arange(0.5, 360, grid_resolution)  # Match MODIS grid
    lat_grid = np.arange(89.5, -90, -grid_resolution)   # Match MODIS grid
    
    # Initialize results dictionary
    daily_grids = {}
    
    for current_date in unique_dates:
        print(f"Processing date: {current_date.date()}")
        
        # Get trajectories for this date
        date_mask = pd.to_datetime(start_times.date) == current_date
        if not np.any(date_mask):
            continue
            
        # Get trajectory data for this date
        traj_lons = lon_avg_am_one.values[date_mask]  # Use AM coordinates
        traj_lats = lat_avg_am_one.values[date_mask]
        traj_delta_day = delta_day.values[date_mask]
        traj_delta_night = delta_night.values[date_mask]
        traj_init_day = init_day.values[date_mask]
        traj_init_night = init_night.values[date_mask]

        # Remove NaN trajectories
        valid_mask = ~(np.isnan(traj_lons) | np.isnan(traj_lats) | 
                      np.isnan(traj_delta_day) | np.isnan(traj_delta_night) |
                      np.isnan(traj_init_day) | np.isnan(traj_init_night))
        
        if not np.any(valid_mask):
            continue
            
        traj_lons = traj_lons[valid_mask]
        traj_lats = traj_lats[valid_mask]
        traj_delta_day = traj_delta_day[valid_mask]
        traj_delta_night = traj_delta_night[valid_mask]
        traj_init_day = traj_init_day[valid_mask]
        traj_init_night = traj_init_night[valid_mask]

        # Grid the trajectory data (enhanced version)
        grids = grid_trajectory_data_enhanced(
            traj_lons, traj_lats, traj_delta_day, traj_delta_night,
            traj_init_day, traj_init_night, lon_grid, lat_grid
        )
        
        # Store results
        daily_grids[current_date.date()] = grids
    
    # Convert to xarray Dataset
    cf_change_dataset = create_cf_change_dataset_enhanced(daily_grids, lon_grid, lat_grid)
    
    return cf_change_dataset


def grid_trajectory_data_enhanced(lons, lats, delta_day, delta_night, 
                                init_day, init_night, lon_grid, lat_grid):
    """
    Enhanced version that grids both changes and initial values.
    """
    
    # Initialize output grids
    delta_day_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    delta_night_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    init_day_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    init_night_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)
    
    # Count grid for averaging multiple trajectories in same grid cell
    count_grid = np.zeros((len(lat_grid), len(lon_grid)), dtype=int)
    sum_day_grid = np.zeros((len(lat_grid), len(lon_grid)))
    sum_night_grid = np.zeros((len(lat_grid), len(lon_grid)))
    sum_init_day_grid = np.zeros((len(lat_grid), len(lon_grid)))
    sum_init_night_grid = np.zeros((len(lat_grid), len(lon_grid)))
    
    for i, (lon, lat, dd, dn, id_val, in_val) in enumerate(zip(lons, lats, delta_day, 
                                                               delta_night, init_day, init_night)):
        # Find grid indices
        lon_idx = np.floor(lon).astype(int)
        lat_idx = np.floor(90 - lat).astype(int)
        
        # Check bounds
        if 0 <= lon_idx < len(lon_grid) and 0 <= lat_idx < len(lat_grid):
            sum_day_grid[lat_idx, lon_idx] += dd
            sum_night_grid[lat_idx, lon_idx] += dn
            sum_init_day_grid[lat_idx, lon_idx] += id_val
            sum_init_night_grid[lat_idx, lon_idx] += in_val
            count_grid[lat_idx, lon_idx] += 1
    
    # Calculate averages where we have data
    valid_mask = count_grid > 0
    delta_day_grid[valid_mask] = sum_day_grid[valid_mask] / count_grid[valid_mask]
    delta_night_grid[valid_mask] = sum_night_grid[valid_mask] / count_grid[valid_mask]
    init_day_grid[valid_mask] = sum_init_day_grid[valid_mask] / count_grid[valid_mask]
    init_night_grid[valid_mask] = sum_init_night_grid[valid_mask] / count_grid[valid_mask]
    
    return {
        'delta_day': delta_day_grid,
        'delta_night': delta_night_grid,
        'init_day': init_day_grid,
        'init_night': init_night_grid
    }


def create_cf_change_dataset_enhanced(daily_grids, lon_grid, lat_grid):
    """
    Enhanced version that properly handles all four variables.
    """
    
    dates = sorted(daily_grids.keys())
    
    # Initialize arrays
    delta_day_array = np.full((len(dates), len(lat_grid), len(lon_grid)), np.nan)
    delta_night_array = np.full((len(dates), len(lat_grid), len(lon_grid)), np.nan)
    init_day_array = np.full((len(dates), len(lat_grid), len(lon_grid)), np.nan)
    init_night_array = np.full((len(dates), len(lat_grid), len(lon_grid)), np.nan)
    
    # Fill arrays
    for i, date in enumerate(dates):
        delta_day_array[i, :, :] = daily_grids[date]['delta_day']
        delta_night_array[i, :, :] = daily_grids[date]['delta_night']
        init_day_array[i, :, :] = daily_grids[date]['init_day']
        init_night_array[i, :, :] = daily_grids[date]['init_night']
    
    # Create Dataset
    dataset = xr.Dataset({
        'delta_day': (['time', 'lat', 'lon'], delta_day_array),
        'delta_night': (['time', 'lat', 'lon'], delta_night_array),
        'init_day': (['time', 'lat', 'lon'], init_day_array),
        'init_night': (['time', 'lat', 'lon'], init_night_array)
    }, coords={
        'time': pd.to_datetime(dates),
        'lat': lat_grid,
        'lon': lon_grid
    })
    
    # Add attributes
    dataset['delta_day'].attrs = {
        'long_name': 'Day Cloud Fraction Change',
        'units': '%',
        'description': 'PM cloud fraction - AM cloud fraction'
    }
    dataset['delta_night'].attrs = {
        'long_name': 'Night Cloud Fraction Change', 
        'units': '%',
        'description': 'Next AM cloud fraction - PM cloud fraction'
    }
    dataset['init_day'].attrs = {
        'long_name': 'Initial Day Cloud Fraction',
        'units': '%',
        'description': 'AM cloud fraction (initial day value)'
    }
    dataset['init_night'].attrs = {
        'long_name': 'Initial Night Cloud Fraction',
        'units': '%',
        'description': 'PM cloud fraction (initial night value)'
    }
    
    return dataset

#### Nd stuff

def match_cf_changes_to_modis_enhanced(cf_dataset, satellite='terra'):
    """
    Enhanced version that includes initial CF values.
    """
    
    matched_data = []
    
    for time_idx, current_time in enumerate(cf_dataset.time.values):
        current_date = pd.Timestamp(current_time)
        print(f"Processing MODIS data for: {current_date.date()}")
        
        try:
            # Load MODIS data for this date )
            doy = current_date.timetuple().tm_yday 
        
            MODIS_data = csat2.MODIS.readin('cdnc_best', 
                                          year=current_date.year, 
                                          doy=doy, 
                                          sds=['Nd_G18'], 
                                          sat=satellite)

            Nd = MODIS_data.Nd_G18
            MOD_lon = Nd.lon.values
            MOD_lon[MOD_lon < 0] += 360  # Shift negative longitudes to 0-360 range
            Nd = Nd.assign_coords(lon=MOD_lon).sortby("lon")
            
      
                        
            # Get CF data for this day
            delta_day = cf_dataset['delta_day'].isel(time=time_idx)
            delta_night = cf_dataset['delta_night'].isel(time=time_idx)
            init_day = cf_dataset['init_day'].isel(time=time_idx)
            init_night = cf_dataset['init_night'].isel(time=time_idx)
            

            matched_data.append({
                'time': current_time,
                'delta_day': delta_day.values,
                'delta_night': delta_night.values,
                'init_day': init_day.values,  # Added initial CF values
                'init_night': init_night.values,
                'nd': Nd.values,
                'cf_lon': delta_day.lon.values,
                'cf_lat': delta_day.lat.values,
                'nd_lon': Nd.lon.values,
                'nd_lat': Nd.lat.values
            })
            
        except Exception as e:
            print(f"Error loading MODIS data for {current_date.date()}: {e}")
            continue
    
    return matched_data




def create_5degree_histogram_analysis(matched_data, 
                                     cf_bins=np.arange(0, 101, 10), 
                                     nd_bins=np.logspace(1, 3, 50),
                                     delta_cf_bins=np.arange(-50, 51, 5)):
    """
    Create histogram analysis on 5-degree spatial bins.
    
    Parameters:
    -----------
    matched_data : list
        List of dictionaries from match_cf_changes_to_modis
    cf_bins : array
        Bins for initial cloud fraction (%)
    nd_bins : array  
        Bins for droplet number concentration (cm-3)
    delta_cf_bins : array
        Bins for cloud fraction changes (%)
        
    Returns:
    --------
    histogram_results : dict
        Dictionary with spatial bins as keys, containing histograms
    """
    
    # Define 5-degree spatial bins
    lon_5deg_bins = np.arange(0, 360, 5)
    lat_5deg_bins = np.arange(-90, 91, 5)
    
    # Initialize results dictionary
    histogram_results = {}
    
    # Process each day's data
    for day_data in matched_data:
        print(f"Processing histogram for {day_data['time']}")
        
        # Get data arrays
        delta_day = day_data['delta_day']
        delta_night = day_data['delta_night']
        nd = day_data['nd'][0].T ## transpose so that array is lat lon, consistent with the cf arrays, maye swap tis in the future?
        cf_lons = day_data['cf_lon']
        cf_lats = day_data['cf_lat']
        
        # Get initial CF values
        init_day = day_data.get('init_day', None)
        init_night = day_data.get('init_night', None)
        
        if init_day is None or init_night is None:
            print("Warning: Initial CF values not found in matched_data")
            continue
        
        # Process each 1-degree grid point
        for i in range(len(cf_lats)):
            for j in range(len(cf_lons)):
                lat = cf_lats[i]
                lon = cf_lons[j]
                
                # Find which 5-degree bin this point belongs to
                lat_5deg_idx = int((lat + 90) // 5)
                lon_5deg_idx = int(lon // 5)
                
                # Skip if out of bounds
                if not (0 <= lat_5deg_idx < len(lat_5deg_bins)-1 and 
                       0 <= lon_5deg_idx < len(lon_5deg_bins)-1):
                    continue
                
                # Create spatial bin key
                spatial_key = (lat_5deg_idx, lon_5deg_idx)
                
                # Get values for this grid point - handle potential arrays
                try:
                    dd_val = np.asarray(delta_day[i, j]).item()
                    dn_val = np.asarray(delta_night[i, j]).item()
                    nd_val = np.asarray(nd[i, j]).item()
                    init_day_val = np.asarray(init_day[i, j]).item()
                    init_night_val = np.asarray(init_night[i, j]).item()
                except Exception as e:
                    # Debug: print shapes to understand the issue
                    print(f"Error at position ({i}, {j}):")
                    print(f"  delta_day shape: {np.asarray(delta_day[i, j]).shape}")
                    print(f"  delta_night shape: {np.asarray(delta_night[i, j]).shape}")
                    print(f"  nd shape: {np.asarray(nd[i, j]).shape}")
                    print(f"  init_day shape: {np.asarray(init_day[i, j]).shape}")
                    print(f"  init_night shape: {np.asarray(init_night[i, j]).shape}")
                    print(f"  lat: {lat}, lon: {lon}")
                    
                    # Skip this grid point if we can't convert to scalar
                    continue
                
                # Skip if any values are NaN - now checking each individually
                if (np.isnan(dd_val) or np.isnan(dn_val) or np.isnan(nd_val) or 
                    np.isnan(init_day_val) or np.isnan(init_night_val)):
                    continue
                
                # Initialize spatial bin if not exists
                if spatial_key not in histogram_results:
                    histogram_results[spatial_key] = {
                        'day_data': [],
                        'night_data': [],
                        'lat_center': lat_5deg_bins[lat_5deg_idx] + 2.5,
                        'lon_center': lon_5deg_bins[lon_5deg_idx] + 2.5
                    }
                
                # Store data points
                histogram_results[spatial_key]['day_data'].append({
                    'delta_cf': dd_val,
                    'init_cf': init_day_val,
                    'nd': nd_val
                })
                
                histogram_results[spatial_key]['night_data'].append({
                    'delta_cf': dn_val,
                    'init_cf': init_night_val,
                    'nd': nd_val
                })
    
    # Create histograms for each spatial bin
    processed_results = {}
    
    for spatial_key, data in histogram_results.items():
        if len(data['day_data']) < 10:  # Skip bins with too few data points
            continue
            
        processed_results[spatial_key] = create_2d_histograms(
            data, cf_bins, nd_bins
        )
        processed_results[spatial_key]['lat_center'] = data['lat_center']
        processed_results[spatial_key]['lon_center'] = data['lon_center']
        processed_results[spatial_key]['n_days'] = len(data['day_data'])
    
    return processed_results



def create_2d_histograms(bin_data, cf_bins, nd_bins):
    """
    Create 2D histograms: Change in CF binned by (initial CF, Nd).
    Also calculate adjusted change in CF.
    """
    
    results = {'day': {}, 'night': {}}
    
    for period in ['day', 'night']:
        data_points = bin_data[f'{period}_data']
        
        if len(data_points) == 0:
            continue
            
        # Convert to arrays
        delta_cf_arr = np.array([d['delta_cf'] for d in data_points])
        init_cf_arr = np.array([d['init_cf'] for d in data_points])
        nd_arr = np.array([d['nd'] for d in data_points])
        
        # Calculate mean delta CF for each initial CF bin (for adjustment)
        mean_delta_by_init_cf = {}
        for cf_idx in range(len(cf_bins)-1):
            cf_mask = (init_cf_arr >= cf_bins[cf_idx]) & (init_cf_arr < cf_bins[cf_idx+1])
            if np.sum(cf_mask) > 0:
                mean_delta_by_init_cf[cf_idx] = np.mean(delta_cf_arr[cf_mask])
            else:
                mean_delta_by_init_cf[cf_idx] = 0.0
        
        # Calculate adjusted delta CF
        adjusted_delta_cf_arr = np.zeros_like(delta_cf_arr)
        for i, init_cf_val in enumerate(init_cf_arr):
            # Find which initial CF bin this point belongs to
            cf_bin_idx = np.digitize(init_cf_val, cf_bins) - 1
            cf_bin_idx = np.clip(cf_bin_idx, 0, len(cf_bins)-2)  # Ensure valid index
            
            # Subtract mean for this initial CF bin
            adjusted_delta_cf_arr[i] = delta_cf_arr[i] - mean_delta_by_init_cf[cf_bin_idx]
        
        # Create the main 2D histogram: initial CF vs Nd, with delta CF as values
        # We'll create a weighted histogram where the weights are the delta CF values
        hist_sum, cf_edges, nd_edges = np.histogram2d(
            init_cf_arr, nd_arr,
            bins=[cf_bins, nd_bins],
            weights=delta_cf_arr
        )
        
        hist_count, _, _ = np.histogram2d(
            init_cf_arr, nd_arr,
            bins=[cf_bins, nd_bins]
        )
        
        # Calculate mean delta CF in each bin
        hist_mean = np.divide(hist_sum, hist_count, 
                             out=np.full_like(hist_sum, np.nan), 
                             where=(hist_count > 0))
        
        # Same for adjusted delta CF
        hist_adj_sum, _, _ = np.histogram2d(
            init_cf_arr, nd_arr,
            bins=[cf_bins, nd_bins],
            weights=adjusted_delta_cf_arr
        )
        
        hist_adj_mean = np.divide(hist_adj_sum, hist_count,
                                 out=np.full_like(hist_adj_sum, np.nan),
                                 where=(hist_count > 0))
        
        # Calculate standard deviation in each bin
        hist_std = np.full_like(hist_mean, np.nan)
        hist_adj_std = np.full_like(hist_adj_mean, np.nan)
        
        for cf_idx in range(len(cf_bins)-1):
            for nd_idx in range(len(nd_bins)-1):
                mask = ((init_cf_arr >= cf_bins[cf_idx]) & (init_cf_arr < cf_bins[cf_idx+1]) &
                       (nd_arr >= nd_bins[nd_idx]) & (nd_arr < nd_bins[nd_idx+1]))
                
                if np.sum(mask) > 1:
                    hist_std[cf_idx, nd_idx] = np.std(delta_cf_arr[mask])
                    hist_adj_std[cf_idx, nd_idx] = np.std(adjusted_delta_cf_arr[mask])
        
        results[period] = {
            'histogram_mean': hist_mean,  # Mean delta CF in each (init_CF, Nd) bin
            'histogram_adjusted_mean': hist_adj_mean,  # Mean adjusted delta CF
            'histogram_std': hist_std,  # Standard deviation of delta CF
            'histogram_adjusted_std': hist_adj_std,  # Standard deviation of adjusted delta CF
            'histogram_count': hist_count,  # Number of points in each bin
            'cf_edges': cf_edges,
            'nd_edges': nd_edges,
            'n_points': len(data_points),
            'mean_delta_by_init_cf': mean_delta_by_init_cf  # For reference
        }
    
    return results

def calculate_nd_regression_analysis(histogram_results, min_points_per_bin=5, min_nd_bins=3):
    """
    Calculate mean adjusted CF change for each Nd bin and perform linear regression.
    
    This function:
    1. For each spatial location and each Nd bin, calculates the mean adjusted CF change
    2. Performs linear regression: mean_adjusted_CF_change = slope * log10(Nd) + intercept
    3. Returns slope (sensitivity), R², p-value for each spatial location
    
    Parameters:
    -----------
    histogram_results : dict
        Results from create_5degree_histogram_analysis
    min_points_per_bin : int
        Minimum points required per Nd bin
    min_nd_bins : int
        Minimum number of Nd bins required for regression
        
    Returns:
    --------
    regression_results : dict
        Dictionary with spatial keys containing regression analysis
    """
    
    regression_results = {}
    
    for spatial_key, data in histogram_results.items():
        lat_center = data['lat_center']
        lon_center = data['lon_center']
        
        regression_results[spatial_key] = {
            'lat': lat_center, 
            'lon': lon_center,
            'day': {}, 
            'night': {}
        }
        
        for period in ['day', 'night']:
            if period not in data:
                continue
                
            period_data = data[period]
            
            # Get bin centers
            nd_centers = (period_data['nd_edges'][:-1] + period_data['nd_edges'][1:]) / 2
            
            # *** CORE CALCULATION: Mean adjusted CF change for each Nd bin ***
            # For each Nd bin, calculate the mean of adjusted CF changes across all initial CF bins
            
            nd_means = []
            adj_cf_means = []
            nd_bin_counts = []
            
            for nd_idx in range(len(nd_centers)):
                # Get all adjusted CF values for this Nd bin across initial CF bins
                nd_slice = period_data['histogram_adjusted_mean'][:, nd_idx]  # Shape: (n_cf_bins,)
                count_slice = period_data['histogram_count'][:, nd_idx]       # Shape: (n_cf_bins,)
                
                # Only include bins with sufficient data
                valid_mask = (count_slice >= min_points_per_bin) & (~np.isnan(nd_slice))
                
                if np.sum(valid_mask) > 0:
                    # Calculate weighted mean adjusted CF change for this Nd bin
                    # Weight by the number of data points in each (init_CF, Nd) bin
                    weights = count_slice[valid_mask]
                    values = nd_slice[valid_mask]
                    
                    if np.sum(weights) > 0:
                        # This is the mean adjusted change in CF for this Nd bin
                        weighted_mean_adj_cf = np.average(values, weights=weights)
                        
                        nd_means.append(nd_centers[nd_idx])
                        adj_cf_means.append(weighted_mean_adj_cf)
                        nd_bin_counts.append(np.sum(weights))
            
            # Now nd_means contains the Nd bin centers
            # adj_cf_means contains the mean adjusted CF change for each Nd bin
            # This is exactly what you asked for!
            
            # *** REGRESSION ANALYSIS ***
            # Perform linear regression if we have enough data points
            # This performs: mean_adjusted_CF_change = slope * log10(Nd) + intercept
            if len(nd_means) >= min_nd_bins:
                nd_means = np.array(nd_means)
                adj_cf_means = np.array(adj_cf_means)  # These are the mean adjusted CF changes per Nd bin
                nd_bin_counts = np.array(nd_bin_counts)
                
                # Use log(Nd) for regression (more physically meaningful)
                log_nd = np.log10(nd_means)
                
                try:
                    # Perform weighted linear regression of mean_adjusted_CF_change vs log10(Nd)
                    weights = np.sqrt(nd_bin_counts)  # Weight by sqrt of total count in each Nd bin
                    
                    # Manual weighted least squares calculation
                    W = np.diag(weights)
                    X = np.column_stack([np.ones(len(log_nd)), log_nd])  # [1, log10(Nd)]
                    
                    # Solve: (X'WX)^(-1) X'Wy where y = mean_adjusted_CF_change
                    XtWX = X.T @ W @ X
                    XtWy = X.T @ W @ adj_cf_means  # adj_cf_means are our y values
                    
                    coeffs = np.linalg.solve(XtWX, XtWy)
                    intercept, slope = coeffs
                    
                    # The slope tells us: change in adjusted CF per log10(Nd)
                    # This is the key result you want to map globally!
                    
                    # Calculate R-squared and other statistics
                    y_pred = intercept + slope * log_nd
                    ss_res = np.sum(weights * (adj_cf_means - y_pred)**2)
                    ss_tot = np.sum(weights * (adj_cf_means - np.average(adj_cf_means, weights=weights))**2)
                    
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Calculate standard error of slope
                    residuals = adj_cf_means - y_pred
                    mse = np.sum(weights * residuals**2) / (len(residuals) - 2)
                    slope_se = np.sqrt(mse * np.linalg.inv(XtWX)[1, 1])
                    
                    # Calculate p-value (approximate)
                    t_stat = slope / slope_se if slope_se > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(residuals) - 2))
                    
                    regression_results[spatial_key][period] = {
                        'slope': slope,  # Change in adjusted ΔCF per log10(Nd) - KEY RESULT!
                        'intercept': intercept,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'slope_se': slope_se,
                        'n_nd_bins': len(nd_means),
                        'total_points': np.sum(nd_bin_counts),
                        'nd_range': (np.min(nd_means), np.max(nd_means)),
                        'nd_means': nd_means,
                        'adj_cf_means': adj_cf_means,
                        'nd_bin_counts': nd_bin_counts
                    }
                    
                except Exception as e:
                    print(f"Regression failed for {spatial_key}, {period}: {e}")
                    continue
    
    return regression_results


def create_global_regression_maps(regression_results, variable='slope', significance_level=0.05):
    """
    Create global maps of regression results.
    
    Parameters:
    -----------
    regression_results : dict
        Results from calculate_nd_regression_analysis
    variable : str
        Variable to plot ('slope', 'r_squared', 'p_value')
    significance_level : float
        Significance level for masking non-significant results
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    
    # Extract spatial coordinates and values
    lats, lons = [], []
    day_values, night_values = [], []
    day_significant, night_significant = [], []
    
    for spatial_key, data in regression_results.items():
        lats.append(data['lat'])
        lons.append(data['lon'])
        
        # Day values
        if 'day' in data and len(data['day']) > 0:
            day_values.append(data['day'][variable])
            day_significant.append(data['day']['p_value'] < significance_level)
        else:
            day_values.append(np.nan)
            day_significant.append(False)
            
        # Night values  
        if 'night' in data and len(data['night']) > 0:
            night_values.append(data['night'][variable])
            night_significant.append(data['night']['p_value'] < significance_level)
        else:
            night_values.append(np.nan)
            night_significant.append(False)
    
    # Convert to arrays
    lats = np.array(lats)
    lons = np.array(lons)
    day_values = np.array(day_values)
    night_values = np.array(night_values)
    day_significant = np.array(day_significant)
    night_significant = np.array(night_significant)
    
    # Try to use Cartopy if available, otherwise use regular matplotlib
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
        projection = ccrs.PlateCarree()
    except ImportError:
        use_cartopy = False
        projection = None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), 
                           subplot_kw={'projection': projection} if use_cartopy else None)
    
    # Set up color scale based on variable
    if variable == 'slope':
        vmin, vmax = -10, 10
        cmap = 'RdBu_r'
        label = 'Slope (Δ Adj CF / log₁₀(Nd))'
        title_suffix = 'Adjusted CF Change Sensitivity to Nd'
    elif variable == 'r_squared':
        vmin, vmax = 0, 1
        cmap = 'viridis'
        label = 'R²'
        title_suffix = 'Regression R²'
    elif variable == 'p_value':
        vmin, vmax = 0, 0.1
        cmap = 'viridis_r'
        label = 'P-value'
        title_suffix = 'Regression P-value'
    else:
        vmin, vmax = np.nanpercentile(np.concatenate([day_values, night_values]), [5, 95])
        cmap = 'viridis'
        label = variable
        title_suffix = variable
    
    # Plot day
    ax = axes[0]
    
    # Only plot significant results if requested
    if variable != 'p_value':
        plot_day_values = np.where(day_significant, day_values, np.nan)
    else:
        plot_day_values = day_values
    
    scatter = ax.scatter(lons, lats, c=plot_day_values, s=50, 
                        cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7,
                        transform=projection if use_cartopy else None)
    
    if use_cartopy:
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'Day - {title_suffix}')
    
    # Plot night
    ax = axes[1]
    
    if variable != 'p_value':
        plot_night_values = np.where(night_significant, night_values, np.nan)
    else:
        plot_night_values = night_values
    
    scatter = ax.scatter(lons, lats, c=plot_night_values, s=50,
                        cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7,
                        transform=projection if use_cartopy else None)
    
    if use_cartopy:
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'Night - {title_suffix}')
    
    fig.subplots_adjust(bottom=0.15)  # leave space for colorbar

    # Create a ScalarMappable for consistent color scale
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # needed for colorbar

    # Place colorbar centered below plots
    cbar_ax = fig.add_axes([0.2, 0.07, 0.6, 0.025])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label, fontsize=12)

    
    return fig, axes


def plot_nd_regression_examples(regression_results, n_examples=6, min_r_squared=0.3):
    """
    Plot examples of Nd regression for selected spatial bins.
    """
    
    # Find good examples with high R-squared
    good_examples = []
    for spatial_key, data in regression_results.items():
        for period in ['day', 'night']:
            if (period in data and len(data[period]) > 0 and 
                data[period]['r_squared'] > min_r_squared):
                
                good_examples.append({
                    'spatial_key': spatial_key,
                    'period': period,
                    'r_squared': data[period]['r_squared'],
                    'slope': data[period]['slope'],
                    'lat': data['lat'],
                    'lon': data['lon']
                })
    
    # Sort by R-squared and take best examples
    good_examples.sort(key=lambda x: x['r_squared'], reverse=True)
    good_examples = good_examples[:n_examples]
    
    # Create plots
    n_cols = 3
    n_rows = int(np.ceil(len(good_examples) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
    
    for i, example in enumerate(good_examples):
        if i >= len(axes):
            break
            
        ax = axes[i]
        spatial_key = example['spatial_key']
        period = example['period']
        
        data = regression_results[spatial_key][period]
        
        # Plot data points
        log_nd = np.log10(data['nd_means'])
        adj_cf = data['adj_cf_means']
        
        # Size points by count
        sizes = 20 + 80 * data['nd_bin_counts'] / np.max(data['nd_bin_counts'])
        
        ax.scatter(data['nd_means'], adj_cf, s=sizes, alpha=0.7, color='blue')
        
        # Plot regression line
        nd_range = np.logspace(np.log10(np.min(data['nd_means'])), 
                              np.log10(np.max(data['nd_means'])), 100)
        log_nd_range = np.log10(nd_range)
        y_pred = data['intercept'] + data['slope'] * log_nd_range
        
        ax.plot(nd_range, y_pred, 'r-', alpha=0.8, linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Droplet Number Concentration (cm⁻³)')
        ax.set_ylabel('Mean Adjusted ΔCF (%)')
        ax.set_title(f'{period.capitalize()}: {example["lat"]:.1f}°N, {example["lon"]:.1f}°E\n'
                    f'Slope={data["slope"]:.2f}, R²={data["r_squared"]:.3f}, '
                    f'p={data["p_value"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(good_examples), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def plot_histogram_analysis(histogram_results, spatial_key, show_adjusted=True):
    """
    Plot 2D histogram analysis for a specific spatial bin.
    Shows both day and night periods, with option to show adjusted values.
    """
    
    if spatial_key not in histogram_results:
        print(f"Spatial key {spatial_key} not found in results")
        return None, None
    
    data = histogram_results[spatial_key]
    lat_center = data['lat_center']
    lon_center = data['lon_center']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2 if show_adjusted else 1, 
                            figsize=(15 if show_adjusted else 8, 12))
    
    if show_adjusted:
        axes = axes.flatten()
    else:
        axes = [axes[0], axes[1]]
    
    plot_idx = 0
    
    for period in ['day', 'night']:
        if period not in data:
            continue
            
        period_data = data[period]
        
        # Plot regular delta CF
        ax = axes[plot_idx]
        
        # Create meshgrid for plotting
        CF_centers = (period_data['cf_edges'][:-1] + period_data['cf_edges'][1:]) / 2
        ND_centers = (period_data['nd_edges'][:-1] + period_data['nd_edges'][1:]) / 2
        CF_mesh, ND_mesh = np.meshgrid(CF_centers, ND_centers)
        
        # Plot mean delta CF
        im = ax.pcolormesh(CF_mesh, ND_mesh, period_data['histogram_mean'].T, 
                          cmap='RdBu_r', vmin=-20, vmax=20, shading='nearest')
        
        ax.set_xlabel('Initial Cloud Fraction (%)')
        ax.set_ylabel('Droplet Number Concentration (cm⁻³)')
        ax.set_yscale('log')
        ax.set_title(f'{period.capitalize()} - Mean ΔCF\n'
                    f'Lat: {lat_center:.1f}°, Lon: {lon_center:.1f}°')
        
        plt.colorbar(im, ax=ax, label='Mean ΔCF (%)')
        
        # Add contours showing count
        contour_levels = [5, 10, 25, 50, 100]
        ax.contour(CF_mesh, ND_mesh, period_data['histogram_count'].T, 
                  levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
        
        plot_idx += 1
        
        # Plot adjusted delta CF if requested
        if show_adjusted:
            ax = axes[plot_idx]
            
            im = ax.pcolormesh(CF_mesh, ND_mesh, period_data['histogram_adjusted_mean'].T, 
                              cmap='RdBu_r', vmin=-20, vmax=20, shading='nearest')
            
            ax.set_xlabel('Initial Cloud Fraction (%)')
            ax.set_ylabel('Droplet Number Concentration (cm⁻³)')
            ax.set_yscale('log')
            ax.set_title(f'{period.capitalize()} - Adjusted Mean ΔCF\n'
                        f'(ΔCF - mean(ΔCF) for init CF)')
            
            plt.colorbar(im, ax=ax, label='Adjusted Mean ΔCF (%)')
            
            # Add contours showing count
            ax.contour(CF_mesh, ND_mesh, period_data['histogram_count'].T, 
                      levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
            
            plot_idx += 1
    
    plt.tight_layout()
    return fig, axes


def analyze_cf_nd_correlation(histogram_results, min_points=10):
    """
    Analyze correlation between initial CF, Nd, and CF changes across all spatial bins.
    """
    
    all_correlations = {
        'day': {'cf_nd': [], 'cf_delta': [], 'nd_delta': [], 'cf_adj_delta': [], 'nd_adj_delta': []},
        'night': {'cf_nd': [], 'cf_delta': [], 'nd_delta': [], 'cf_adj_delta': [], 'nd_adj_delta': []}
    }
    
    spatial_summaries = {}
    
    for spatial_key, data in histogram_results.items():
        lat_center = data['lat_center']
        lon_center = data['lon_center']
        
        spatial_summaries[spatial_key] = {
            'lat': lat_center, 'lon': lon_center,
            'day': {}, 'night': {}
        }
        
        for period in ['day', 'night']:
            if period not in data:
                continue
                
            period_data = data[period]
            
            # Extract valid bins with sufficient data
            valid_mask = period_data['histogram_count'] >= min_points
            valid_indices = np.where(valid_mask)
            
            if len(valid_indices[0]) == 0:
                continue
            
            # Get bin centers
            cf_centers = (period_data['cf_edges'][:-1] + period_data['cf_edges'][1:]) / 2
            nd_centers = (period_data['nd_edges'][:-1] + period_data['nd_edges'][1:]) / 2
            
            # Extract values for valid bins
            init_cf_vals = cf_centers[valid_indices[0]]
            nd_vals = nd_centers[valid_indices[1]]
            delta_vals = period_data['histogram_mean'][valid_mask]
            adj_delta_vals = period_data['histogram_adjusted_mean'][valid_mask]
            
            if len(init_cf_vals) < 3:  # Need at least 3 points for correlation
                continue
                
            # Calculate correlations
            try:
                corr_cf_nd = np.corrcoef(init_cf_vals, nd_vals)[0, 1]
                corr_cf_delta = np.corrcoef(init_cf_vals, delta_vals)[0, 1]
                corr_nd_delta = np.corrcoef(nd_vals, delta_vals)[0, 1]
                corr_cf_adj_delta = np.corrcoef(init_cf_vals, adj_delta_vals)[0, 1]
                corr_nd_adj_delta = np.corrcoef(nd_vals, adj_delta_vals)[0, 1]
                
                # Store correlations if not NaN
                if not np.isnan(corr_cf_nd):
                    all_correlations[period]['cf_nd'].append(corr_cf_nd)
                if not np.isnan(corr_cf_delta):
                    all_correlations[period]['cf_delta'].append(corr_cf_delta)
                if not np.isnan(corr_nd_delta):
                    all_correlations[period]['nd_delta'].append(corr_nd_delta)
                if not np.isnan(corr_cf_adj_delta):
                    all_correlations[period]['cf_adj_delta'].append(corr_cf_adj_delta)
                if not np.isnan(corr_nd_adj_delta):
                    all_correlations[period]['nd_adj_delta'].append(corr_nd_adj_delta)
                
                # Store in spatial summary
                spatial_summaries[spatial_key][period] = {
                    'n_valid_bins': len(init_cf_vals),
                    'corr_cf_nd': corr_cf_nd,
                    'corr_cf_delta': corr_cf_delta,
                    'corr_nd_delta': corr_nd_delta,
                    'corr_cf_adj_delta': corr_cf_adj_delta,
                    'corr_nd_adj_delta': corr_nd_adj_delta,
                    'mean_delta': np.mean(delta_vals),
                    'mean_adj_delta': np.mean(adj_delta_vals)
                }
                
            except Exception as e:
                print(f"Error calculating correlations for {spatial_key}, {period}: {e}")
                continue
    
    # Calculate global statistics
    global_stats = {}
    for period in ['day', 'night']:
        global_stats[period] = {}
        for corr_type, values in all_correlations[period].items():
            if len(values) > 0:
                global_stats[period][corr_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n_regions': len(values)
                }
    
    return global_stats, spatial_summaries


def print_correlation_summary(global_stats):
    """
    Print a summary of the correlation analysis.
    """
    
    print("=== Global CF-Nd Correlation Analysis ===\n")
    
    for period in ['day', 'night']:
        if period not in global_stats:
            continue
            
        print(f"{period.upper()} PERIOD:")
        print("-" * 40)
        
        corr_names = {
            'cf_nd': 'Initial CF vs Nd',
            'cf_delta': 'Initial CF vs ΔCF',
            'nd_delta': 'Nd vs ΔCF',
            'cf_adj_delta': 'Initial CF vs Adjusted ΔCF',
            'nd_adj_delta': 'Nd vs Adjusted ΔCF'
        }
        
        for corr_type, name in corr_names.items():
            if corr_type in global_stats[period]:
                stats = global_stats[period][corr_type]
                print(f"{name:25}: {stats['mean']:6.3f} ± {stats['std']:5.3f} "
                      f"(median: {stats['median']:6.3f}, n={stats['n_regions']})")
        
        print()
    
    print("Note: Adjusted ΔCF = ΔCF - mean(ΔCF) for each initial CF bin")
    print("This removes the baseline trend with initial CF to isolate Nd effects.")


def summarize_global_regression_results(regression_results):
    """
    Print summary statistics of the global regression analysis.
    """
    
    print("=== Global Nd Regression Analysis Summary ===\n")
    
    for period in ['day', 'night']:
        slopes = []
        r_squares = []
        p_values = []
        n_points = []
        
        for spatial_key, data in regression_results.items():
            if period in data and len(data[period]) > 0:
                reg_data = data[period]
                slopes.append(reg_data['slope'])
                r_squares.append(reg_data['r_squared'])
                p_values.append(reg_data['p_value'])
                n_points.append(reg_data['total_points'])
        
        if len(slopes) == 0:
            continue
            
        slopes = np.array(slopes)
        r_squares = np.array(r_squares)
        p_values = np.array(p_values)
        n_points = np.array(n_points)
        
        # Calculate statistics
        significant_mask = p_values < 0.05
        n_significant = np.sum(significant_mask)
        
        print(f"{period.upper()} PERIOD:")
        print("-" * 40)
        print(f"Total spatial bins: {len(slopes)}")
        print(f"Significant bins (p<0.05): {n_significant} ({100*n_significant/len(slopes):.1f}%)")
        print(f"Mean slope: {np.mean(slopes):.3f} ± {np.std(slopes):.3f}")
        print(f"Mean slope (significant only): {np.mean(slopes[significant_mask]):.3f} ± {np.std(slopes[significant_mask]):.3f}")
        print(f"Mean R²: {np.mean(r_squares):.3f} ± {np.std(r_squares):.3f}")
        print(f"Median p-value: {np.median(p_values):.4f}")
        print(f"Mean data points per bin: {np.mean(n_points):.0f}")
        
        # Positive vs negative slopes
        pos_slopes = slopes[slopes > 0]
        neg_slopes = slopes[slopes < 0]
        pos_sig = slopes[(slopes > 0) & significant_mask]
        neg_sig = slopes[(slopes < 0) & significant_mask]
        
        print(f"Positive slopes: {len(pos_slopes)} ({len(pos_sig)} significant)")
        print(f"Negative slopes: {len(neg_slopes)} ({len(neg_sig)} significant)")
        print()
    
    print("Note: Slope units are (% Adjusted ΔCF) per log₁₀(Nd)")
    print("Positive slopes indicate higher Nd leads to more positive CF changes")
    print("Negative slopes indicate higher Nd leads to more negative CF changes")


def run_complete_analysis_with_regression(ds, satellite='terra'):
    """
    Run the complete analysis pipeline including regression analysis and mapping.
    
    This function performs the full workflow:
    1. Creates CF change grids from trajectory data
    2. Matches with MODIS droplet concentration data  
    3. Bins data into 5-degree spatial bins
    4. Creates 2D histograms of CF changes vs (initial CF, Nd)
    5. Calculates adjusted CF changes (removing initial CF dependence)
    6. For each spatial bin: calculates mean adjusted CF change for each Nd bin
    7. Performs linear regression: mean_adj_CF_change vs log10(Nd) 
    8. Creates global maps of regression slopes (sensitivity to Nd)
    9. Provides comprehensive statistical analysis
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Trajectory dataset containing isccp_data, lon, lat, start_time
    satellite : str
        MODIS satellite to use ('terra' or 'aqua')
    
    Returns:
    --------
    results_package : dict
        Complete results including histograms, regressions, and figures
    """
    
    print("Step 1: Creating enhanced CF change grids...")
    cf_dataset = create_global_cf_change_grids_enhanced(ds)
    
    print("Step 2: Matching with MODIS data...")
    matched_data = match_cf_changes_to_modis_enhanced(cf_dataset, satellite)
    
    print("Step 3: Creating 5-degree histogram analysis...")
    histogram_results = create_5degree_histogram_analysis(matched_data)
    
    print("Step 4: Analyzing correlations...")
    global_stats, spatial_summaries = analyze_cf_nd_correlation(histogram_results)
    
    print("Step 5: Calculating Nd regression analysis...")
    print("        - Computing mean adjusted CF change for each Nd bin")
    print("        - Performing linear regression vs log10(Nd)")
    regression_results = calculate_nd_regression_analysis(histogram_results)
    
    print("Step 6: Creating global regression maps...")
    # Create maps for slope (the key result!)
    fig_slope, axes_slope = create_global_regression_maps(regression_results, 'slope')
    fig_slope.suptitle('Global Map: Adjusted CF Change Sensitivity to Nd', fontsize=16)
    
    # Create maps for R-squared
    fig_r2, axes_r2 = create_global_regression_maps(regression_results, 'r_squared')
    fig_r2.suptitle('Global Map: Regression Quality (R²)', fontsize=16)
    
    print("Step 7: Plotting example regressions...")
    fig_examples, axes_examples = plot_nd_regression_examples(regression_results)
    
    print("Step 8: Summarizing results...")
    print_correlation_summary(global_stats)
    print("\n" + "="*60 + "\n")
    summarize_global_regression_results(regression_results)
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(histogram_results)} spatial bins with sufficient data.")
    print(f"Regression analysis completed for {len(regression_results)} spatial bins.")
    print("\nKey outputs:")
    print("- Regression slope: Change in adjusted CF per log10(Nd) for each spatial location")
    print("- Global maps show spatial patterns of aerosol-cloud sensitivity")
    print("- Positive slopes: Higher Nd → more positive CF changes")
    print("- Negative slopes: Higher Nd → more negative CF changes")
    
    results_package = {
        'histogram_results': histogram_results,
        'global_stats': global_stats,
        'spatial_summaries': spatial_summaries,
        'regression_results': regression_results,  # Contains the slopes for global mapping!
        'figures': {
            'slope_maps': (fig_slope, axes_slope),
            'r2_maps': (fig_r2, axes_r2),
            'examples': (fig_examples, axes_examples)
        }
    }
    
    return results_package


# Usage examples and main execution
def example_usage():
    """
    Example of how to use the complete analysis with regression mapping.
    
    The main workflow is:
    1. Run the complete analysis
    2. Examine global maps  
    3. Look at specific regional examples
    4. Analyze statistical significance
    """
    
    # Run complete analysis (replace 'ds' with your actual dataset)

    results = run_complete_analysis_with_regression(ds)
    
    # Access different components
    histogram_results = results['histogram_results']
    regression_results = results['regression_results']  # This contains the slope maps!
    figures = results['figures']
    
    #Show the slope map (key result!)
    figures['slope_maps'][0].show()
    
    #Plot histogram for a specific spatial bin
    spatial_key = (10, 20)  # Example: 10th latitude bin, 20th longitude bin
    if spatial_key in histogram_results:
        fig, ax = plot_histogram_analysis(histogram_results, spatial_key, show_adjusted=True)
        plt.show()

    # Create custom maps for p-values
        fig_pval, axes_pval = create_global_regression_maps(regression_results, 'p_value')
        plt.show()
    
    # Look at regression results for a specific region
    if spatial_key in regression_results:
        region_data = regression_results[spatial_key]
        print(f"Region at {region_data['lat']:.1f}°, {region_data['lon']:.1f}°:")
        for period in ['day', 'night']:
            if period in region_data and len(region_data[period]) > 0:
                reg_data = region_data[period]
                print(f"  {period}: slope = {reg_data['slope']:.3f} ± {reg_data['slope_se']:.3f}")
                print(f"  {period}: R² = {reg_data['r_squared']:.3f}, p = {reg_data['p_value']:.3f}")
    
    # Find regions with strongest positive/negative sensitivities
    strong_positive = []
    strong_negative = []
    for spatial_key, data in regression_results.items():
        for period in ['day', 'night']:
            if period in data and len(data[period]) > 0:
                reg_data = data[period]
                if reg_data['p_value'] < 0.05:  # Significant only
                    if reg_data['slope'] > 5:
                        strong_positive.append((spatial_key, period, reg_data['slope']))
                    elif reg_data['slope'] < -5:
                        strong_negative.append((spatial_key, period, reg_data['slope']))
    
    print(f"Found {len(strong_positive)} regions with strong positive sensitivity")
    print(f"Found {len(strong_negative)} regions with strong negative sensitivity")
    
    pass


