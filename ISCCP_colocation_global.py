#%%
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
from csat2.ISCCP import Granule
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask.array as da
import dask
from functools import partial
import os
import warnings

#%%
class OptimizedTrajectoryColocation:
    """
    parallelized trajectory-ISCCP colocation 
    """
    
    def __init__(self, trajectory_file, chunk_size=100000, n_workers=None, debug=False):

        self.trajectory_file = trajectory_file
        self.chunk_size = chunk_size
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.debug = debug
        
        print(f"Initializing with chunk_size={chunk_size:,}, n_workers={self.n_workers}, debug={self.debug}")
        
        with xr.open_dataset(trajectory_file) as ds:
            self.n_trajectories = len(ds.trajectory)
            self.n_steps = len(ds.step)
            self.start_times = ds.start_time.values
            
        print(f"Dataset: {self.n_trajectories:,} trajectories × {self.n_steps} steps")
        
        self.time_lookup = self._build_time_lookup_table()
        
    def _build_time_lookup_table(self):
        """
        Pre-compute which ISCCP times each trajectory should be active for.
        This eliminates the need for nested loops later.
        """
        print("Building time lookup table...")
        
        # Convert start times to datetime objects
        start_times_dt = pd.to_datetime(self.start_times)
        
        # Determine time range
        earliest = start_times_dt.min()
        latest = start_times_dt.max() + pd.Timedelta(hours=24)  # Add max trajectory duration
        
        # Generate ISCCP times (3-hourly)
        isccp_times = pd.date_range(
            start=earliest.floor('3H'), 
            end=latest.ceil('3H'), 
            freq='3H'
        )
        
        print(f"ISCCP time range: {isccp_times[0]} to {isccp_times[-1]} ({len(isccp_times)} time slots)")
        
        # For each ISCCP time, determine which trajectories are active
        lookup = {}
        
        for i, isccp_time in enumerate(isccp_times):
            time_since_start = (isccp_time - start_times_dt).total_seconds() / 3600  # hours
            active_mask = (time_since_start >= 0) & (time_since_start <= 24)
            active_traj_indices = np.where(active_mask)[0]
            
            if len(active_traj_indices) > 0:
                step_indices = np.round(time_since_start[active_mask]).astype(int)
                step_indices = np.clip(step_indices, 0, self.n_steps - 1)
                
                lookup[isccp_time] = {
                    'traj_indices': active_traj_indices,
                    'step_indices': step_indices,
                    'n_active': len(active_traj_indices)
                }
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(isccp_times)} time slots")
        
        print(f"Time lookup table complete: {len(lookup)} active time slots")
        return lookup
    
    def process_isccp_time_chunk(self, isccp_time, traj_chunk_start, traj_chunk_end):
        """Process a single ISCCP time for a chunk of trajectories."""
        if isccp_time not in self.time_lookup:
            return None, None, None  # No active trajectories for this time
        
        lookup_data = self.time_lookup[isccp_time]
        active_traj_indices = lookup_data['traj_indices']
        step_indices = lookup_data['step_indices']
        
        # Filter for current chunk
        chunk_mask = (active_traj_indices >= traj_chunk_start) & (active_traj_indices < traj_chunk_end)
        if not np.any(chunk_mask):
            return None, None, None
        
        chunk_traj_indices = active_traj_indices[chunk_mask] - traj_chunk_start
        chunk_step_indices = step_indices[chunk_mask]
        global_traj_indices = active_traj_indices[chunk_mask]
        
        # Load trajectory chunk
        with xr.open_dataset(self.trajectory_file) as ds:
            chunk_lons = ds.lon.isel(trajectory=slice(traj_chunk_start, traj_chunk_end)).values
            chunk_lats = ds.lat.isel(trajectory=slice(traj_chunk_start, traj_chunk_end)).values
        
        lons = chunk_lons[chunk_traj_indices, chunk_step_indices]
        lats = chunk_lats[chunk_traj_indices, chunk_step_indices]
        
        valid_mask = ~np.isnan(lons) & ~np.isnan(lats)
        if not np.any(valid_mask):
            return None, None, None
        
        lons = lons[valid_mask] % 360
        lats = lats[valid_mask]
        valid_global_traj = global_traj_indices[valid_mask]
        valid_steps = chunk_step_indices[valid_mask]
        
        return lons, lats, (valid_global_traj, valid_steps)
    
    def colocate_isccp_time(self, isccp_time, collection='isccp-basic', 
                           product='hgg', varname='cldamt_irtypes'):
        """Process all trajectory chunks for a single ISCCP time."""
        print(f"\nProcessing ISCCP time: {isccp_time}")
        
        if isccp_time not in self.time_lookup:
            print("  No active trajectories")
            return {}
        
        try:
            doy = isccp_time.timetuple().tm_yday
            granule = Granule(isccp_time.year, doy, isccp_time.hour)
            
            if not granule.check(collection, product):
                print(f"  Downloading ISCCP data...")
                granule.download(collection, product)
        except Exception as e:
            print(f"  Error loading ISCCP data: {e}")
            return {}
        
        all_lons, all_lats, all_indices = [], [], []
        n_chunks = (self.n_trajectories + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.n_trajectories)
            
            result = self.process_isccp_time_chunk(isccp_time, start_idx, end_idx)
            lons, lats, indices = result
            
            if lons is not None:
                all_lons.append(lons)
                all_lats.append(lats)
                all_indices.append(indices)
        
        if not all_lons:
            print("  No valid parcels found")
            return {}
        
        final_lons = np.concatenate(all_lons)
        final_lats = np.concatenate(all_lats)
        final_traj_indices = np.concatenate([idx[0] for idx in all_indices])
        final_step_indices = np.concatenate([idx[1] for idx in all_indices])
        
        print(f"  Colocating {len(final_lons):,} parcels...")
        
        try:
            isccp_values = granule.geolocate(collection, product, varname, final_lons, final_lats)
            
            if 'cloud_irtype' in isccp_values.dims:
                isccp_values = isccp_values.isel(cloud_irtype=0).squeeze()
            
            results = {}
            for i, (traj_idx, step_idx) in enumerate(zip(final_traj_indices, final_step_indices)):
                key = (int(traj_idx), int(step_idx))
                value = isccp_values.values[i] if hasattr(isccp_values, 'values') else isccp_values[i]
                results[key] = value
            
            print(f"  Successfully colocated {len(results):,} parcels")
            return results
            
        except Exception as e:
            print(f"  Error in ISCCP colocation: {e}")
            return {}
    
    def run_parallel_colocation(self, collection='isccp-basic', product='hgg', 
                               varname='cldamt_irtypes', output_file=None):
        """Run colocation using parallel processing across ISCCP times."""
        print(f"\n{'='*60}")
        print(f"STARTING PARALLEL COLOCATION")
        print(f"{'='*60}")
        
        isccp_times = list(self.time_lookup.keys())
        if self.debug:
            isccp_times = isccp_times[:2]
            print(f"[DEBUG MODE] Restricting to {len(isccp_times)} ISCCP time slots")
        
        print(f"Processing {len(isccp_times)} ISCCP time slots using {self.n_workers} workers")
        
        if output_file:
            temp_file = output_file.replace('.nc', '_temp.dat')
            isccp_data = np.memmap(temp_file, dtype='float32', mode='w+', 
                                  shape=(self.n_trajectories, self.n_steps))
            isccp_data[:] = np.nan
        else:
            isccp_data = np.full((self.n_trajectories, self.n_steps), np.nan, dtype=np.float32)
        
        colocation_func = partial(self.colocate_isccp_time, 
                                 collection=collection, product=product, varname=varname)
        
        completed = 0
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_time = {executor.submit(colocation_func, time): time 
                             for time in isccp_times}
            
            for future in as_completed(future_to_time):
                isccp_time = future_to_time[future]
                completed += 1
                
                try:
                    results = future.result()
                    for (traj_idx, step_idx), value in results.items():
                        isccp_data[traj_idx, step_idx] = value
                    
                    print(f"Completed {completed}/{len(isccp_times)}: {isccp_time} "
                          f"({len(results):,} matches)")
                    
                except Exception as e:
                    print(f"Error processing {isccp_time}: {e}")
        
        print(f"\n{'='*60}")
        print(f"COLOCATION COMPLETE")
        print(f"{'='*60}")
        
        ds_output = self._create_output_dataset(isccp_data, collection, product, varname)
        
        if output_file:
            print(f"Saving results to {output_file}")
            if self.debug:
                ds_output.to_netcdf(output_file)
            else:
                encoding = {
                    'isccp_data': {'chunksizes': (self.chunk_size, self.n_steps)},
                    'lon': {'chunksizes': (self.chunk_size, self.n_steps)},
                    'lat': {'chunksizes': (self.chunk_size, self.n_steps)}
                }
                ds_output.to_netcdf(output_file, encoding=encoding)
            
            if isinstance(isccp_data, np.memmap):
                del isccp_data
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return ds_output
    
    def _create_output_dataset(self, isccp_data, collection, product, varname):
        """Create the final output dataset."""
        with xr.open_dataset(self.trajectory_file, chunks={'trajectory': self.chunk_size}) as ds_traj:
            ds_output = xr.Dataset(
                data_vars=dict(
                    lon=(["trajectory", "step"], ds_traj.lon.data),
                    lat=(["trajectory", "step"], ds_traj.lat.data),
                    isccp_data=(["trajectory", "step"], isccp_data)
                ),
                coords=dict(
                    trajectory=ds_traj.trajectory,
                    step=ds_traj.step,
                    start_time=ds_traj.start_time
                ),
                attrs=dict(
                    isccp_collection=collection,
                    isccp_product=product,
                    isccp_variable=varname,
                    description=f"Optimized trajectory-ISCCP colocation",
                    n_trajectories=self.n_trajectories,
                    chunk_size=self.chunk_size,
                    n_workers=self.n_workers,
                    debug_mode=self.debug
                )
            )
        
        ds_output.isccp_data.attrs = {
            'long_name': f'ISCCP {varname}',
            'source': 'ISCCP',
            'colocation_method': 'optimized_parallel'
        }
        
        return ds_output


def run_optimized_colocation(trajectory_file, output_file, 
                           chunk_size=50000, n_workers=None,
                           collection='isccp-basic', product='hgg', 
                           varname='cldamt_irtypes', debug=False):
    """
    Main function to run optimized trajectory-ISCCP colocation.
    """
    if debug:
        # Avoid overwriting big outputs by accident
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_debug{ext}"
    
    print(f"Starting optimized colocation:")
    print(f"  Input: {trajectory_file}")
    print(f"  Output: {output_file}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  Workers: {n_workers or 'auto'}")
    print(f"  Debug: {debug}")
    
    processor = OptimizedTrajectoryColocation(
        trajectory_file=trajectory_file,
        chunk_size=chunk_size,
        n_workers=n_workers,
        debug=debug
    )
    
    ds_output = processor.run_parallel_colocation(
        collection=collection,
        product=product,
        varname=varname,
        output_file=output_file
    )
    
    analyze_colocation_results(ds_output)
    return ds_output


def analyze_colocation_results(ds):
    """Quick analysis of colocation results."""
    isccp_data = ds.isccp_data.values
    valid_data = isccp_data[~np.isnan(isccp_data)]
    
    print(f"\n{'='*40}")
    print(f"COLOCATION RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"Total trajectory points: {isccp_data.size:,}")
    print(f"Valid ISCCP matches: {len(valid_data):,}")
    print(f"Overall match rate: {len(valid_data)/isccp_data.size*100:.2f}%")
    
    if len(valid_data) > 0:
        print(f"ISCCP data range: {valid_data.min():.3f} to {valid_data.max():.3f}")
        print(f"Mean ± std: {valid_data.mean():.3f} ± {valid_data.std():.3f}")

#%%
if __name__ == "__main__":
    trajectory_file = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis/trajectories_20150101_20160101.nc"
    output_file = "/disk1/Users/gjp23/outputs/traj_positions/global_analysis/trajectories_isccp_optimized_20150101_20160101.nc"
    
    chunk_size = 100000
    n_workers = 12
    
    try:
        ds_result = run_optimized_colocation(
            trajectory_file=trajectory_file,
            output_file=output_file,
            chunk_size=chunk_size,
            n_workers=n_workers,
            debug=True  # 🚨 only first 2 ISCCP slots + unchunked output
        )
        
        print("\nOptimized colocation complete!")
        
    except Exception as e:
        print(f"Error in optimized colocation: {e}")
        raise
# %%
