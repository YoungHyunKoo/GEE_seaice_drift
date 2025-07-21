import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob, os
import rasterio
from datetime import datetime
import pickle
from nansat import Nansat, Domain, NSR
from tqdm import tqdm
from pyproj import Proj, transform
import argparse

import requests
import zipfile

from sea_ice_drift.lib import get_spatial_mean, get_uint8_image
from sea_ice_drift.ftlib import feature_tracking
from sea_ice_drift.pmlib import pattern_matching
from sea_ice_drift.libdefor import get_deformation_nodes
from functions import *

import warnings
warnings.filterwarnings('ignore')

import geemap
import ee
from datetime import datetime, timedelta

try:
  ee.Initialize(project = "utsa-spring2024")
except:
  ee.Authenticate()
  ee.Initialize(project = "utsa-spring2024")

def get_S1_array(center, t1, t2, pixel_size = 60, distance = 100000, proj = "EPSG:3409"):
    # proj - EPSG:3409 (EASE South); EPSG:3976 (NSIDC south polar stereo); EPSG:3857 (Web Mercator)

    geometry = ee.Geometry.Point(center[0], center[1])
    
    if proj == "EPSG:3857":
        extent = geometry.buffer(distance = distance).bounds()
        dp = appropriate_resolution(center, pixel_size)
    else:
        extent = geometry.buffer(distance = distance, proj = proj).bounds(proj = proj)
        dp = pixel_size
    
    # Define necessary functions with the defined bbox --------------------------
    def collection_addbands(img):
        bands = img.bandNames() # First band ('HH' or 'VV')
        band = [ee.Algorithms.If(bands.contains('HH'), 'HH', 'VV')]
        norm = img.select(band).divide(img.select('angle')).rename('norm')
        img2 = img.addBands(norm, overwrite=True).select('norm')
        return img2
    def add_coverage(img):    
        tol = 10000
        overlap = img.geometry().intersection(extent, tol)
        ratio = overlap.area(tol).divide(extent.area(tol))
        return img.set({'coverage_ratio': ratio})
    # calculate coverage area of image to roi
    def coverage(img):
        tol = 10000
        overlap = img.geometry().intersection(extent, tol)
        ratio = overlap.area(tol).divide(extent.area(tol))
        return ratio.getInfo()
    # -------------------------------------------------------------------------
    
    collection0 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(extent).filterDate(t1, t2)\
    .filter(ee.Filter.eq('instrumentMode', 'EW'))
    
    collection1 = collection0.map(collection_addbands)
    collection = collection1.map(add_coverage).filter(ee.Filter.gt('coverage_ratio', 0.5)).sort('coverage_ratio', False)

    S1_ids = collection.aggregate_array('system:id').getInfo()    

    if len(S1_ids) > 0:
        # print(S1_ids, dp)
        S1_id = S1_ids[0]
        
        img = collection_addbands(ee.Image(S1_id)).setDefaultProjection(proj)
        #("EPSG:3857") #.reproject("EPSG:4326") # ("EPSG:3976") NSIDC southpolar
        img = add_coverage(img)

        name = os.path.basename(S1_id)[17:32]
        description = f"S1_{name}"                   
        
        img2 = img.clip(extent).select("norm")
            
        a = geemap.ee_to_numpy(img2, scale = dp)[:, :, 0]
        # a[a>=0] = np.nan

        return a, S1_id
        
    else:
        return np.array([]), ""

def get_bbox(center, distance):
    geometry = ee.Geometry.Point(center[0], center[1]).buffer(distance = distance).bounds()
    extent = np.array(geometry.getInfo()['coordinates'][0])

    east, north = extent[2] #np.max(extent, axis = 0)
    west, south = extent[0] #np.min(extent, axis = 0)

    if east > 180:
        east = -360 + east
    elif east < -180:
        east = 360 - east
    if west > 180:
        west = -360 + west
    elif west < -180:
        west = 360 - west

    bbox = [west, south, east, north]
    
    return bbox

# SIV database
def get_siv_mask(year, datapath):
    file = f"{datapath}\\icemotion_weekly_sh_25km_{year}0101_{year}1231_v4.1.nc"
    
    with netCDF4.Dataset(file, 'r') as nc:
        xs = np.array(nc.variables['x']) #[20:-20]
        ys = np.array(nc.variables['y']) #[20:-20]
        xx1, yy1 = np.meshgrid(xs, ys)
        lat = np.array(nc.variables['latitude']) #[20:-20, 20:-20]
        lon = np.array(nc.variables['longitude']) #[20:-20, 20:-20]
        # lon = np.where(lon < 0, 360+lon, lon)
    
        # days = np.array(nc.variables['time']) #.astype(float)
        times = nc.variables['time']
        times = num2date(times[:], units = times.units)
        u = np.array(nc.variables['u']) #[:, 20:-20, 20:-20] 
        u[u < -9000] = np.nan
        u = u*0 + 1

    return xx1, yy1, lat, lon, u

def parse_args() -> argparse.Namespace:    
    # General settings
    parser = argparse.ArgumentParser(description='Argument settings')       
    parser.add_argument(
        '--year',
        type=int,
        default=2021,
        help='Target year',
    )
    
    args = parser.parse_args()

    return args

######## START ###############################################################
args = parse_args()

year = args.year

w = [5, 2]
pixel_size = 100
step = 8
pm_step = 20
distance = 25000 * step / 2

proj = "EPSG:3409"
transformer = Transformer.from_crs(proj, "EPSG:4326")

sivpath = "D:\\IS2_topo_DL\\SIV";
datapath = "D:\\NERC-NSF\\ice_vel";
xx1, yy1, lat, lon, u = get_siv_mask(year, sivpath)
row, col = lat.shape

for ii in range(0, row, step):
    for jj in range(0, col, step):

        try:
            ee.Initialize(project = "utsa-spring2024")
        except:
            ee.Authenticate()
            ee.Initialize(project = "utsa-spring2024")

        clat = float(lat[ii, jj])
        clon = float(lon[ii, jj])      

        center0 = np.array([clon, clat]) # lon, lat (Actually, it's upper right corner of the polygon)
        center = center0.copy()        
        
        geometry = ee.Geometry.Point(center[0], center[1])
        
        extent = geometry.buffer(distance = distance, proj = proj).bounds(proj = proj)
        bbox = [center[0]-w[0], center[1]-w[1], center[0]+w[0], center[1]+w[1]]        
        extent_coords = np.array(extent.coordinates().getInfo()[0])
        
        # lat0, lon0 = transformer.transform(extent_coords[:, 0], extent_coords[:, 1])
        # bbox = [lon0.min(), lon0.max(), lat0.min(), lat0.max()]        
        
        dp = pixel_size
        
        # Define necessary functions with the defined bbox --------------------------
        def collection_addbands(img):
            bands = img.bandNames() # First band ('HH' or 'VV')
            band = [ee.Algorithms.If(bands.contains('HH'), 'HH', 'VV')]
            norm = img.select(band).divide(img.select('angle')).rename('norm')
            img2 = img.addBands(norm, overwrite=True).select('norm')
            return img2
        def add_coverage(img):    
            tol = 10000
            overlap = img.geometry().intersection(extent, tol)
            ratio = overlap.area(tol).divide(extent.area(tol))
            return img.set({'coverage_ratio': ratio})
        # calculate coverage area of image to roi
        def coverage(img):
            tol = 10000
            overlap = img.geometry().intersection(extent, tol)
            ratio = overlap.area(tol).divide(extent.area(tol))
            return ratio.getInfo()
        # -------------------------------------------------------------------------
        
        for month in range(1, 13): 
        
            ### LOAD SENTINEL-1 DATA FROM GOOGLE EARTH ENGINE AND SAVE AS NUMPY ARRAYS #######################
            start_date = f"{year}-{str(month).zfill(2)}-01" #02
            if month == 12:
                end_date = f"{year+1}-01-01" #20
            else:
                end_date = f"{year}-{str(month+1).zfill(2)}-01" #20

            jidx = int(datetime.strptime(start_date, "%Y-%m-%d").strftime("%j")) // 7
            
            pkl_era5 = f"{datapath}\\ERA5_wind_{year}{str(month).zfill(2)}_{str(ii).zfill(3)}_{str(jj).zfill(3)}.pkl"            
            
            if np.nansum(u[jidx:jidx+4, ii-step:ii+step, jj-step:jj+step]) > 4*(2*step)**2 * 0.3 and os.path.exists(pkl_era5) == False:

                print(f"----- [{clon:.1f}, {clat:.1f} ({ii}, {jj})] {year} {str(month).zfill(2)} -----")
                
                ##### LOAD AND SAVE ERA5 REANALYSIS DATA ####################### 
                days = [str(i).zfill(2) for i in range(1, 32)]
                ds, fl = retrieve_hourly_ERA5_bbox(year, str(month).zfill(2), days, bbox)
            
                era_times = ds.valid_time.values
                era_lat = ds.latitude.values
                era_lon = ds.longitude.values
                era_u10 = ds.u10.values
                era_v10 = ds.v10.values
                era_t2m = ds.t2m.values
                era_sic = ds.siconc.values
                
                pkl_save = [era_times, era_lat, era_lon, era_u10, era_v10, era_t2m, era_sic, bbox]
                try:
                    with open(pkl_era5, 'wb') as handle:
                        pickle.dump(pkl_save, handle)
                except:
                    pkl_era5 = f"ERA5_wind_{year}{str(month).zfill(2)}_{str(ii).zfill(3)}_{str(jj).zfill(3)}.pkl"
                    with open(pkl_era5, 'wb') as handle:
                        pickle.dump(pkl_save, handle)                    
                del ds
                
                try:
                    os.remove(fl)
                except:
                    pass
                    
                print("ERA5 data is loaded!")
                ##### =====================================================================================

                '''
                d0 = datetime.strptime(start_date, "%Y-%m-%d")
                d2 = datetime.strptime(end_date, "%Y-%m-%d")
                
                dn = abs(d2-d0).days
                
                map = geemap.Map()
                array, times, Hs, Ws, xxs, yys = [], [], [], [], [], []
                
                for i in tqdm(range(dn)):
                
                    t1 = (d0 + timedelta(days = i))
                    t2 = (d0 + timedelta(days = i+1))
                
                    t1_str = t1.strftime("%Y-%m-%d")
                    
                    collection0 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(extent).filterDate(t1, t2)\
                    .filter(ee.Filter.eq('instrumentMode', 'EW'))
                    
                    collection1 = collection0.map(collection_addbands)
                    collection = collection1.map(add_coverage).filter(ee.Filter.gt('coverage_ratio', 0.3))
                
                    S1_ids = collection.aggregate_array('system:id').getInfo()
                
                    if len(S1_ids) > 1:
                        rs = collection.aggregate_array('coverage_ratio').getInfo()
                        S1_ids = [S1_ids[np.argmax(rs)]]
                
                    for k in S1_ids:
                        
                        img = collection_addbands(ee.Image(k)).setDefaultProjection(proj)
                        img = add_coverage(img)
                
                        name = os.path.basename(k)[17:32]
                        description = f"S1_{name}" 
                        # band_coord = ee.Image.pixelCoordinates(proj)
                        # img = img.addBands(band_coord)
                        img = img.clip(extent)                  
                        
                        img2 = img.select("norm")
                        # xx2 = img.select("x")
                        # yy2 = img.select("y")
                            
                        a = geemap.ee_to_numpy(img2, scale = dp)[:, :, 0]
                        # xx = geemap.ee_to_numpy(xx2, scale = dp)[:, :, 0]
                        # yy = geemap.ee_to_numpy(yy2, scale = dp)[:, :, 0]
                        
                        a[a>=0] = np.nan
                        array.append(a)
                        Hs.append(a.shape[0])
                        Ws.append(a.shape[1])
                        # xxs.append(xx)
                        # yys.append(yy)
                        times.append(datetime.strptime(description[-15:], "%Y%m%dT%H%M%S"))

                if len(array) > 1:
                    H_target = np.bincount(Hs).argmax()
                    W_target = np.bincount(Ws).argmax()
                    ind_target = np.where((Hs == H_target) & (Ws == W_target))[0]
        
                    array1 = np.zeros([len(ind_target), H_target, W_target])
                    for i, n in enumerate(ind_target):
                        array1[i] = array[n]
                    
                    # del array
                
                    print("SAR images: ", array1.shape[0])
                    
                    x0 = str(center[0].round(1))
                    y0 = str(center[1].round(1))
                
                    ### DERIVE AND SAVE SEA ICE DRIFT #######################
                    ind_final1 = []
                    ind_final2 = []
                    pkl_objects = []
                    upms = []
                    vpms = []
                    apms = []
                    rpms = []
                    hpms = []
                    xpms = []
                    ypms = []
                    
                    for ind1 in range(0, array1.shape[0]-1):
                        ind2 = ind1 + 1
                        pkl_object = derive_drift(array1[ind1], array1[ind2], times[ind1], times[ind2], extent_coords, pixel_size, pm_step = pm_step)
                    
                        if len(pkl_object) > 0:
                            # pkl_objects.append(pkl_object)
                            upm, vpm, apm, rpm, hpm, xpm, ypm = pkl_object
                            upms.append(upm)
                            vpms.append(vpm)
                            apms.append(apm)
                            rpms.append(rpm)
                            hpms.append(hpm)
                            xpms.append(xpm)
                            ypms.append(ypm)
                            
                            if ind1 not in ind_final1:
                                ind_final1.append(ind1)
                            if ind2 not in ind_final2:
                                ind_final2.append(ind2)
                    
                    upms = np.array(upms)
                    vpms = np.array(vpms)
                    apms = np.array(apms)
                    rpms = np.array(rpms)
                    hpms = np.array(hpms)
                    xpms = np.array(xpms)
                    ypms = np.array(ypms)

                    if len(upms) > 0:
                    
                        pkl_objects = [upms, vpms, apms, rpms, hpms, xpms, ypms]                            
                        
                        pkl_save = [pkl_objects, ind_final1, ind_final2, times, extent_coords]
                        try:
                            pkl_s1 = f"{datapath}\\S1_vel_{year}{str(month).zfill(2)}_{str(ii).zfill(3)}_{str(jj).zfill(3)}.pkl"
                            with open(pkl_s1, 'wb') as handle:
                                pickle.dump(pkl_save, handle)
                        except:
                            pkl_s1 = f"S1_vel_{year}{str(month).zfill(2)}_{str(ii).zfill(3)}_{str(jj).zfill(3)}.pkl"
                            with open(pkl_s1, 'wb') as handle:
                                pickle.dump(pkl_save, handle)                            
            
                        print(f"Ice drift tracking: {len(ind_final1)}")
                        del pkl_objects, upms, vpms, apms, rpms, hpms, xpms, ypms, array, array1
                        '''

print("Done!")