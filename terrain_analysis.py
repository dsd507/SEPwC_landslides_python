"""
Calculate hazard risk of probability for landslides
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import rioxarray
import rasterio
import xarray as xr
from xrspatial import slope as xr_slope
from xrspatial import proximity as xr_proximity

def extract_values_from_raster(da, shapes):
    """Extract raster values that are nearest to input point geometries."""
    # Extract x and y coordinates from the input point geometries.
    x_coords = xr.DataArray([geom.x for geom in shapes], dims="points")
    y_coords = xr.DataArray([geom.y for geom in shapes], dims="points")
    # Sample one raster value per point for the training dataframe.
    values = da.sel(x=x_coords, y=y_coords, method="nearest").values
    return np.asarray(values).ravel()
    

def make_classifier(x, y, verbose=False):
    return


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    return


def create_dataframe(topo, geo, lc, dist_fault, slope, shapes, landslide_label):
    """Build a labelled training dataframe from raster values at sample points."""
    # Sample each raster layer at the same points for the classifier inputs.
    data = {
        "elev": extract_values_from_raster(topo, shapes),
        "fault": extract_values_from_raster(dist_fault, shapes),
        "slope": extract_values_from_raster(slope, shapes),
        "LC": extract_values_from_raster(lc, shapes),
        "Geol": extract_values_from_raster(geo, shapes),
        "ls": landslide_label,
    }

    # Remove samples with missing raster data before training the model.
    return pd.DataFrame(data).dropna()

def reproject_to_match(in_raster, template_raster):
    """Reproject and resample a raster to match a template raster."""
    return in_raster.rio.reproject_match(template_raster)


def calculate_distance_to_faults(fault_shapefile, template_raster):
    return


def main(args_list=None):
    parser = argparse.ArgumentParser(
        prog="Landslide hazard using ML",
        description="Calculate landslide hazards using machine learning"
    )
    parser.add_argument('--topography', required=True, help="topographic raster file")
    parser.add_argument('--geology', required=True, help="geology raster file")
    parser.add_argument('--landcover', required=True, help="landcover raster file")
    parser.add_argument('--faults', required=True, help="fault location shapefile")
    parser.add_argument("landslides", help="landslide location shapefile")
    parser.add_argument("output", help="output probability raster file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print progress")
    
    args = parser.parse_args(args_list)


if __name__ == '__main__':
    main()
