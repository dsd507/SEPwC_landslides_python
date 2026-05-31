"""
Calculate hazard risk of probability for landslides
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import rioxarray
import rasterio
from rasterio.features import rasterize
import xarray as xr
from xrspatial import slope as xr_slope
from xrspatial import proximity as xr_proximity


def extract_values_from_raster(da, shapes):
    """Extract raster values that are nearest to input point geometries."""
    x_coords = xr.DataArray([geom.x for geom in shapes], dims="points")
    y_coords = xr.DataArray([geom.y for geom in shapes], dims="points")
    # Sample one raster value per point for the training dataframe.
    values = da.sel(x=x_coords, y=y_coords, method="nearest").values
    return values.ravel()


def make_classifier(x, y, verbose=False):
    """Train a random forest classifier."""
    # Split the labelled data so the model can be checked on unseen samples.
    features_train, features_test, labels_train, labels_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(features_train, labels_train)

    if verbose:
        predictions = classifier.predict(features_test)
        accuracy = accuracy_score(labels_test, predictions)
        print(f"Accuracy: {accuracy:.3f}")

    return classifier


def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    """Predict landslide probabilities for the raster grid."""
    # Needs every raster layer to build the inputs, so it has a few extra arguments.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # Build classifier inputs for every cell using the same columns as training.
    cell_features = pd.DataFrame({
        "elev": topo.values.ravel(),
        "fault": dist_fault.values.ravel(),
        "slope": slope.values.ravel(),
        "LC": lc.values.ravel(),
        "Geol": geo.values.ravel(),
    })

    # Only classify cells that have data in every layer.
    complete_cells = ~cell_features.isna().any(axis=1)
    probability_values = np.zeros(len(cell_features))
    probability_values[complete_cells] = classifier.predict_proba(
        cell_features.loc[complete_cells]
    )[:, 1]
    return probability_values.reshape(topo.shape)


def create_dataframe(topo, geo, lc, dist_fault, slope, shapes, landslide_label):
    """Build a labelled training dataframe from raster values at sample points."""
    # The tests call this with a set list of arguments, so it goes over the limit.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
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
    """Calculate distance to the nearest fault on the template raster grid."""
    faults = gpd.read_file(fault_shapefile)
    faults = faults.to_crs(template_raster.rio.crs)

    # Turn the fault lines into a raster mask on the same grid as the template.
    fault_mask = rasterize(
        [(geometry, 1) for geometry in faults.geometry],
        out_shape=template_raster.shape,
        transform=template_raster.rio.transform(),
        fill=0,
        dtype="uint8",
    )

    fault_grid = template_raster.copy(data=fault_mask)

    # Measure distance from every cell to the nearest fault cell.
    return xr_proximity(fault_grid, target_values=[1])


def make_background_points(template_raster, count):
    """Create random background points within the raster extent."""
    # Use the raster bounds so the points fall inside the study area.
    minx, miny, maxx, maxy = template_raster.rio.bounds()
    np.random.seed(42)
    sample_x = np.random.uniform(minx, maxx, count)
    sample_y = np.random.uniform(miny, maxy, count)
    return gpd.GeoSeries(
        [Point(x, y) for x, y in zip(sample_x, sample_y)],
        crs=template_raster.rio.crs)


def main(args_list=None):
    """Run the landslide hazard model."""
    # main runs the whole pipeline, so it uses more local variables than the limit.
    # pylint: disable=too-many-locals
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
    parser.add_argument('--plot', help="save a plot of the risk map to this image file")
    args = parser.parse_args(args_list)

    # Topography sets the grid that every other layer is matched against.
    topo = rioxarray.open_rasterio(args.topography, masked=True).squeeze()

    # Geology and landcover are snapped to that grid so the layers stack cleanly.
    geo = reproject_to_match(
        rioxarray.open_rasterio(args.geology, masked=True).squeeze(), topo)
    lc = reproject_to_match(
        rioxarray.open_rasterio(args.landcover, masked=True).squeeze(), topo)

    # Slope and distance to faults are derived from the topography.
    slope = xr_slope(topo)
    dist_fault = calculate_distance_to_faults(args.faults, topo)

    if args.verbose:
        print("Loaded rasters and derived slope and fault distance.")

    # Landslide locations are the positive examples for the classifier.
    landslides = gpd.read_file(args.landslides).to_crs(topo.rio.crs)
    positive_samples = create_dataframe(
        topo, geo, lc, dist_fault, slope, landslides.geometry.centroid, 1)

    # Match those with the same number of random points as negative examples.
    background_points = make_background_points(topo, len(landslides))
    negative_samples = create_dataframe(topo, geo, lc, dist_fault, slope, background_points, 0)

    # Combine both sets and fit the random forest.
    training_data = pd.concat([positive_samples, negative_samples])
    classifier = make_classifier(
        training_data.drop("ls", axis=1), training_data["ls"],
        verbose=args.verbose)

    # Run the trained model over every cell to get a probability grid.
    probability_values = make_prob_raster_data(
        topo, geo, lc, dist_fault, slope, classifier)

    if args.verbose:
        print("Writing probability raster to " + args.output)

    # Save the result as a single band raster on the topography grid.
    with rasterio.open(
        args.output, "w",
        driver="GTiff",
        height=probability_values.shape[0],
        width=probability_values.shape[1],
        count=1,
        dtype=probability_values.dtype,
        crs=topo.rio.crs,
        transform=topo.rio.transform(),
    ) as output_raster:
        output_raster.write(probability_values, 1)

    # Optionally save an image of the risk map.
    if args.plot:
        probability_da = topo.copy(data=probability_values)
        probability_da.plot(cmap="viridis")
        plt.title("Landslide hazard probability")
        plt.savefig(args.plot)
        if args.verbose:
            print("Saved risk map to " + args.plot)


if __name__ == '__main__':
    main()
