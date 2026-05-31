"""
Extra tests for the landslide terrain analysis
"""
import os
import sys
from pathlib import Path
import pytest
import rioxarray
import geopandas as gpd

sys.path.insert(0, os.pardir)
sys.path.insert(0, ".")
from terrain_analysis import make_background_points, create_dataframe


@pytest.fixture
def data_dir():
    """Return the path to the test data folder."""
    return Path(__file__).parent / "data"


@pytest.fixture
def template_raster(data_dir):
    """Load the small template raster."""
    return rioxarray.open_rasterio(
        data_dir / "raster_template.tif", masked=True).squeeze()


class TestHelpers:

    def test_background_points_within_bounds(self, template_raster):
        """Check every background point falls inside the raster extent."""
        minx, miny, maxx, maxy = template_raster.rio.bounds()
        points = make_background_points(template_raster, 100)
        assert (points.x >= minx).all() and (points.x <= maxx).all()
        assert (points.y >= miny).all() and (points.y <= maxy).all()

    def test_background_points_crs_matches(self, template_raster):
        """Check the points share the template raster's CRS."""
        points = make_background_points(template_raster, 10)
        assert points.crs == template_raster.rio.crs

    def test_create_dataframe_negative_label(self, data_dir, template_raster):
        """Check create_dataframe gives negative samples a zero label."""
        points = gpd.read_file(data_dir / "test_point.shp")
        negative_samples = create_dataframe(
            template_raster, template_raster, template_raster,
            template_raster, template_raster, points.geometry, 0)
        assert len(negative_samples) == 2
        assert (negative_samples["ls"] == 0).all()
