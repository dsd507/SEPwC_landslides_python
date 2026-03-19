import pytest
import os
import sys
sys.path.insert(0,os.pardir)
sys.path.insert(0,".")
from terrain_analysis import *
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from dataclasses import asdict
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sklearn # pip install scikit-learn
import io
from contextlib import redirect_stdout

@pytest.fixture
def data_dir():
    # Anchors to the folder where this test file sits
    return Path(__file__).parent / "data"

@pytest.fixture
def main_dir():
    # Anchors to the folder where this test file sits
    return Path(__file__).parent / os.pardir / "data"

@pytest.fixture
def sample_rasters(data_dir):
    """Returns a dictionary of paths to test files."""
    return {
        "topo": data_dir / "raster_template.tif",
        "landslides": data_dir / "test_point.shp"
    }

class TestTerrainAnalysis():

    def test_extract_values_alignment(self,sample_rasters):
        """Test that clipping returns the expected number of pixels."""
        topo = rioxarray.open_rasterio(sample_rasters["topo"], masked=True).squeeze()
        landslides = gpd.read_file(sample_rasters["landslides"])
        
        # Run the extraction
        vals = extract_values_from_raster(topo, landslides.geometry)
        
        assert isinstance(vals, np.ndarray)
        assert len(vals) > 0  # Ensure we actually grabbed data
        assert not np.isnan(vals).any()
        assert len(vals) == 2
        assert vals[0] == pytest.approx(2509.6870)
        assert vals[1] == pytest.approx(2534.5088)


    def test_create_dataframe_schema(self,sample_rasters):
        """Verify the ML dataframe has the correct columns for Scikit-Learn."""
        topo = rioxarray.open_rasterio(sample_rasters["topo"], masked=True).squeeze()
        # Dummy data for the other layers to test stacking
        slope = topo.copy()
        geol = topo.copy()
        lc = topo.copy()
        faults = topo.copy()
        
        landslides = gpd.read_file(sample_rasters["landslides"])
        
        df = create_dataframe(topo, geol, lc, faults, slope, landslides.geometry, landslide_label=1)
        
        assert isinstance(df, (gpd.geodataframe.GeoDataFrame, pd.DataFrame))
        assert len(df) == 2
        expected_cols = {"elev", "fault", "slope", "LC", "Geol", "ls"}
        assert expected_cols.issubset(df.columns)
        assert (df['ls'] == 1).all()

    def test_reproject_match_error(self,sample_rasters):
        """Ensure our script handles misaligned CRSs gracefully."""
        topo = rioxarray.open_rasterio(sample_rasters["topo"], masked=True).squeeze()

        # Create a 'fake' geology layer with a different CRS
        geol_wrong_crs = topo.copy().rio.write_crs("EPSG:4326") 
        
        # This mimics the main() logic: reproject_match should fix it
        aligned_geo = reproject_to_match(geol_wrong_crs, topo)
        
        assert aligned_geo.rio.crs == topo.rio.crs
        assert aligned_geo.shape == topo.shape
    
 
    def test_make_classifier(self):

        test_data =  np.random.normal(size=20)
        data = {
            "x1": test_data,
            "x2": test_data * 2.45,
            "y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        df = pd.DataFrame(data)
        classifier = make_classifier(df.drop('y',axis=1),df['y'])
        assert type(classifier) == sklearn.ensemble._forest.RandomForestClassifier
        assert classifier.n_classes_ == 2


    def test_lint(self):
        files =  ["terrain_analysis.py"]
        #pylint_options = ["--disable=line-too-long,import-error,fixme"]
        pylint_options = []

        report = CollectingReporter()
        result = Run(
                    files,
                    reporter=report,
                    exit=False,
                )
        score = result.linter.stats.global_note
        nErrors = len(report.messages)

        print("Score: " + str(score))
        line_format = "{path}:{line}:{column}: {msg_id}: {msg} ({symbol})"
        for error in report.messages:
            print(line_format.format(**asdict(error)))   

        score_thresholds = [3, 5, 7, 9]
        error_thresholds = [500, 250, 100, 50, 10, 0]
        
        results = {"pass": 0, "fail": 0}

        for t in score_thresholds:
            if score > t:
                results["pass"] += 1
            else:
                results["fail"] += 1

        for t in error_thresholds:
            if nErrors <= t: 
                results["pass"] += 1
            else:
                results["fail"] += 1

        print(f"You passed {results['pass']} out of {len(score_thresholds) + \
                len(error_thresholds)} lint checks.")
        
        # Finally, trigger a failure if they didn't get a perfect score
        # or just assert results["fail"] == 0
        assert results["fail"] == 0, f"You failed {results['fail']} lint tests."

class TestRegression():

    def test_regression(self, main_dir):

        args = ["--topography",
                os.path.join(main_dir,"AW3D30.tif"),
                "--geology",
                os.path.join(main_dir,"Geology.tif"),
                "--landcover",
                os.path.join(main_dir,"Landcover.tif"),
                "--faults",
                os.path.join(main_dir,"Confirmed_faults.shp"),
                os.path.join(main_dir,"landslides.shp"),
                "test.tif"]
        f = io.StringIO() 
        with redirect_stdout(f):
            main(args_list = args)
        output = f.getvalue()
        assert len(output) < 5

        raster = rasterio.open("test.tif")
        values = raster.read(1)
        assert values.max() <= 1
        assert values.min() >= 0
        os.remove("test.tif")
        

    def test_regression_verbose(self, main_dir):

        args = ["--topography",
                os.path.join(main_dir,"AW3D30.tif"),
                "--geology",
                os.path.join(main_dir,"Geology.tif"),
                "--landcover",
                os.path.join(main_dir,"Landcover.tif"),
                "--faults",
                os.path.join(main_dir,"Confirmed_faults.shp"),
                os.path.join(main_dir,"landslides.shp"),
                '--v',
                "test.tif"]
        f = io.StringIO() 
        with redirect_stdout(f):
            main(args_list = args)
        output = f.getvalue()
        assert len(output) > 50

        raster = rasterio.open("test.tif")
        values = raster.read(1)
        assert values.max() <= 1
        assert values.min() >= 0
        os.remove("test.tif")



