import geopandas as gpd
from pathlib import Path

# Path to your shapefile
shapefile_path = Path('/home/jldz9/DL/DL_drake/shp/shurbcrown.shp')

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Display the attribute table
print(gdf)