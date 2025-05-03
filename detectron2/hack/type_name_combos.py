
"""Analyze a given shp file with both "Type" and "Names" keys to see if they
match.

If Type is just an index into the class list and Names are just these classes,
the 2 keys are redundant. If not, there is something deeper going on.  

Usage: 
    1. Modify shp="path/to/your/shp"
    2. python3 ./type_name_combos.py
"""
import geopandas as gpd
import pandas as pd

shp = "/home/desinotorious/rtmp/data/shola/data/Labels_Polygons_All.shp"

def analyze_types_and_names(shp_path):
    # Read the shapefile
    gdf = gpd.read_file(shp_path)
    
    # Create a DataFrame of unique Types-Name combinations
    type_name_pairs = gdf[['Types', 'Name']].drop_duplicates()
    
    # Sort by both columns for easier reading
    type_name_pairs = type_name_pairs.sort_values(['Types', 'Name'])
    
    # Get counts of each Name for each Type
    type_name_counts = gdf.groupby(['Types', 'Name']).size().reset_index(name='count')
    
    print("Unique Types-Name combinations:")
    print(type_name_pairs.to_string(index=False))
    print("\nCounts for each combination:")
    print(type_name_counts.to_string(index=False))
    
    # Check if Types is just an index into Names
    names = gdf['Name'].unique()
    types = gdf['Types'].unique()
    
    print("\nUnique Names:", len(names))
    print("Unique Types:", len(types))
    
    # Check if each Name always corresponds to the same Type
    is_one_to_one = len(gdf.groupby('Name')['Types'].nunique().unique()) == 1
    print("\nIs there a one-to-one relationship between Names and Types?", is_one_to_one)
    
    if not is_one_to_one:
        print("\nNames with multiple Types:")
        multiple_types = gdf.groupby('Name')['Types'].nunique()
        print(multiple_types[multiple_types > 1].to_string())

# Use it like this:
analyze_types_and_names(shp)
