import json
import argparse
import pandas as pd
import urllib.parse


def convert_json_to_excel(offloaded_metadata_path, map_metrics_path, output_path):
    # Load the offloaded metadata
    with open(offloaded_metadata_path) as f:
        data = json.load(f)

    # Load the map metrics
    with open(map_metrics_path) as f:
        map_metrics_data = json.load(f)

    # Create a lookup dictionary for map metrics by site name
    map_metrics_lookup = {}
    for metric_entry in map_metrics_data:
        site_name = metric_entry["name"]
        map_metrics_lookup[site_name] = metric_entry["stats"]

    tiles = []
    sites_dict = {}

    for entry in data:
        site_name = entry["parent"]["name"]
        parent_image = entry["parent"]["image"]

        # Properly encode URLs to prevent corruption
        tile_url = entry["image"]["source"]
        tile_preview = entry["image"].get("preview")

        # Capture tile info with GPS coordinates
        tile_info = {
            "site_id": site_name,
            "index": entry["image"].get("index"),
            "origin_x": entry["image"]["pixel_origin"][0],
            "origin_y": entry["image"]["pixel_origin"][1],
            "tile_url": tile_url,
            "tile_preview": tile_preview,
            # Add GPS coordinates for tiles
            "bounds": entry["image"].get("bounds"),
            "lon": entry["image"].get("center")["lon"],
            "lat": entry["image"].get("center")["lat"],
            "crs": entry["image"].get("crs"),
            "size": entry["image"].get("size")
        }
        tiles.append(tile_info)

        # Capture site info only once with GPS coordinates
        if site_name not in sites_dict:
            map_url = parent_image.get("source")
            preview_url = parent_image.get("preview")

            site_info = {
                "site_id": site_name,
                "map_url": map_url,
                "preview_url": preview_url,
                # Add GPS coordinates for sites
                "bounds": parent_image.get("bounds"),
                "lon": parent_image.get("center")["lon"],
                "lat": parent_image.get("center")["lat"],
                "crs": parent_image.get("crs")
            }

            # Add map metrics stats if available for this site
            if site_name in map_metrics_lookup:
                site_stats = map_metrics_lookup[site_name]
                # Add each stat as a separate column
                for stat_name, stat_value in site_stats.items():
                    site_info[stat_name] = stat_value

            sites_dict[site_name] = site_info

    tiles_df = pd.DataFrame(tiles)
    sites_df = pd.DataFrame(sites_dict.values())

    # Use xlsxwriter engine which handles long URLs better
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        sites_df.to_excel(writer, sheet_name="sites", index=False)
        tiles_df.to_excel(writer, sheet_name="tiles", index=False)

        # Get the workbook
        workbook = writer.book

        # Set text format for URL columns to prevent Excel from interpreting them as formulas
        text_format = workbook.add_format({'num_format': '@'})

        # Apply text format to URL columns in both sheets
        for sheet_name in ['sites', 'tiles']:
            worksheet = writer.sheets[sheet_name]

            # Find URL columns and apply text format
            for col_num, col_name in enumerate(tiles_df.columns if sheet_name == 'tiles' else sites_df.columns):
                if 'url' in col_name.lower():
                    worksheet.set_column(
                        col_num, col_num, 50, text_format)  # Set width to 50
                else:
                    worksheet.set_column(col_num, col_num, 20)  # Default width

    print(f"Excel file saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offloaded_metadata", required=True,
                        help="Path to offloaded_tile_metadata.json")
    parser.add_argument("--map_metrics", required=True,
                        help="Path to map_metrics.json")
    parser.add_argument("--output_excel", required=True,
                        help="Path to output Excel file (.xlsx)")
    args = parser.parse_args()

    convert_json_to_excel(args.offloaded_metadata,
                          args.map_metrics, args.output_excel)
