from rasterio.io import MemoryFile
import requests


def get_elevation_at_point(x, y, coverage_id="dtm_05m"):
    """
    Get elevation at a specific point from PDOK AHN WCS service

    Parameters:
    - x: RD x-coordinate (EPSG:28992)
    - y: RD y-coordinate (EPSG:28992)
    - coverage_id: coverage identifier (e.g., 'dsm_05m', 'dtm_05m')

    Returns:
    - Elevation in meters (float), or 0.0 if elevation cannot be retrieved
    """

    # Create a small bounding box around the point (e.g., 10x10 meters)
    buffer = 5  # meters
    bbox = f"{x-buffer},{y-buffer},{x+buffer},{y+buffer}"

    # Build WCS GetCoverage request
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": coverage_id,
        "CRS": "EPSG:28992",
        "BBOX": bbox,
        "WIDTH": "10",
        "HEIGHT": "10",
        "FORMAT": "GeoTIFF",
    }

    # Make request
    response = requests.get("https://service.pdok.nl/rws/ahn/wcs/v1_0", params=params)

    if response.status_code == 200:
        # Read elevation from GeoTIFF
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                # Get the elevation value at the center of the raster
                elevation_array = dataset.read(1)
                center_idx = elevation_array.shape[0] // 2
                elevation = elevation_array[center_idx, center_idx]

                # Check for NoData values
                if dataset.nodata is not None and elevation == dataset.nodata:
                    return 0.0  # Return 0.0 instead of None

                return float(elevation)
    else:
        # Return 0.0 instead of None when API call fails
        return 0.0


# Try the functions
if __name__ == "__main__":
    # Example RD coordinates (x, y)
    rd_x = 131859.3354
    rd_y = 507540.5984

    elevation = get_elevation_at_point(rd_x, rd_y, coverage_id="dtm_05m")
    if elevation is not None:
        print(f"Elevation at point ({rd_x}, {rd_y}): {elevation:.2f} m")
    else:
        print(f"Could not retrieve elevation for point ({rd_x}, {rd_y})")
