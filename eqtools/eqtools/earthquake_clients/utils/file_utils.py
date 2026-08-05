import os
import json
from pathlib import Path

from ...gmttools import read_gmt_lines


_EARTHQUAKE_CLIENTS_DIR = Path(__file__).resolve().parents[1]


def _package_data_dir(data_dir):
    """Return a package-relative data directory without requiring it to exist."""
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return _EARTHQUAKE_CLIENTS_DIR / path


def get_data_dir_path(data_dir):
    """Resolve packaged data first, then an optional working-directory override.

    Missing optional data are represented by the expected package path.  The
    listing helpers handle that path as an empty directory instead of raising.
    """
    resource_dir = _package_data_dir(data_dir)
    if resource_dir.is_dir():
        return str(resource_dir)

    local_dir = Path(data_dir)
    if local_dir.is_dir():
        return str(local_dir.resolve())

    return str(resource_dir)

def find_files_in_directory(directory, extensions):
    """Find matching files, returning an empty list for a missing directory."""
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.suffix in extensions
    )

def find_and_prioritize_files(data_dir, extensions, selected_files=None):
    """
    Find and prioritize files in the resource and local directories.
    If a file exists in both directories, the local file is prioritized.
    If a file with the same name but different extension exists in both directories, the .json file is prioritized.
    If selected_files is None, find all files with the specified extensions.
    Otherwise, prioritize based on the selected_files list.
    """
    resource_dir = str(_package_data_dir(data_dir))
    local_dir = str(Path(data_dir).resolve())

    if selected_files is None:
        resource_files = find_files_in_directory(resource_dir, extensions)
        local_files = find_files_in_directory(local_dir, extensions)
    else:
        resource_files = [f for f in selected_files if os.path.exists(os.path.join(resource_dir, f))]
        local_files = [f for f in selected_files if os.path.exists(os.path.join(local_dir, f))]
    
    extension_rank = {".json": 0, ".geojson": 0, ".gmt": 1}

    def preferred(files, directory):
        choices = {}
        for filename in files:
            name, extension = os.path.splitext(filename)
            candidate = os.path.join(directory, filename)
            rank = extension_rank.get(extension.lower(), 99)
            current = choices.get(name)
            if current is None or rank < current[0]:
                choices[name] = (rank, candidate)
        return {name: candidate for name, (_, candidate) in choices.items()}

    resource_choices = preferred(resource_files, resource_dir)
    local_choices = preferred(local_files, local_dir)
    if selected_files is None:
        names = sorted(set(resource_choices) | set(local_choices))
    else:
        names = []
        for filename in selected_files:
            name = os.path.splitext(filename)[0]
            if name not in names:
                names.append(name)
    return [
        local_choices.get(name, resource_choices.get(name))
        for name in names
        if name in local_choices or name in resource_choices
    ]

def load_geojson_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading GeoJSON data from {file_path}: {e}")
        return None

def create_geojson_from_gmt(gmt_path, json_path, use_ogr=None):
    import json
    from osgeo import ogr

    def use_ogr_conversion():
        try:
            # Open GMT file
            gmt_ds = ogr.Open(gmt_path)
            if gmt_ds is None:
                raise IOError(f"Cannot open GMT file {gmt_path}")

            # Create GeoJSON file
            driver = ogr.GetDriverByName('GeoJSON')
            if driver is None:
                raise IOError("GeoJSON driver is not available")

            geojson_ds = driver.CreateDataSource(json_path)
            if geojson_ds is None:
                raise IOError(f"Cannot create GeoJSON file {json_path}")

            # Copy layers
            for i in range(gmt_ds.GetLayerCount()):
                layer = gmt_ds.GetLayerByIndex(i)
                geojson_ds.CopyLayer(layer, layer.GetName())

            # Close data sources
            gmt_ds = None
            geojson_ds = None

            # Read the generated GeoJSON file
            with open(json_path, 'r') as f:
                geojson_data = json.load(f)

            return geojson_data
        except Exception as e:
            print(f"Error using OGR to create GeoJSON data from GMT file {gmt_path}: {e}")
            return None

    def use_custom_conversion():
        try:
            segments = read_gmt_lines(gmt_path)
            geojson_data = {
                "type": "FeatureCollection",
                "features": []
            }
            for segment in segments:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": list(zip(segment['X'], segment['Y']))
                    },
                    "properties": {}
                }
                geojson_data["features"].append(feature)
            with open(json_path, 'w') as f:
                json.dump(geojson_data, f)
            return geojson_data
        except IOError as e:
            print(f"Error creating GeoJSON data from GMT file {gmt_path}: {e}")
            return None

    if use_ogr is None:
        # Try using OGR for conversion
        geojson_data = use_ogr_conversion()
        if geojson_data is not None:
            return geojson_data
        # If OGR conversion fails, use custom logic
        return use_custom_conversion()
    elif use_ogr:
        # Force using OGR for conversion
        return use_ogr_conversion()
    else:
        # Force using custom logic for conversion
        return use_custom_conversion()
