import uuid
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def new_uuid() -> str:
    """Return a compact UUID string (hex, no dashes)."""
    return uuid.uuid4().hex

def aggregate_solid_geometries(child_ids, city_objects):
    solids = []
    for cid in child_ids:
        child = city_objects.get(cid)
        if not child or "geometry" not in child:
            continue
        for geom in child["geometry"]:
            if geom["type"] == "Solid":
                solids.append(geom)
    return solids if solids else None


def convert_to_json_serializable(obj: Any):
    """Recursively cast numpy dtypes → native Python so json.dump() works."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
#  CityJSON builder (BIM-Legal mapping)
# ---------------------------------------------------------------------------
def remove_consecutive_duplicates(coords):
    return [pt for i, pt in enumerate(coords) if i == 0 or pt != coords[i - 1]]

def build_cityjson_model(rooms_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Return a CityJSON dict using the hierarchy requested by the user:

        BIMLegalApartmentComplex   → CityObjectGroup
        ApartmentComplex           → Building
        ApartmentUnit              → BuildingPart
        SharedPart                 → single Room (container) with isSharedPart = true
        BIMLegalSpaceUnit          → Room (leaf) with geometry
    """
    # ------------------------------------------------------------------
    #  Base CityJSON scaffold
    # ------------------------------------------------------------------

    cj: Dict[str, Any] = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {},
        "vertices": [],
        "metadata": {
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/28992"
        },
    }

    # ------------------------------------------------------------------
    #  Vertex store with deduplication
    # ------------------------------------------------------------------
    vertices: list[list[float]] = []
    vertex_index: Dict[tuple, int] = {}

    def add_vertex(coord):
        """Add unique vertex (scaled to int), return its index, and print it for debugging."""
        scaled = tuple(int(round(c * 10000)) for c in coord)
        if scaled not in vertex_index:
            idx = len(vertices)
            vertex_index[scaled] = idx
            vertices.append(list(scaled))
            print(f"Vertex {idx}: {scaled}")
        return vertex_index[scaled]

    # ------------------------------------------------------------------
    #  Top-level: BIMLegalApartmentComplex → CityObjectGroup
    # ------------------------------------------------------------------
    group_uuid = new_uuid()
    cj["CityObjects"][group_uuid] = {
        "type": "CityObjectGroup",
        "attributes": {
            "name": "BIMLegalApartmentComplex"
        },
        "children": []  # Building(s) will be appended here
    }

    # ------------------------------------------------------------------
    #  Building (ApartmentComplex) – We assume one complex for now; extend if
    #  you have multiple by grouping on a column.
    # ------------------------------------------------------------------
    building_uuid = new_uuid()
    cj["CityObjects"][building_uuid] = {
        "type": "Building",
        "attributes": {
            "apartmentComplexIndex": "AC-01"  # customise if you have a field
        },
        "children": [],  # units + shared container
        "parents": [group_uuid]
    }
    cj["CityObjects"][group_uuid]["children"].append(building_uuid)

    # ------------------------------------------------------------------
    #  Prepare data – numeric apartment index
    # ------------------------------------------------------------------
    rooms_gdf = rooms_gdf.copy()
    rooms_gdf["apartment"] = pd.to_numeric(rooms_gdf["appartement"], errors="coerce")

    # ------------------------------------------------------------------
    #  Containers:   BuildingPart for each ApartmentUnit
    # ------------------------------------------------------------------
    apartment_units: Dict[int, Dict[str, Any]] = {}
    for idx in rooms_gdf["apartment"].dropna().astype(int).unique():
        unit_uuid = new_uuid()
        apartment_units[idx] = {
            "uuid": unit_uuid,
            "children": []
        }
        cj["CityObjects"][unit_uuid] = {
            "type": "BuildingPart",
            "attributes": {
                "apartmentIndex": str(idx)
            },
            "children": apartment_units[idx]["children"],
            "parents": [building_uuid]
        }
        # Register under Building
        cj["CityObjects"][building_uuid]["children"].append(unit_uuid)

    # ------------------------------------------------------------------
    #  SharedPart container – one Room that groups all shared rooms
    # ------------------------------------------------------------------
    shared_room_container_uuid = new_uuid()
    cj["CityObjects"][shared_room_container_uuid] = {
        "type": "BuildingPart",
        "attributes": {
            "name": "SharedPart",
            "isSharedPart": True
        },
        "children": [],
        "parents": [building_uuid]

    }
    # We add it regardless; if it ends up empty, we can prune later
    cj["CityObjects"][building_uuid]["children"].append(shared_room_container_uuid)

    # ------------------------------------------------------------------
    #  Iterate rooms → leaf Rooms with geometry
    # ------------------------------------------------------------------
    for idx, row in rooms_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue

        room_uuid = new_uuid()

        # ---------- Geometry handling ---------------------------------
        solid_faces: list[list[int]] = []

        if hasattr(geom, "exterior"):
            coords_base = remove_consecutive_duplicates(list(geom.exterior.coords))

            # Do NOT close the polygon
            if coords_base[0] == coords_base[-1]:
                coords_base = coords_base[:-1]

            coords_top = [(x, y, z + row["extrusion_height"]) for x, y, z in coords_base]

            bottom_face = [add_vertex(c) for c in coords_base]
            top_face = [add_vertex(c) for c in reversed(coords_top)]

            wall_faces = []
            for i in range(len(coords_base)):  # original base without closure
                p1_base = coords_base[i]
                p2_base = coords_base[(i + 1) % len(coords_base)]  # wrap-around
                p1_top = coords_top[i]
                p2_top = coords_top[(i + 1) % len(coords_top)]
                wall_faces.append([add_vertex(p1_base), add_vertex(p2_base),
                                   add_vertex(p2_top), add_vertex(p1_top)])

            solid_faces = [bottom_face, top_face] + wall_faces

        elif isinstance(geom, list):
            # Pre-built face list [[(x,y,z),...], ...]
            for face in geom:
                solid_faces.append([add_vertex(tuple(face)) for face in face])
        else:
            raise TypeError(f"Unsupported geometry type for row {idx}: {type(geom)}")

        cj["CityObjects"][room_uuid] = {
            "type": "BuildingRoom",
            "attributes": {
                "name": row["ruimte"],
                "extrusion_height": row.get("extrusion_height"),
                "isSharedPart": pd.isna(row["apartment"])
            },
            "geometry": [{
                "type": "MultiSurface",
                "lod": "1",
                "boundaries": [ [face] for face in solid_faces ]
            }]
        }

        # ---------- Attach to parent ----------------------------------
        if pd.notna(row["apartment"]):
            parent_dict = apartment_units[int(row["apartment"])]
        else:
            parent_dict = cj["CityObjects"][shared_room_container_uuid]

        parent_dict["children"].append(room_uuid)

        if pd.notna(row["apartment"]):
            parent_uuid = apartment_units[int(row["apartment"])]["uuid"]
        else:
            parent_uuid = shared_room_container_uuid
        cj["CityObjects"][room_uuid]["parents"] = [parent_uuid]

    # ------------------------------------------------------------------
    #  Prune shared container if empty
    # ------------------------------------------------------------------
    if not cj["CityObjects"][shared_room_container_uuid]["children"]:
        cj["CityObjects"].pop(shared_room_container_uuid)
        cj["CityObjects"][building_uuid]["children"].remove(shared_room_container_uuid)


    # ------------------------------------------------------------------
    cj["vertices"] = vertices
    cj["transform"] = {
        "scale": [0.0001, 0.0001, 0.0001],
        "translate": [0.0, 0.0, 0.0]  # Optional: adjust if you apply a spatial offset
    }

    return cj


# ---------------------------------------------------------------------------
#  Export helper
# ---------------------------------------------------------------------------

def export_to_cityjson(rooms_gdf: gpd.GeoDataFrame, out_path: Path | str) -> None:
    cj = build_cityjson_model(rooms_gdf)
    cj = convert_to_json_serializable(cj)
    out_path = Path(out_path)
    for i, v in enumerate(cj["vertices"]):
        if len(v) != 3:
            print(f"Invalid vertex at index {i}: {v}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cj, f, indent=2)
    print(f"CityJSON written ➜ {out_path.resolve()}")


# ------------------------------------------------------------------------------
# 5.  Example usage
# ------------------------------------------------------------------------------
# Replace empty strings and whitespace with NaN, then drop or handle them
# extruding the floors to the calculated heights

from pathlib import Path

input_folder = Path("gpkg_outputs")
for gpkg_file in input_folder.glob("*.gpkg"):
    pand_data = gpd.read_file(gpkg_file)
    print(f"Processing {gpkg_file.name}")
    print(pand_data.info())

    output_file = gpkg_file.with_suffix('.city.json').with_name(f"{gpkg_file.stem}.city.json")
    export_to_cityjson(pand_data, output_file)