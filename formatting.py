import ifcopenshell.api
import geopandas as gpd
from shapely import wkt
from shapely.geometry import mapping
import os
import ifcopenshell.api.root
import ifcopenshell.api.unit
import ifcopenshell.api.context
import ifcopenshell.api.project
import ifcopenshell.api.spatial
import ifcopenshell.api.geometry
import ifcopenshell.api.aggregate

gdf = gpd.read_file(r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\separate_pand_rooms.csv")  # or construct it another way
print(gdf.info())
print(gdf.head(5))

import geopandas as gpd
import ifcopenshell
import ifcopenshell.api
import os

from shapely.geometry import Polygon, MultiPolygon
import ifcopenshell.api
import ifcopenshell.util.element
from shapely.geometry import Polygon, MultiPolygon
import ifcopenshell
import ifcopenshell.api

def make_faceted_brep(model, context, geom):
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise ValueError("Expected Polygon or MultiPolygon")

    faces = []
    for poly in polygons:
        if not poly.exterior:
            continue

        # Outer loop
        outer_pts = [model.create_entity("IfcCartesianPoint", Coordinates=list(pt)) for pt in poly.exterior.coords]
        polyloop = model.create_entity("IfcPolyLoop", Polygon=outer_pts)
        outer_bound = model.create_entity("IfcFaceOuterBound", Bound=polyloop, Orientation=True)

        # Inner loops
        inner_bounds = []
        for hole in poly.interiors:
            inner_pts = [model.create_entity("IfcCartesianPoint", Coordinates=list(pt)) for pt in hole.coords]
            inner_loop = model.create_entity("IfcPolyLoop", Polygon=inner_pts)
            inner_bound = model.create_entity("IfcFaceBound", Bound=inner_loop, Orientation=True)
            inner_bounds.append(inner_bound)

        face = model.create_entity("IfcFace", Bounds=[outer_bound] + inner_bounds)
        faces.append(face)

    shell = model.create_entity("IfcClosedShell", CfsFaces=faces)
    brep = model.create_entity("IfcFacetedBrep", Outer=shell)

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="Brep",
        Items=[brep],
    )

    return shape_rep


def create_faceted_brep(model, context, geom):
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")

    faces = []
    for poly in polygons:
        if not poly.is_valid:
            continue

        # Convert exterior ring
        outer = [(x, y, z) for x, y, z in poly.exterior.coords]
        outer_loop = ifcopenshell.api.run("geometry.add_polyloop", model, points=outer)

        # Convert inner rings (holes)
        inner_loops = []
        for interior in poly.interiors:
            inner = [(x, y, z) for x, y, z in interior.coords]
            loop = ifcopenshell.api.run("geometry.add_polyloop", model, points=inner)
            inner_loops.append(loop)

        face = ifcopenshell.api.run("geometry.add_face", model, outer=outer_loop, inners=inner_loops)
        faces.append(face)

    shell = ifcopenshell.api.run("geometry.add_closed_shell", model, faces=faces)
    brep = ifcopenshell.api.run("geometry.add_faceted_brep", model, shell=shell)

    # Wrap in a representation
    representation = ifcopenshell.api.run("geometry.add_representation", model,
        context=context,
        items=[brep],
        representation_type="Brep",
        representation_identifier="Body"
    )
    return representation


# Iterate over each unique bag_pnd (one project per bag_pnd)
for bag_pnd, rooms in gdf.groupby("bag_pnd"):
    # Create a new model
    model = ifcopenshell.api.project.create_file()

    # Create core project structure
    project = ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject", name=f"Project_{bag_pnd}")
    ifcopenshell.api.unit.assign_unit(model)

    context = ifcopenshell.api.context.add_context(model, context_type="Model")
    body = ifcopenshell.api.context.add_context(model, context_type="Model",
                                                context_identifier="Body",
                                                target_view="MODEL_VIEW",
                                                parent=context)

    site = ifcopenshell.api.root.create_entity(model, ifc_class="IfcSite", name=f"Site_{bag_pnd}")
    building = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuilding", name=f"Building_{bag_pnd}")
    storey = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingStorey", name="Ground Floor")

    ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[building])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=building, products=[storey])

    for _, room in rooms.iterrows():
        geom = room["optimized_rooms_3d"]

        # Ensure valid 3D geometry
        try:
            geom = wkt.loads(room["optimized_rooms_3d"])
        except Exception as e:
            print(f"Error parsing geometry for room {room['room_id']}: {e}")
            continue

        if not geom.has_z:
            print(f"Skipping room without 3D geometry: {room['room_id']}")
            continue

        # Create IfcSpace
        space = ifcopenshell.api.run("root.create_entity", model, {
            "ifc_class": "IfcSpace",
            "name": f"Room_{room['ruimte']}",
        })

        # Assign to storey
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            products=[space],
            relating_structure=storey
        )

        # Default placement (origin)
        ifcopenshell.api.run(
            "geometry.edit_object_placement",
            model,
            product=space
        )

        # Generate IFC geometry from shapely 3D polygon
        try:
            shape = make_faceted_brep(model, body, geom)
            ifcopenshell.api.run("geometry.assign_representation", model,
                                 product=space,
                                 representation=shape,
                                 )
        except Exception as e:
            print(f"Failed to create faceted brep for room {room['ruimte']}: {e}")

    # Save the file
    filename = os.path.join("werkmap", f"project_{bag_pnd}.ifc")
    model.write(filename)
    print(f"Saved IFC for {bag_pnd} to {filename}")
