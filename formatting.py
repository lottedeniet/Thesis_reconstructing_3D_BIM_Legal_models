import ifcopenshell.api
import geopandas as gpd
import shapely
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
from matplotlib import pyplot as plt

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


from shapely.geometry import Polygon, MultiPolygon


from shapely.geometry import Polygon, MultiPolygon
import ifcopenshell




from ifcopenshell.api import run

def create_cartesian_point(model, coords):
    # Create IfcCartesianPoint and assign coordinate aggregate manually
    point = model.create_entity("IfcCartesianPoint")
    point.__dict__["wrapped_data"].set_argument("Coordinates", coords)
    return point

def create_direction(model, ratios):
    direction = model.create_entity("IfcDirection")
    direction.__dict__["wrapped_data"].set_argument("DirectionRatios", ratios)
    return direction

def create_local_placement(model, x=0.0, y=0.0, z=0.0):
    point = model.create_entity("IfcCartesianPoint", Coordinates=((x, y, z)))
    dir_z = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    dir_x = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))

    axis2placement = model.create_entity(
        "IfcAxis2Placement3D",
        Location=point,
        Axis=dir_z,
        RefDirection=dir_x
    )

    return model.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=None,
        RelativePlacement=axis2placement
    )




def make_extruded_brep(model, context, geom, height, z_offset):
    # Ensure the geometry is a Polygon
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise ValueError("Expected Polygon or MultiPolygon")

    # Create a profile for the extrusion
    profile = model.create_entity("IfcArbitraryClosedProfileDef",
                                  ProfileType="AREA",
                                  OuterCurve=model.create_entity("IfcPolyline", Points=[model.create_entity("IfcCartesianPoint", Coordinates=list(pt)) for pt in polygons[0].exterior.coords]))

    # Extrude the profile
    extruded_solid = model.create_entity("IfcExtrudedAreaSolid",
                                         SweptArea=profile,
                                         ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=[0.0, 0.0, 1.0]),
                                         Depth=height)

    # Create the shape representation
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[extruded_solid],
    )

    # Return the shape representation with the offset applied
    print(z_offset+ height)
    return shape_rep, z_offset + height


# Before grouping
gdf["floor_index"] = gdf["floor_index"].astype(int)

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
    ifcopenshell.api.aggregate.assign_object(model, relating_object=project, products=[site])
    ifcopenshell.api.aggregate.assign_object(model, relating_object=site, products=[building])

    storeys = {}
    z_offset = 0.0  # Initialize Z offset for stacking floors

    for floor_idx, floor_rooms in rooms.groupby("floor_index"):
        print(rooms["floor_index"].value_counts())

        storey = ifcopenshell.api.root.create_entity(model, ifc_class="IfcBuildingStorey", name=f"Storey_{floor_idx}")
        ifcopenshell.api.aggregate.assign_object(model, relating_object=building, products=[storey])
        storeys[floor_idx] = storey
        floor_height = floor_rooms["extrusion_height"].astype(float).max()
        # Iterate over each room and create extrusion
        for _, room in floor_rooms.iterrows():
            print(room['ruimte'])
            try:
                geom = wkt.loads(room["optimized_rooms_3d"])  # Load the geometry (WKT)
                print(f"Room {room['ruimte']} | Floor {floor_idx} | Geom hash: {hash(geom.wkt)}")
            except Exception as e:
                print(f"Error parsing geometry for room {room['room']}: {e}")
                continue

            if not geom.has_z:  # Skip if no 3D geometry
                print(f"Skipping room without 3D geometry: {room['room']}")
                continue

            # Create IfcSpace for room
            space = ifcopenshell.api.run("root.create_entity", model, {
                "ifc_class": "IfcSpace",
                "name": f"Room_{room['ruimte']}",
            })

            # Get extrusion height
            extrusion_height = float(room["extrusion_height"])  # Assuming this is in your DataFrame
            print(extrusion_height)

            # Create the faceted BREP with the updated Z offset
            print(f"Room: {room['ruimte']}, Geometry: {geom.wkt[:60]}..., Height: {extrusion_height}")

            try:
                rep, new_z_offset = make_extruded_brep(model, body, geom, extrusion_height, z_offset)

            except Exception as e:
                print(f"Error creating extruded solid for room {room['ruimte']}: {e}")
                continue

            # Assign representation to space
            space.Representation = rep
            print(rep.ContextOfItems)
            print(rep.Items)

            # Assign to spatial structure and placement
            ifcopenshell.api.run("spatial.assign_container", model,
                                 products=[space],
                                 relating_structure=storeys[room["floor_index"]])

            space.ObjectPlacement = create_local_placement(model, 0.0, 0.0, z_offset)
            print(f"Added space {space.Name} to storey {room['floor_index']}")
            print(f"Created {space.is_a()} with name {space.Name}")

        z_offset += float(floor_height)

# Ensure the output directory exists
output_dir = "werkmap"
os.makedirs(output_dir, exist_ok=True)


# Plot (only X/Y will be shown, Z is ignored)

# Save the file
filename = os.path.join(output_dir, f"project_{bag_pnd}.ifc")
model.write(filename)
print(f"Saved IFC for {bag_pnd} to {filename}")