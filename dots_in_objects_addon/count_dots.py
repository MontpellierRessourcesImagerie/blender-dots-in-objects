import bpy
import mathutils
import json
import csv
import io
import numpy as np

from nd_co_occurrences.graph import (
    build_graph, 
    search_co_occurrences,
    count_combinations,
    counted_combinations_to_csv
)

def get_object_collection():
    """
    Probes the scene to find the collection containing the nuclei.
    The nuclei collection's name must start with "nuclei" and contain only meshes.
    Returns the collection object, not its name
    """
    for c in bpy.data.collections:
        if not c.name.lower().startswith("nuclei"):
            continue
        if any([o.type != 'MESH' for o in c.objects]):
            continue
        return c
    return None

def get_dots_collections():
    """
    Probes the scene to find all the collections containing spots.
    A collection of spot must have a name starting with "spots" and contain only empties.
    Returns the list of collection objects, not their names
    """
    dots_collections = []
    for c in bpy.data.collections:
        if not c.name.lower().startswith("dots-"):
            continue
        if any([o.type != 'EMPTY' for o in c.objects]):
            continue
        dots_collections.append(c)
    return dots_collections

def get_total_n_vertices(collection):
    """
    Returns the total number of vertices in the objects collection.
    Used to allocate the memory used by the KDTree.
    """
    if collection is None:
        raise ValueError("The objects collection cannot be None")
    n_vertices = [len(obj.data.vertices) for obj in collection.objects if obj.type == 'MESH']
    return sum(n_vertices)
    

def build_kd_tree(objects_collection):
    """
    Builds a KD-Tree containing the vertices of all objects present in the collection.
    Returns the KD Tree, and the array of correspondance with objects.
    """
    if objects_collection is None:
        raise ValueError("The collection containing objects is missing.")
    
    ttl_n_vertices = get_total_n_vertices(objects_collection)
    kd = mathutils.kdtree.KDTree(ttl_n_vertices)
    vertex_to_object_map = [] # So we can retrieve the closest object
    
    for obj in objects_collection.objects:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        matrix = obj.matrix_world
        for vert in mesh.vertices:
            kd.insert(matrix @ vert.co, len(vertex_to_object_map))
            vertex_to_object_map.append(obj)
    
    kd.balance()
    return kd, vertex_to_object_map

def get_dots_per_channel():
    all_spots = {}
    collections = get_dots_collections()
    p = "dots-"
    for collection in collections:
        name = collection.name[len(p):]
        locations = [(tuple(obj.location), obj) for obj in collection.objects]
        all_spots[name] = locations
    return all_spots

def count_dots(objects_collection):
    all_spots = get_dots_per_channel()
    template = {c: 0 for c in all_spots.keys()}
    accumulator = {n.name: template.copy() for n in objects_collection.objects}
    for ch_name, spots in all_spots.items():
        for (_, spot) in spots:
            if spot.parent is None:
                continue
            name = spot.parent.name
            accumulator[name][ch_name] += 1
    return accumulator

def counters_dict_to_csv(data, field_order=None, include_header=True):
    """
    Convert {name: {metric: value, ...}, ...} to a CSV string.
    Headers: "Nucleus" + all inner keys (or field_order if provided).
    """
    if field_order is None:
        fields = sorted({k for inner in data.values() for k in inner.keys()})
    else:
        fields = list(field_order)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    if include_header:
        writer.writerow(["Nucleus", *fields])

    for nucleus, inner in data.items():  # preserves input order
        writer.writerow([nucleus] + [inner.get(k, "") for k in fields])

    return buf.getvalue()

def new_text_in_editor(contents="", name="dots-per-object.csv"):
    txt = bpy.data.texts.new(name=name)
    if contents:
        txt.from_string(contents)
    #_show_in_text_editor(txt)
    return txt

def dots_to_closest_object(collection):
    """
    Loops through the spots (empties) and searches for the closest vertex.
    Sets the parent of each spot to its owner nuclei.
    Counts the number of spots per nuclei
    """
    kd, mapping = build_kd_tree(collection)
    all_spots = get_dots_per_channel()
    
    for dots_channel, spots in all_spots.items():
        print(f"Processing spots channel: {dots_channel}")
        for s_co, empty in spots:
            empty.parent = None
            co, index, _ = kd.find(s_co)
            closest_object = mapping[index]
            v1 = mathutils.Vector(co) - closest_object.location # origin to surface point
            v2 = empty.location - mathutils.Vector(co) # vertex to empty
            v1.normalize()
            v2.normalize()
            inside = v1.dot(v2) <= 0
            if not inside:
                continue
            empty.parent = closest_object
            empty.matrix_parent_inverse = closest_object.matrix_world.inverted()

def remove_dots_outside_objects():
    dots_collections = get_dots_collections()
    for collection in dots_collections:
        all_dots = [obj for obj in collection.objects if obj.type == 'EMPTY']
        counter = 0
        for obj in all_dots:
            if obj.parent is None:
                bpy.data.objects.remove(obj)
                counter += 1
        print(f"Removed {counter} dots from collection {collection.name}")

def dots_per_object(collection):
    acc = count_dots(collection)
    csv_txt = counters_dict_to_csv(acc)
    new_text_in_editor(csv_txt, name="dots-per-object.csv")
    print("Results available in: scripting/dots-per-object.csv")

def build_control_meshes(co_occ_levels, all_spots, graph, collection):
    for level, tuples in co_occ_levels.items():
        if level < 2:
            continue
        for tuple_idx, vertex_tuple in enumerate(tuples):
            # Create a new mesh
            mesh = bpy.data.meshes.new(f"co-occ-level{level}-tuple{tuple_idx}")
            
            # Extract 3D coordinates for vertices
            verts = []
            vertex_map = {}  # Maps graph vertex index to mesh vertex index
            for i, graph_vertex in enumerate(vertex_tuple):
                spot_idx, pc_idx = graph_vertex
                coord = all_spots[pc_idx][spot_idx]
                verts.append(coord)
                vertex_map[graph_vertex] = i
            
            # Build edges from graph
            edges = []
            for v1 in vertex_tuple:
                if v1 in graph:
                    for v2 in graph[v1]:
                        v2_idx = vertex_map.get(v2)
                        v1_idx = vertex_map.get(v1)
                        if v2_idx is not None and v1_idx is not None:
                            edges.append((v1_idx, v2_idx))
            
            # Create mesh geometry
            mesh.from_pydata(verts, edges, [])
            mesh.update()
            
            # Create object and link to collection
            obj = bpy.data.objects.new(f"co-occ-level{level}-tuple{tuple_idx}", mesh)
            collection.objects.link(obj)


def count_co_occurrences(max_dist, per_obj, obj_collection):
    collections = get_dots_collections()
    all_spots = []
    all_owners = []
    owners_pool = set([np.int64(o.name.replace("obj-", "")) for o in obj_collection.objects]) if obj_collection is not None else None
    
    for collection in collections:
        spots = [obj.location for obj in collection.objects if obj.type == 'EMPTY']
        owners = [np.int64(obj.parent.name.replace("obj-", "")) if obj.parent is not None else 0 for obj in collection.objects if obj.type == 'EMPTY']
        all_spots.append(np.array(spots))
        all_owners.append(np.array(owners))
    
    graph = build_graph(
        all_spots, 
        max_dist, 
        all_owners if per_obj else None
    )
    co_occ_levels = search_co_occurrences(
        graph, 
        len(collections), 
        all_spots
    )
    counter = count_combinations(
        co_occ_levels, 
        len(collections),
        all_owners if per_obj else None,
        owners_pool if per_obj else None
    )
    csv_txt = counted_combinations_to_csv(counter)
    new_text_in_editor(csv_txt, name="counted-co-occurrences.csv")
    
    co_occ_collection = bpy.data.collections.get("co-occs-control")
    if co_occ_collection is None:
        co_occ_collection = bpy.data.collections.new("co-occs-control")
        bpy.context.scene.collection.children.link(co_occ_collection)

    build_control_meshes(co_occ_levels, all_spots, graph, co_occ_collection)