import bpy
import mathutils
import json
import csv
import io

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
    spots_collections = []
    for c in bpy.data.collections:
        if not c.name.lower().startswith("spots"):
            continue
        if any([o.type != 'EMPTY' for o in c.objects]):
            continue
        spots_collections.append(c)
    return spots_collections

def get_total_vertices(collection):
    """
    Returns the total number of vertices in the nuclei collection.
    Used to allocate the memory used by the KDTree.
    """
    if collection is None:
        raise ValueError("The nuclei collection cannot be None")
    acc = 0
    n_vertices = [len(obj.data.vertices) for obj in collection.objects]
    return sum(n_vertices)
    

def build_kd_tree():
    """
    Builds a KD-Tree containing the vertices of all nuclei present in the collection.
    Returns the KD Tree, and the array of correspondance with objects.
    """
    nuclei_collection = get_object_collection()
    if nuclei_collection is None:
        raise ValueError("The collection containing nuclei is missing.")
    ttl_vertices = get_total_vertices(nuclei_collection)
    kd = mathutils.kdtree.KDTree(ttl_vertices)
    vertex_to_object_map = [] # So we can retrieve the closest object
    normals = [] # Vertices normals, for I/O test
    
    for obj in nuclei_collection.objects:
        mesh = obj.data
        matrix = obj.matrix_world
        for i, vert in enumerate(mesh.vertices):
            normals.append(vert.normal)
            kd.insert(matrix @ vert.co, len(vertex_to_object_map))
            vertex_to_object_map.append(obj)
    
    kd.balance()
    return kd, vertex_to_object_map, normals

def get_dots_per_channel():
    all_spots = {}
    collections = get_dots_collections()
    p = "spots_"
    for collection in collections:
        name = collection.name[len(p):]
        locations = [(tuple(obj.location), obj) for obj in collection.objects]
        all_spots[name] = locations
    return all_spots

def count_dots():
    all_spots = get_dots_per_channel()
    nuclei_collection = get_object_collection()
    template = {c: 0 for c in all_spots.keys()}
    accumulator = {n.name: template.copy() for n in nuclei_collection.objects}
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

def new_text_in_editor(contents="", name="results.csv"):
    txt = bpy.data.texts.new(name=name)
    if contents:
        txt.from_string(contents)
    #_show_in_text_editor(txt)
    return txt

def dots_to_closest_object():
    """
    Loops through the spots (empties) and searches for the closest vertex.
    Sets the parent of each spot to its owner nuclei.
    Counts the number of spots per nuclei
    """
    kd, mapping, normals = build_kd_tree()
    all_spots = get_dots_per_channel()
    
    for spots_channel, spots in all_spots.items():
        print(f"Processing spots channel: {spots_channel}")
        for rank, (s_co, empty) in enumerate(spots):
            empty.parent = None
            co, index, dist = kd.find(s_co)
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

def dots_per_object():
    dots_to_closest_object()
    acc = count_dots()
    csv_txt = counters_dict_to_csv(acc)
    new_text_in_editor(csv_txt)
    print("Results available in: scripting/results.csv")

if __name__ == "__main__":
    dots_per_object()

