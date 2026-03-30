import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from random import shuffle

def random_colors(n):
    """
    Generates a list of N colors in hexadecimal format.
    The output format must be compatible with matplotlib's color specifications.
    """
    colors = []
    for _ in range(n):
        color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF + 1))
        colors.append(color)
    return colors

def generate_2d_points_cloud(n_items, width, height):
    points = np.random.rand(n_items, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    return points

def generate_3d_points_cloud(n_items, width, height, depth):
    points = np.random.rand(n_items, 3)
    points[:, 0] *= width
    points[:, 1] *= height
    points[:, 2] *= depth
    return points

def generate_random_sets(n_sets, max_n_items, width, height, depth=None):
    sets = []
    for _ in range(n_sets):
        n_items = np.random.randint(int(0.75 * max_n_items), max_n_items + 1)
        if depth is None:
            points = generate_2d_points_cloud(n_items, width, height)
        else:
            points = generate_3d_points_cloud(n_items, width, height, depth)
        sets.append(points)
    return sets

def build_graph(points_clouds, distance_threshold):
    graph = {}
    for i, point_cloud in enumerate(points_clouds):
        for j, other_cloud in enumerate(points_clouds):
            if i == j:
                continue
            for p1 in point_cloud:
                for p2 in other_cloud:
                    distance = np.linalg.norm(p1 - p2)
                    if distance < distance_threshold:
                        v1 = (float(p1[0]), float(p1[1]), i)
                        n  = (float(p2[0]), float(p2[1]), j)
                        graph.setdefault(v1, set()).add(n)
    return graph

def dfs(vertex, visited, labels, current_label, graph):
    stack = [vertex]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            labels[v] = current_label
            neighbors = graph.get(v, [])
            stack.extend(neighbors)

def connected_components_labeling(graph):
    """
    Produces another dictionary with the same keys as the input graph (vertices).
    Values are integers representing the connected component label for each vertex.
    """
    visited = set()
    labels = {}
    current_label = 0

    for vertex in graph.keys():
        if vertex not in visited:
            dfs(vertex, visited, labels, current_label, graph)
            current_label += 1

    return labels

def get_sub_graph_by_label(graph, labels, target_label):
    sub_graph = {}
    for vertex, label in labels.items():
        if label == target_label:
            sub_graph[vertex] = graph.get(vertex, set())
    return sub_graph

def find_n_tuples(vertex, subgraph, n, found_tuples, current_tuple, taken_partitions):
    """
    Recursively records all tuples of size N in the given subgraph.
    Each tuple must be a set of vertices belonging to N different partitions.
    The 'taken' variable indicates which partitions have already been included in the current tuple.
    The 'current_tuple' variable holds the current combination of vertices being explored.
    When a valid tuple of size N is found, it is added to the 'found_tuples' set
    """
    if taken_partitions[vertex[-1]]:
        return  # This vertex's partition is already represented in the current tuple
    current = current_tuple.copy()
    current.append(vertex)
    current = sorted(current)
    if len(current) == n:
        found_tuples.add(tuple(current))
        return
    taken = taken_partitions.copy()
    taken[vertex[-1]] = True
    for neighbor in subgraph.get(vertex, []):
        find_n_tuples(neighbor, subgraph, n, found_tuples, current, taken)

def total_length_of_tuple(t_set, graph):
    length = 0
    t = list(t_set)
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            v1, v2 = t[i], t[j]
            if v2 in graph.get(v1, []):
                length += 1
    return length

def remove_vertices_from_graph(graph, biggest_set):
    new_graph = {}
    flatten_set = set()
    for t in biggest_set:
        flatten_set.update(set(t))
    for vertex, neighbors in graph.items():
        if vertex not in flatten_set:
            new_neighbors = set(n for n in neighbors if n not in flatten_set)
            new_graph[vertex] = new_neighbors
    return new_graph

def find_non_overlapping_set_from(tuples, t_current, taken, current_set, best_set, graph):
    if taken.intersection(set(t_current)):
        return
    current = current_set.copy()
    current.add(t_current)
    if len(current) > len(best_set):
        best_set.clear()
        best_set.update(current)
    elif len(current) == len(best_set) and total_length_of_tuple(current, graph) < total_length_of_tuple(best_set, graph):
        best_set.clear()
        best_set.update(current)
    taken_current = taken.copy()
    taken_current.update(set(t_current))
    for t in tuples:
        find_non_overlapping_set_from(tuples, t, taken_current, current, best_set, graph)

def find_non_overlapping_set(tuples, graph):
    biggest_size = 0
    biggest_found = []
    for t in tuples:
        candidate = set()
        find_non_overlapping_set_from(tuples, t, set(), set(), candidate, graph)
        if len(candidate) > biggest_size:
            biggest_size = len(candidate)
            biggest_found = candidate
        elif len(candidate) == biggest_size and total_length_of_tuple(candidate, graph) < total_length_of_tuple(biggest_found, graph):
            biggest_found = candidate
            biggest_size = len(biggest_found)
    return biggest_found

def search_co_occurrences(graph, n_partitions):
    ccl = connected_components_labeling(graph)
    labels = set(ccl.values())
    levels = {k: set() for k in range(1, n_partitions + 1)}

    for label in labels:
        sub_graph = get_sub_graph_by_label(graph, ccl, label)
        for k in range(n_partitions, 0, -1):
            tuples = set()
            for vertex in sub_graph.keys():
                find_n_tuples(vertex, sub_graph, k, tuples, [], [False] * n_partitions)
            biggest_set = find_non_overlapping_set(tuples, sub_graph)
            sub_graph = remove_vertices_from_graph(sub_graph, biggest_set)
            levels[k].update(biggest_set)
        print(f"Label {label}:")
    return levels

def plot_graph(graph, ccl=None, colormap=None, levels=None):
    colors_pool = random_colors(40)
    plt.figure(figsize=(8, 8))
    for v1, neighbors in graph.items():
        x1, y1, _ = v1
        for neighbor in neighbors:
            x2, y2, _ = neighbor
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)
    
    xs = [v[0] for v in graph.keys()]
    ys = [v[1] for v in graph.keys()]
    
    edge_colors = [colors_pool[int(v[2]) % len(colors_pool)] for v in graph.keys()]
    sizes = [100 for _ in graph.keys()]
    
    if colormap:
        face_colors = [colormap.get(v, (0.5, 0.5, 0.5)) for v in graph.keys()]
    else:
        face_colors = edge_colors

    if levels:
        matching = {}
        for level, groups in levels.items():
            for vertices in groups:
                for vertex in vertices:
                    matching[vertex] = level
        sizes = [matching.get(v, 0) * 100 for v in graph.keys()]

    plt.scatter(xs, ys, c=face_colors, s=sizes, label='Vertices', edgecolors=edge_colors)

    if ccl:
        for vertex in graph.keys():
            x, y, _ = vertex
            label = ccl[vertex]
            plt.text(x, y, str(label), ha='center', va='center', fontsize=8, color='white', weight='bold')

    if levels:
        for level, tuples in levels.items():
            for t in tuples:
                points = np.array([[v[0], v[1]] for v in t])
                centroid = points.mean(axis=0)
                radius = np.max(np.linalg.norm(points - centroid, axis=1))
                circle = plt.Circle(centroid, radius, fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
                plt.gca().add_patch(circle)

    plt.title("Graph of co-occurrences")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()

import random

def random_color():
    r, g, b = (random.random() for _ in range(3))
    return (r, g, b)

def assign_colors(graph, levels):
    color_mapping = {}
    for _, tuples in levels.items():
        for t in tuples:
            clr = random_color()
            for vertex in t:
                color_mapping[vertex] = clr
    return color_mapping

def main():
    n_sets = 5
    max_n_items = 150
    width, height = 2048, 2048
    points_clouds = generate_random_sets(n_sets, max_n_items, width, height)

    for i, cloud in enumerate(points_clouds):
        print(f"Set {i+1}: {len(cloud)} points")
    
    graph = build_graph(points_clouds, distance_threshold=40)
    co_occ_levels = search_co_occurrences(graph, n_sets)
    print("============")
    for level, tuples in co_occ_levels.items():
        print(f"Level {level}: {len(tuples)} tuples")
    color_mapping = assign_colors(graph, co_occ_levels)
    plot_graph(graph, levels=co_occ_levels)

def test_dev():
    A = (0,  0, 0)
    B = (1,  0, 1)
    C = (2,  0, 0)
    D = (1, -1, 2)
    E = (2, -2, 0)
    F = (2, -3, 2)
    G = (1, -4, 1)

    t_to_a = {
        (0, 0, 0): 'A',
        (1, 0, 1): 'B',
        (2, 0, 0): 'C',
        (1, -1, 2): 'D',
        (2, -2, 0): 'E',
        (2, -3, 2): 'F',
        (1, -4, 1): 'G'
    }

    graph = {
        A: [B, D],
        B: [A, C, D, E],
        C: [B, D],
        D: [A, B, C, E],
        E: [B, D, F, G],
        F: [E, G],
        G: [E, F]
    }
    n_partitions = 3

    levels = search_co_occurrences(graph, n_partitions)
    for level, tuples in levels.items():
        print(f"Level {level}:")
        for t in tuples:
            print("  ", [t_to_a[v] for v in t])

if __name__ == "__main__":
    main()