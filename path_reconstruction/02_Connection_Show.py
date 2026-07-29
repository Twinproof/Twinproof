import matplotlib.pyplot as plt
import ast
import math
import pandas as pd
from collections import deque
import matplotlib
from pathlib import Path
matplotlib.use("TkAgg")


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


SCRIPT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(SCRIPT_DIR / "Paths.csv")


def reverse_path_situation(path_situation):
    reversed_path = []
    for direction, length in reversed(path_situation):
        new_dir = direction if direction == 0 else -direction
        reversed_path.append((new_dir, length))
    return reversed_path

graph = {}
for _, row in df.iterrows():
    start, end = row['Start_Anchor'], row['End_Anchor']
    path = ast.literal_eval(row['Path_Situation'])
    graph.setdefault(start, []).append((end, path))
    graph.setdefault(end, []).append((start, reverse_path_situation(path)))


def get_path_coordinates(start_point, path_situation, initial_direction=0):
    coords = [start_point]
    x, y = start_point
    direction = initial_direction  

    
    for turn, length in path_situation:
        if turn == 1:  
            direction -= 90
        elif turn == -1:  
            direction += 90
        direction %= 360  

        rad = math.radians(direction)
        x += length * math.cos(rad)
        y += length * math.sin(rad)
        coords.append((x, y))

    return coords, direction  


def draw_all_paths(graph, start_id=0):
    plt.figure(figsize=(10, 10))
    plt.title("Anchor path network")
    plt.axis('equal')

    positions = {}
    orientations = {}
    visited_edges = set()
    visited_nodes = set()

    positions[start_id] = (0, 0)
    orientations[start_id] = 0
    queue = deque()
    queue.append((start_id, (0, 0)))

    while queue:
        current_id, current_pos = queue.popleft()
        current_dir = orientations[current_id]
        visited_nodes.add(current_id)

        for neighbor_id, path_situation in graph.get(current_id, []):
            edge_key = tuple(sorted((current_id, neighbor_id)))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            coords, final_dir = get_path_coordinates(current_pos, path_situation, current_dir)
            end_pos = coords[-1]

            if neighbor_id not in positions:
                positions[neighbor_id] = end_pos
                orientations[neighbor_id] = final_dir
                queue.append((neighbor_id, end_pos))

            
            xs, ys = zip(*coords)
            plt.plot(xs, ys, 'b')
            plt.plot(xs[0], ys[0], 'ro')  # Start
            plt.plot(xs[-1], ys[-1], 'bo')  
            plt.text(xs[0], ys[0], str(current_id), fontsize=9, ha='right', va='bottom')
            plt.text(xs[-1], ys[-1], str(neighbor_id), fontsize=9, ha='left', va='top')

    
    for anchor in graph.keys():
        if anchor not in positions:
            print(f"Unconnected isolated nodes:{anchor}")
            positions[anchor] = (9999, 9999)
            plt.plot(9999, 9999, 'ko')
            plt.text(9999, 9999, str(anchor), fontsize=9)

    # plt.grid(True)
    plt.show()


draw_all_paths(graph, start_id=0)
