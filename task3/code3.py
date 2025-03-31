import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage.draw import line

img = cv2.imread("maze.png", cv2.IMREAD_GRAYSCALE)
#closing image at starting and enging point
def image_close(img,start,end):
    cv2.line(img, start, end, (0, 0, 0), 5)

image_close(img,(11, 335), (131,335))
image_close(img,(132, 22), (192,22))
image_close(img,(445, 278), (445,335))
cv2.imwrite("maze2.png", img)
ret,maze= cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#check if node already exist
def is_free(x, y, maze):
    return maze[y, x] == 255

#making new node at place where it is not exist in the frame 
def make_nodes(maze,sx,sy,ex,ey, num_nodes=200):
    nodes = []
    while len(nodes) < num_nodes:
        x, y = np.random.randint(sx,ex), np.random.randint(sy,ey)
        if is_free(x, y, maze):
            nodes.append((x, y))
    return np.array(nodes)

#check if the wall is not in between two node
def is_valid_edge(p1, p2, maze):
    x1, y1 = p1
    x2, y2 = p2
    rr, cc = line(y1, x1, y2, x2)# Bresenham’s line algorithm
    return np.all(maze[rr, cc] == 255)
#graph-based roadmap for path finding
def build_prm_graph(nodes, maze, k=10):
    tree = KDTree(nodes)
    graph = nx.Graph()
    for i, node in enumerate(nodes):
        graph.add_node(i, pos=node)
        _, indices = tree.query(node, k=k+1)
        for j in indices[1:]:
            if is_valid_edge(node, nodes[j], maze):
                dist = np.linalg.norm(np.array(node) - np.array(nodes[j]))
                graph.add_edge(i, j, weight=dist)
    return graph
#find nearent point
def nearest_node(point, nodes):
    distances = np.linalg.norm(nodes - np.array(point), axis=1)
    return np.argmin(distances)
# finding final path between starting point and end point
def find_path(graph, nodes, start, end):
    start_idx = nearest_node(start, nodes)
    end_idx = nearest_node(end, nodes)
    path = nx.shortest_path(graph, source=start_idx, target=end_idx, weight='weight')
    return [nodes[i] for i in path]
# ploting all the nodes edge and starting and ending point in a grid
def visualize(maze, nodes, graph, path,start,end,name):
    plt.figure(figsize=(10, 6))
    plt.title(name)
    plt.imshow(maze, cmap='gray')
    for edge in graph.edges:
        p1, p2 = nodes[edge[0]], nodes[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.3)
    plt.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=10)
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label="Path")
    plt.scatter([start[0]], [start[1]], c='yellow', s=100, label="Start")
    plt.scatter([end[0]], [end[1]], c='red', s=100, label="End")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


# for easy start easy end
nodes1 = make_nodes(maze,8,19,125,338, num_nodes=100)
graph1 = build_prm_graph(nodes1, maze, k=20)  
start1, end1 = (48, 324), (103, 324)
path1 = find_path(graph1, nodes1, start1, end1)
visualize(maze, nodes1, graph1, path1,start1,end1,"easy-start easy-end")

# for hard start hard end
nodes2 = make_nodes(maze,132,20,444,338, num_nodes=300)
graph2 = build_prm_graph(nodes2, maze, k=20) 
start2, end2 = (162, 28), (433, 306)
path2 = find_path(graph2, nodes2, start2, end2)
visualize(maze, nodes2, graph2, path2,start2,end2,"hard-start hard-end")