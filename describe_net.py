import pickle
import numpy as np


w = pickle.load(open("winner.pkl", "rb"))

nodes = w.nodes
connections = w.connections


for i, v in nodes.items():
    print(f"Node {i}")
    print(f"\tc: {v.c1}")
    print(f"\tp: {v.p1}")
    print(f"\td: {v.d1}")
    print("\tCONNECTIONS")
    for j, cv in connections.items():
        if i == j[1] or i == j[0]:
            print(f"\t\t{j} with weight {cv.weight}")
