import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': [r"\usepackage{amsmath}"]
})



def read_file(fname):
    nodes = {}
    edges = {}

    with open(fname) as f:
        read_nodes = False
        read_conns = False

        for l in f:
            if "Nodes" in l:
                read_nodes = True
            elif "Connections" in l:
                read_conns = True
                read_nodes = False
            elif read_nodes:
                id_, params = l.split(" ", maxsplit=1)
                id_ = id_.replace("\t", "")
                params = params.replace("SNPSNodeGene(", "").replace(")", "")
                _, c, p, d, f, _, _ = params.split(", ")
                c = int(float(c.split("=")[1]))
                p = int(float(p.split("=")[1]))
                d = int(float(d.split("=")[1]))
                f = int(float(f.split("=")[1]))
                nodes[id_] = [c, p, d, f]
            elif read_conns:
                s = l.replace("DefaultConnectionGene(", "").replace(")", "").replace("key=(", "")
                from_, to, w, _ = s.split(", ")
                #from_, to = key.replace("key=(", "").replace(")", "").split(", ")
                w = max(0, int(float(w.split("=")[1])))
                if w > 0:
                    if from_ not in edges:
                        edges[from_] = []
                    edges[from_].append((to, w))

    return nodes, edges


nodes, edges = read_file(sys.argv[1])
labels = {}
edge_labels = {}

G = nx.DiGraph()

for n, vals in nodes.items():
    G.add_node(n)
    labels[n] = f"${n}\\\\ E/a^{{{vals[0]}}}\\rightarrow a^{{{vals[1]}}}; {vals[2]} \\\\ {vals[3]}$"

for orig, dests in edges.items():
    for d, w in dests:
        if orig not in nodes:
            nodes[orig] = []
            labels[orig] = sys.argv[1 - int(orig)]
            G.add_node(orig)
        edge_labels[(orig, d)] = w
        G.add_edge(orig, d)

size = 5500
plotsize = size * 100
pos = {}

for i, n in enumerate(nodes):
    pos[n] = [1 if int(n) > 0 else -1, i * plotsize // len(nodes) * size]

pos = nx.spring_layout(G)
nx.draw(G, pos, labels=labels, node_size=size, node_color="#fff", edgecolors="#000")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.show()
# nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='r')
