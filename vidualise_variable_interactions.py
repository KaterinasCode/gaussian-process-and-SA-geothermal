import networkx as nx
import matplotlib.pyplot as plt

A = nx.Graph()
A.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
i = 1

while i < 18:
    j = 1
    while j < i:
        if i != j:
            A.add_edge(i, j, weight=s2[i-1][j-1])
        j += 1
    i += 1

edges_A, weights_A = zip(*nx.get_edge_attributes(A, 'weight').items())

plt.figure(figsize=(8, 8))  # Adjust the figsize to make the plot more square

plt.subplot(111, aspect='equal')  # Set the aspect ratio to 'equal'

pos = nx.circular_layout(A)
node_colors = ['white'] * len(A.nodes())
edge_colors = list(weights_A)

nx.draw(A, pos=pos, cmap=plt.get_cmap('binary'), node_color=node_colors, node_size=500, edgecolors='black',
        edge_color=edge_colors, width=2, with_labels=True, edge_cmap=plt.cm.binary)

plt.title("Depth")

# Add a colorbar for the weights
sm = plt.cm.ScalarMappable(cmap=plt.cm.binary, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
sm.set_array([])
plt.colorbar(sm, label="Weight")

plt.savefig('network_obj1.png')
plt.show()
