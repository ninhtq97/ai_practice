import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
from itertools import combinations
from math import comb
from networkx.algorithms.community.quality import modularity

rnd.seed()

G = nx.read_edgelist("karate")

nx.draw(G, with_labels=True)
plt.show()

def common_neighbors(G,i,j):
    # G - the graph
    # i,j - the link
    return len(set(G.neighbors(i)) & set(G.neighbors(j)))


def link_list(G,i,score_func):
    # G - the graph
    # i - the node
    # score_func - the link scoring function, having the signature score_func(G,i,j)
    links = []
    for j in G.nodes():
        if i != j and not G.has_edge(i,j):
            e = (i,j)
            sc = score_func(G,i,j)
            links.append([e,sc])
    links.sort(key=lambda x: x[1], reverse=True)
    return links


def link_prediction(G,k,i,score_func):
    # G - the graph
    # k - the number of links to predict
    # i - the node
    # score_func - the link scoring function, having the signature score_func(G,i,j)
    links = link_list(G,i,score_func)
    return links[:k]


def evaluate_link_prediction(G, k, score_func):
    edges = list(G.edges())
    correct = 0
    total = 0

    for u, v in edges:
        # copy graph và remove edge
        G_temp = G.copy()
        if G_temp.has_edge(u, v):
            G_temp.remove_edge(u, v)

        # dự đoán từ u
        preds_u = link_prediction(G_temp, k, u, score_func)
        predicted_nodes_from_u = {link[0][1] for link in preds_u}

        # dự đoán từ v
        preds_v = link_prediction(G_temp, k, v, score_func)
        predicted_nodes_from_v = {link[0][1] for link in preds_v}

        # kiểm tra cạnh bị xóa có xuất hiện không
        if v in predicted_nodes_from_u or u in predicted_nodes_from_v:
            correct += 1
        total += 1

    return correct / total if total else 0


print(link_prediction(G,5,'10',common_neighbors))

acc = evaluate_link_prediction(G, 5, common_neighbors)
print(f"Leave-One-Out Accuracy: {acc:.2f}")
print("\n-----------------------------\n")

diameter = nx.diameter(G)
avg_distance = nx.average_shortest_path_length(G)

print("Diameter:", diameter)
print("Average shortest path length:", avg_distance)
print("\n-----------------------------\n")

pagerank = nx.pagerank(G)
pr_values = list(pagerank.values())
degrees = [G.degree(n) for n in G.nodes()]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(pr_values, bins=10, color='skyblue', edgecolor='black')
plt.title("PageRank Distribution")
plt.xlabel("PageRank")
plt.ylabel("Count")

plt.subplot(1,2,2)
plt.hist(degrees, bins=10, color='lightgreen', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# -----------------------------
# RANDOM NETWORK
# -----------------------------

def random_network_from_graph(G):
    """
    Sinh random graph cùng N và p với G (Erdos-Renyi)
    """
    N = G.number_of_nodes()
    M = G.number_of_edges()
    p = 2 * M / (N * (N - 1))
    G_rand = nx.gnp_random_graph(N, p)
    return G_rand, N, p

def compare_graph_metrics(G, G_rand):
    """
    So sánh average distance, assortativity, clustering
    """
    avg_dist_G = nx.average_shortest_path_length(G)
    assort_G = nx.degree_assortativity_coefficient(G)
    clust_G = nx.average_clustering(G)

    # Random graph có thể disconnected → lấy largest connected component
    if not nx.is_connected(G_rand):
        largest_cc = max(nx.connected_components(G_rand), key=len)
        G_rand_sub = G_rand.subgraph(largest_cc).copy()
    else:
        G_rand_sub = G_rand

    avg_dist_rand = nx.average_shortest_path_length(G_rand_sub)
    assort_rand = nx.degree_assortativity_coefficient(G_rand)
    clust_rand = nx.average_clustering(G_rand)

    return {
        'avg_distance': (avg_dist_G, avg_dist_rand),
        'assortativity': (assort_G, assort_rand),
        'clustering': (clust_G, clust_rand)
    }

def plot_graphs_side_by_side(G1, title1, G2, title2):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    nx.draw(G1, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(title1)

    plt.subplot(1,2,2)
    nx.draw(G2, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title(title2)
    plt.show()

# -----------------------------
# TRIANGLES
# -----------------------------

def triangles_per_node(G):
    """
    Trả về dict node -> số tam giác node tham gia
    """
    triangles = {node: 0 for node in G.nodes()}
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        for v, w in combinations(neighbors, 2):
            if G.has_edge(v, w):
                triangles[u] += 1
    return triangles

def total_triangles(G):
    """
    Tổng số tam giác trong graph
    """
    tdict = triangles_per_node(G)
    return sum(tdict.values()) // 3

def expected_triangles_random_graph(N, p):
    """
    Số tam giác kỳ vọng trong G(n,p)
    """
    return comb(N,3) * (p**3)


# -----------------------------
# COMMUNITY DETECTION
# -----------------------------

def girvan_newman_max_modularity(G):
    """
    Girvan-Newman (divisive) – trả về communities có modularity cao nhất
    """
    G_temp = G.copy()

    best_mod = -1
    best_partition = [set(G_temp.nodes())]

    while G_temp.number_of_edges() > 0:
        # 1. Tính edge betweenness centrality
        edge_bw = nx.edge_betweenness_centrality(G_temp)
        # 2. Cạnh có centrality cao nhất
        max_bw = max(edge_bw.values())
        for e in [edge for edge, bw in edge_bw.items() if bw == max_bw]:
            G_temp.remove_edge(*e)

        # 3. Xác định communities
        communities = list(nx.connected_components(G_temp))
        if len(communities) == 1:
            continue

        # 4. Tính modularity
        mod = modularity(G, communities)

        # 5. Cập nhật best
        if mod > best_mod:
            best_mod = mod
            best_partition = communities

        # Stop nếu mỗi node là 1 cộng đồng
        if len(communities) == G.number_of_nodes():
            break

    return best_partition, best_mod

def draw_communities(G, communities):
    """
    Vẽ graph với màu theo cộng đồng
    """
    node_color_map = {}
    for idx, com in enumerate(communities):
        for node in com:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    nx.draw(G, with_labels=True, node_color=colors, cmap=plt.get_cmap('tab20'))
    plt.show()


# --- Random graph ---
G_rand, N, p = random_network_from_graph(G)
metrics = compare_graph_metrics(G, G_rand)
print(f"Average distance: Karate={metrics['avg_distance'][0]:.3f}, Random={metrics['avg_distance'][1]:.3f}")
print(f"Assortativity:    Karate={metrics['assortativity'][0]:.3f}, Random={metrics['assortativity'][1]:.3f}")
print(f"Clustering coeff: Karate={metrics['clustering'][0]:.3f}, Random={metrics['clustering'][1]:.3f}")
plot_graphs_side_by_side(G, "Karate", G_rand, "Random Graph")

print("\n-----------------------------\n")

# --- Triangles ---
triangles_dict = triangles_per_node(G)
print("Triangles per node:", triangles_dict)
print("Total triangles:", total_triangles(G))
print(f"Expected triangles random: {expected_triangles_random_graph(N,p):.2f}")

print("\n-----------------------------\n")

# --- Community Detection ---
best_partition, best_mod = girvan_newman_max_modularity(G)
print(f"Best modularity: {best_mod:.3f}")
print("Communities:", best_partition)
draw_communities(G, best_partition)
