import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(gph_file, title, output_file):
    # Read graph
    G = nx.DiGraph()
    with open(gph_file, 'r') as f:
        for line in f:
            parent, child = line.strip().split(',')
            G.add_edge(parent, child)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=15, 
                          arrowstyle='->', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")

# Generate all three
visualize_graph('example.gph', 'Small Dataset: Titanic Network', 'small_graph.png')
visualize_graph('medium.gph', 'Medium Dataset: Wine Quality Network', 'medium_graph.png')
visualize_graph('large.gph', 'Large Dataset: Network Structure', 'large_graph.png')