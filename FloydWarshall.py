import numpy as np
from Graph import Graph

def floyd_warshall(adj_matrix):
    """
    Computes All-Pairs Shortest Paths using the Floyd-Warshall algorithm.
    
    This is the main function called by BenchmarkRuntime.py.
    
    Args:
        adj_matrix (numpy.ndarray): n x n adjacency matrix where:
            - adj_matrix[i][j] = weight of edge from i to j
            - adj_matrix[i][j] = inf if no edge exists
            - adj_matrix[i][i] = 0 (distance from node to itself)
    
    Returns:
        numpy.ndarray: n x n matrix where entry [i][j] contains the shortest
                       distance from node i to node j
    """
    # Make a copy so we don't modify the original matrix
    dist = np.copy(adj_matrix)
    n = len(dist)
    
    # Floyd-Warshall Triple Loop
    # Try each intermediate vertex k
    for k in range(n):
        # For each source vertex i
        for i in range(n):
            # For each destination vertex j
            for j in range(n):
                # If path i -> k -> j is shorter than current i -> j, update it
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist


def floyd_warshall_with_paths(adj_matrix):
    """
    Computes All-Pairs Shortest Paths with path reconstruction using Floyd-Warshall.
    
    This is for the EXTRA CREDIT portion of the project.
    
    Args:
        adj_matrix (numpy.ndarray): n x n adjacency matrix
    
    Returns:
        tuple: (distance_matrix, next_matrix)
            - distance_matrix: n x n matrix of shortest distances
            - next_matrix: n x n matrix where [i][j] stores the next node
              to visit on the shortest path from i to j
    """
    # Make a copy of the adjacency matrix
    dist = np.copy(adj_matrix)
    n = len(dist)
    
    # Initialize the next matrix
    # next[i][j] = j if there's a direct edge from i to j, None otherwise
    next_node = np.full((n, n), None)
    
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] != float('inf'):
                next_node[i][j] = j
    
    # Floyd-Warshall with path reconstruction
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    return dist, next_node


def reconstruct_path(next_matrix, start, end):
    """
    Reconstructs the shortest path from start to end using the next matrix.
    
    Args:
        next_matrix: Matrix from floyd_warshall_with_paths
        start (int): Starting node
        end (int): Ending node
    
    Returns:
        list: The shortest path from start to end, or empty list if no path exists
    """
    if next_matrix[start][end] is None:
        return []  # No path exists
    
    path = [start]
    current = start
    
    while current != end:
        current = next_matrix[current][end]
        if current is None:
            return []  # Path broken
        path.append(current)
    
    return path


if __name__ == "__main__":
    # Test the implementation
    print("Testing Floyd-Warshall Algorithm")
    print("=" * 50)
    
    # Create a small test graph
    test_graph = Graph(n=5, density=0.3, Randomseed=42)
    
    print("\nTest Graph:")
    print(f"Nodes: {test_graph.getNodes()}")
    print(f"Edges: {test_graph.getEdgeWeights()}")
    
    # Get adjacency matrix
    adj_matrix = test_graph.getAdjacencyMatrix()
    print("\nAdjacency Matrix:")
    print(adj_matrix)
    
    # Run Floyd-Warshall
    print("\n--- Computing All-Pairs Shortest Paths ---")
    apsp_matrix = floyd_warshall(adj_matrix)
    
    print("\nAll-Pairs Shortest Path Distance Matrix:")
    print(apsp_matrix)
    
    # Test with path reconstruction (Extra Credit)
    print("\n--- Testing Path Reconstruction (Extra Credit) ---")
    dist_matrix, next_matrix = floyd_warshall_with_paths(adj_matrix)
    
    # Example: Find path from node 0 to node 3
    start_node = 0
    end_node = 3
    path = reconstruct_path(next_matrix, start_node, end_node)
    
    print(f"\nShortest path from {start_node} to {end_node}:")
    print(f"Path: {path}")
    print(f"Distance: {dist_matrix[start_node][end_node]}")
    
    # Visualize the path
    if path:
        test_graph.showGraphPath(path)
    
    # Verify correctness by comparing a few paths
    print("\n--- Sanity Check: Comparing with NetworkX ---")
    import networkx as nx
    
    for i in [0, 1]:
        for j in [2, 3]:
            try:
                nx_dist = nx.shortest_path_length(test_graph.getGraph(), i, j, weight='weight')
                fw_dist = apsp_matrix[i][j]
                match = "✓" if abs(nx_dist - fw_dist) < 0.001 else "✗"
                print(f"Path {i} → {j}: NetworkX={nx_dist:.1f}, Floyd-Warshall={fw_dist:.1f} {match}")
            except nx.NetworkXNoPath:
                print(f"Path {i} → {j}: No path exists")
