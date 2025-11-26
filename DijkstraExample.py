import heapq
from Graph import Graph

# Dijkstra Algorithm Implementation for Graph usage reference
def run_dijkstra(graph_obj, start_node, end_node):
    """
    Returns: (total_distance, path_list)
    """

    # We use graph_obj.getNodes() so we don't assume IDs are 0..n-1
    nodes = graph_obj.getNodes()
    distances = {node: float('inf') for node in nodes}
    previous_nodes = {node: None for node in nodes} # For path reconstruction
    
    distances[start_node] = 0
    
    # Priority Queue: Stores tuples of (current_distance, current_node)
    pq = [(0, start_node)]
    
    while pq:
        # Pop the node with the smallest distance
        current_dist, u = heapq.heappop(pq)
        
        # Optimization: If we reached the target, we can stop early
        if u == end_node:
            break
        
        # If we found a shorter path to u strictly after pushing this to PQ, ignore
        if current_dist > distances[u]:
            continue
            
        # graph_obj.getNeighbors(u) returns {neighbor_id: weight}
        neighbors = graph_obj.getNeighbors(u)
        
        for v, weight in neighbors.items():
            new_dist = current_dist + weight
            
            # Relaxation Step
            if new_dist < distances[v]:
                distances[v] = new_dist
                previous_nodes[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    #Path Reconstruction (Backtracking)
    path = []
    current = end_node
    
    # If the end node is still infinite, there is no path
    if distances[end_node] == float('inf'):
        return float('inf'), []
        
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
        
    path.reverse() # Reverse it to get Start -> End
    
    return distances[end_node], path


if __name__ == "__main__":
    nodes = 30
    
    my_graph = Graph(nodes, 0.01) 
    start_id = 0
    end_id = nodes//2
    my_graph.removeEdge(0,end_id) # Remove direct edge just to make it more interesting
    
    print(f"--- Running Dijkstra from {start_id} to {end_id} ---")

    cost, path = run_dijkstra(my_graph, start_id, end_id)
    
    if cost == float('inf'):
        print("No path found!")
    else:
        print(f"Minimum Cost: {cost}")
        print(f"Path Taken: {path}")
        
    # Look for the path 0 -> ... -> 5 and add up the edge weights
    my_graph.showGraphPath(path)
    import networkx as nx
    comparison = nx.dijkstra_path(my_graph.getGraph(), start_id, end_id)
    my_graph.showGraphPath(comparison)
    