from DijkstraExample import run_dijkstra
from Graph import Graph

def repeated_dijkstra(graph_obj):
    """
    Compute all-pairs shortest paths using repeated Dijkstra.
    
    Args:
        graph_obj: Graph object with getNodes() and getNeighbors() methods
    
    Returns:
        tuple: (all_distances, all_paths) where:
            - all_distances[start][end] = shortest distance
            - all_paths[start][end] = shortest path as list of nodes
    """
    nodes = graph_obj.getNodes()

    all_distances = {}
    all_paths = {}
    
    # For each starting node
    for start in nodes:
        all_distances[start] = {}
        all_paths[start] = {}
        
        # For each ending node
        for end in nodes:
            # If start and end node is the same
            if start == end:
                all_distances[start][end] = 0
                all_paths[start][end] = [start]
            # Otherwise run dijkstra from start to end
            else:
                distance, path = run_dijkstra(graph_obj, start, end)
                all_distances[start][end] = distance
                all_paths[start][end] = path
    
    return all_distances, all_paths

if __name__ == "__main__":
    nodes = 30
    
    my_graph = Graph(nodes, 0.01) 
    start_id = 0
    end_id = nodes//2
    # my_graph.removeEdge(0,end_id) # Remove direct edge just to make it more interesting
    
    print(f"--- Running Repeated Dijkstra ---")
    
    all_distances, all_paths = repeated_dijkstra(my_graph)
    
    # Gets value from start to end
    cost = all_distances[start_id][end_id]
    path = all_paths[start_id][end_id]
    
    if cost == float('inf'):
        print("No path found!")
    else:
        print(f"Minimum Cost: {cost}")
        print(f"Path Taken: {path}")
        
    # Look for the path 0 -> ... -> end_id and add up the edge weights
    my_graph.showGraphPath(path)
    import networkx as nx
    comparison = nx.dijkstra_path(my_graph.getGraph(), start_id, end_id)
    my_graph.showGraphPath(comparison)
