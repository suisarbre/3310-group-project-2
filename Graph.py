import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    G : nx.DiGraph = None
    def __init__(self,n:int,density:float,weightRange:tuple=(1,100)) -> None:
        """
        Initializes a random directed weighted graph.

        Args:
            n (int): The number of vertices (nodes).
            density (float): The probability of an edge creation (0.0 to 1.0).
            weightRange (tuple): The range (min, max) for random edge weights.
        """
        
        # Seed is set for reproducibility during debugging. 
        # TODO: Remove seed=11252025 before running final benchmarks to get random results.
        self.G = nx.gnp_random_graph(n, density, seed=11252025, directed=True)
        
        # Add Weights (random within specified range, default 1-100)
        for (u, v) in self.G.edges():
            self.G.edges[u, v]['weight'] = random.randint(weightRange[0], weightRange[1])
    
    def showGraph(self) -> None:
        """
        Visualizes the graph using Matplotlib. 
        This is a blocking call (code stops until window is closed).
        """
        if self.G is None:
            print("Graph is empty.")
            return
        
        pos = nx.spring_layout(self.G)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', node_size=500)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.show()
        
    def getGraph(self) -> nx.DiGraph:
        """
        Returns the underlying NetworkX graph object.
        
        Returns:
            nx.DiGraph: The NetworkX graph instance.
        """
        return self.G
    
    def setGraph(self, G: nx.DiGraph) -> None:
        """
        Manually sets the graph object (useful for testing specific scenarios).
        
        Args:
            G (nx.DiGraph): The new graph to set.
        """
        self.G = G
        
    def getAdjacencyMatrix(self) -> np.matrix:
        """
        Generates the Adjacency Matrix required for Floyd-Warshall.
        
        Non-edges are represented as np.inf (Infinity)
        Diagonal elements (distance to self) are 0.
        
        Returns:
            np.matrix: A 2D numpy array of Ints.
        """
        
        n = self.G.number_of_nodes()
        mat = np.full((n, n), np.inf)
        
        # Set diagonal to 0
        np.fill_diagonal(mat, 0.0)
        
        # Fill existing edges
        for u, v, data in self.G.edges(data=True):
            mat[u][v] = data['weight']
            
        return mat
    
    def getEdgeWeights(self) -> dict[tuple[int, int], int]:
        """
        Retrieves a dictionary of all edge weights.
        
        Returns:
            Dict[Tuple[int, int], int]: Dictionary format {(u, v): weight}
        """
        weights = {}
        for (u, v, wt) in self.G.edges.data('weight'):
            weights[(u, v)] = wt
        return weights
    
    #==========================================================
    def addEdge(self, u: int, v: int, weight: int) -> None:
        """
        Adds a directed edge from u to v with a specific weight.
        
        If u or v do not exist in the graph, they will be added.
        Args:
            u (int): The starting vertex.
            v (int): The ending vertex.
            weight (int): The weight of the edge.
        """
        # NetworkX automatically adds nodes u or v if they don't exist
        self.G.add_edge(u, v, weight=weight)
        
    def removeEdge(self, u: int, v: int) -> None:
        """
        Removes the edge from u to v if it exists.
        Args:
            u (int): The starting vertex.
            v (int): The ending vertex.
        """
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)
            
    def setEdgeWeight(self, u: int, v: int, new_weight: int) -> None:
        """
        Updates the weight of an existing edge.
        Args:
            u (int): The starting vertex.
            v (int): The ending vertex.
            new_weight (int): The new weight to set.
        """
        if self.G.has_edge(u, v):
            self.G[u][v]['weight'] = new_weight
        else:
            print(f"Warning: Edge {u}->{v} does not exist. Cannot set weight.") 
    
    def addNode(self, u: int) -> None:
        """
        Adds a single node u.
        Args:
            u (int): The vertex to add.
        """
        self.G.add_node(u)
        
    def removeNode(self, u: int) -> None:
        """
        Removes node u and all connected edges.
        Args:
            u (int): The vertex to remove.
        """
        if self.G.has_node(u):
            self.G.remove_node(u)
            
            
if __name__ == "__main__":
    print("Generating Graph...")
    my_graph = Graph(6, 0.8)
    weights = my_graph.getEdgeWeights()
    print("Edge Weights:", weights)
    

    print("Visualizing...")
    my_graph.showGraph()
    
    print("Adjacency Matrix:")
    mat = my_graph.getAdjacencyMatrix()
    print(mat)