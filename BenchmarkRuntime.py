# BenchmarkRuntime.py
#
# This script benchmarks the runtime of:
#   (1) Repeated Dijkstra's Algorithm (APSP by running Dijkstra from every node)
#   (2) Floyd–Warshall Algorithm (classic APSP)
#
# For multiple graph sizes and densities, we:
#   - generate a new random connected directed graph
#   - run both algorithms
#   - measure runtime using time.perf_counter()
#   - repeat multiple times and compute the average runtime
#
# This file serves as the official runtime experiment module for the report.


import time
from Graph import Graph
from RepeatedDijkstra import all_pairs_dijkstra     # <-- Change name if needed
from FloydWarshall import floyd_warshall            # <-- Change name if needed


def benchmark_one_setting(n, density, num_runs=5):
    """
    Benchmark both APSP algorithms for a fixed graph size and density.

    Args:
        n (int): Number of vertices in the generated graph.
        density (float): Edge density of the directed graph (0.0–1.0).
        num_runs (int): How many independent trials to run.

    Returns:
        (float, float): A tuple containing:
            - average runtime of Repeated Dijkstra
            - average runtime of Floyd–Warshall
    """

    total_dij = 0.0
    total_fw = 0.0

    for _ in range(num_runs):

        # -------------------------
        # 1. Generate a random graph
        # -------------------------
        # Graph() ensures:
        #   - directed weighted graph
        #   - connectivity guaranteed by initial cycle
        g = Graph(n=n, density=density)

        # Precompute adjacency matrix for Floyd–Warshall
        adj = g.getAdjacencyMatrix()

        # -------------------------
        # 2. Time Repeated Dijkstra
        # -------------------------
        start = time.perf_counter()
        _ = all_pairs_dijkstra(g)   # Return value not needed for timing
        end = time.perf_counter()
        total_dij += (end - start)

        # -------------------------
        # 3. Time Floyd–Warshall
        # -------------------------
        start = time.perf_counter()
        _ = floyd_warshall(adj)
        end = time.perf_counter()
        total_fw += (end - start)

    # Compute average runtime across trials
    avg_dij = total_dij / num_runs
    avg_fw = total_fw / num_runs

    return avg_dij, avg_fw


def main():
    """
    Runs the full benchmark across:
        - multiple graph sizes
        - sparse and dense configurations
        - repeated trials per (n, density)

    Prints results in CSV format, which makes it easy to copy into Excel or Google Sheets.
    """

    # Graph sizes to test (modify if needed)
    sizes = [10, 20, 30, 40, 50]

    # Number of repeated trials per configuration
    num_runs = 5

    # Two density regimes: sparse vs. dense
    densities = [
        ("sparse", 0.10),
        ("dense", 0.50)
    ]

    print("n, density_label, density_value, avg_dijkstra (s), avg_floydwarshall (s)")

    for n in sizes:
        for label, d in densities:

            avg_dij, avg_fw = benchmark_one_setting(n, d, num_runs=num_runs)

            # CSV-style output for easy export
            print(f"{n}, {label}, {d}, {avg_dij:.6f}, {avg_fw:.6f}")


if __name__ == "__main__":
    main()
