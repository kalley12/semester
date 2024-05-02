Slp1.1
Write programs to sort a list of n numbers in ascending order using selection sort
public class SelectionSort {
    public static void main(String[] args) {
        int[] numbers = {5, 2, 9, 1, 5, 6};

        System.out.println("Original array:");
        printArray(numbers);

        selectionSort(numbers);

        System.out.println("\nSorted array:");
        printArray(numbers);
    }

    // Selection sort algorithm
    public static void selectionSort(int[] arr) {
        int n = arr.length;

        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            // Swap the found minimum element with the first element
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }

    // Utility method to print an array
    public static void printArray(int[] arr) {
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}
Write a program to sort a given set of elements using the Quick sort method and determine the time required to sort the elements. Repeat the experiment for different values of n, the number of elements in the list to be sorted. The elements can be read from a file or can be generated using the random number generator.
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class QuickSortTiming {
    public static void main(String[] args) {
        // Different values of n
        int[] sizes = {1000, 5000, 10000, 20000};

        for (int size : sizes) {
            int[] elements = generateRandomArray(size); // Generate random elements
            long startTime = System.nanoTime();
            quickSort(elements, 0, elements.length - 1); // Sort the array using Quick Sort
            long endTime = System.nanoTime();
            long duration = (endTime - startTime) / 1000000; // Convert nanoseconds to milliseconds
            System.out.println("Time taken to sort " + size + " elements: " + duration + " milliseconds");
        }
    }

    // Quick Sort algorithm
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);

            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    // Partition function for Quick Sort
    public static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                // Swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        // Swap arr[i+1] and arr[high] (or pivot)
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

    // Utility method to generate a random array of given size
    public static int[] generateRandomArray(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (int) (Math.random() * 1000); // Generate random numbers between 0 and 999
        }
        return arr;
    }
}

Slep.2.1
Write a program to sort n randomly generated elements using Heapsort method.[
import java.util.Arrays;

public class HeapSort {
    public static void main(String[] args) {
        int n = 10; // Number of elements
        int[] arr = generateRandomArray(n); // Generate random elements

        System.out.println("Original array:");
        System.out.println(Arrays.toString(arr));

        heapSort(arr); // Sort the array using Heap Sort

        System.out.println("Sorted array:");
        System.out.println(Arrays.toString(arr));
    }

    // Heap Sort algorithm
    public static void heapSort(int[] arr) {
        int n = arr.length;

        // Build heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);

        // One by one extract an element from heap
        for (int i = n - 1; i > 0; i--) {
            // Move current root to end
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;

            // call max heapify on the reduced heap
            heapify(arr, i, 0);
        }
    }

    // To heapify a subtree rooted with node i which is an index in arr[]
    public static void heapify(int[] arr, int n, int i) {
        int largest = i; // Initialize largest as root
        int left = 2 * i + 1; // left = 2*i + 1
        int right = 2 * i + 2; // right = 2*i + 2

        // If left child is larger than root
        if (left < n && arr[left] > arr[largest])
            largest = left;

        // If right child is larger than largest so far
        if (right < n && arr[right] > arr[largest])
            largest = right;

        // If largest is not root
        if (largest != i) {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;

            // Recursively heapify the affected sub-tree
            heapify(arr, n, largest);
        }
    }

    // Utility method to generate a random array of given size
    public static int[] generateRandomArray(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = (int) (Math.random() * 1000); // Generate random numbers between 0 and 999
        }
        return arr;
    }
}

Write a program to implement Strassen’s Matrix multiplication
public class StrassenMatrixMultiplication {
    public static void main(String[] args) {
        int[][] A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        int[][] B = {{17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}};

        int[][] C = strassenMultiply(A, B);

        System.out.println("Resultant Matrix:");
        printMatrix(C);
    }

    public static int[][] strassenMultiply(int[][] A, int[][] B) {
        int n = A.length;
        int[][] C = new int[n][n];

        // Base case: If the matrices are 1x1
        if (n == 1) {
            C[0][0] = A[0][0] * B[0][0];
        } else {
            // Split matrices into quadrants
            int[][] A11 = subMatrix(A, 0, 0);
            int[][] A12 = subMatrix(A, 0, n / 2);
            int[][] A21 = subMatrix(A, n / 2, 0);
            int[][] A22 = subMatrix(A, n / 2, n / 2);

            int[][] B11 = subMatrix(B, 0, 0);
            int[][] B12 = subMatrix(B, 0, n / 2);
            int[][] B21 = subMatrix(B, n / 2, 0);
            int[][] B22 = subMatrix(B, n / 2, n / 2);

            // Calculate intermediate matrices
            int[][] M1 = strassenMultiply(addMatrices(A11, A22), addMatrices(B11, B22));
            int[][] M2 = strassenMultiply(addMatrices(A21, A22), B11);
            int[][] M3 = strassenMultiply(A11, subtractMatrices(B12, B22));
            int[][] M4 = strassenMultiply(A22, subtractMatrices(B21, B11));
            int[][] M5 = strassenMultiply(addMatrices(A11, A12), B22);
            int[][] M6 = strassenMultiply(subtractMatrices(A21, A11), addMatrices(B11, B12));
            int[][] M7 = strassenMultiply(subtractMatrices(A12, A22), addMatrices(B21, B22));

            // Calculate result sub-matrices
            int[][] C11 = addMatrices(subtractMatrices(addMatrices(M1, M4), M5), M7);
            int[][] C12 = addMatrices(M3, M5);
            int[][] C21 = addMatrices(M2, M4);
            int[][] C22 = addMatrices(subtractMatrices(addMatrices(M1, M3), M2), M6);

            // Merge result sub-matrices into result matrix
            mergeMatrices(C, C11, 0, 0);
            mergeMatrices(C, C12, 0, n / 2);
            mergeMatrices(C, C21, n / 2, 0);
            mergeMatrices(C, C22, n / 2, n / 2);
        }

        return C;
    }

    // Utility methods for matrix operations
    public static int[][] subMatrix(int[][] matrix, int row, int col) {
        int n = matrix.length / 2;
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix[row + i][col + j];
            }
        }
        return result;
    }

    public static void mergeMatrices(int[][] result, int[][] subMatrix, int row, int col) {
        int n = subMatrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[row + i][col + j] = subMatrix[i][j];
            }
        }
    }

    public static int[][] addMatrices(int[][] A, int[][] B) {
        int n = A.length;
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    }

    public static int[][] subtractMatrices(int[][] A, int[][] B) {
        int n = A.length;
        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        return result;
    }

    public static void printMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            for (int num : row) {
                System.out.print(num + " ");
            }
            System.out.println();
        }
    }
}
Write a program to sort a given set of elements using the Quick sort method and determine the time required to sort the elements
import java.util.Arrays;

public class QuickSortTime {
    public static void main(String[] args) {
        int[] arr = {5, 2, 9, 1, 5, 6}; // Input array

        System.out.println("Original array:");
        System.out.println(Arrays.toString(arr));

        long startTime = System.nanoTime(); // Start time
        quickSort(arr, 0, arr.length - 1); // Sort the array using Quick Sort
        long endTime = System.nanoTime(); // End time

        System.out.println("Sorted array:");
        System.out.println(Arrays.toString(arr));

        long duration = (endTime - startTime) / 1000000; // Calculate duration in milliseconds
        System.out.println("Time taken to sort: " + duration + " milliseconds");
    }

    // Quick Sort algorithm
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);

            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    // Partition function for Quick Sort
    public static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                // Swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        // Swap arr[i+1] and arr[high] (or pivot)
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }
}
Write a program to find Minimum Cost Spanning Tree of a given undirected graph using Prims algorithm
import java.util.Arrays;

public class PrimMinimumSpanningTree {
    public static void main(String[] args) {
        int[][] graph = {
            {0, 2, 0, 6, 0},
            {2, 0, 3, 8, 5},
            {0, 3, 0, 0, 7},
            {6, 8, 0, 0, 9},
            {0, 5, 7, 9, 0}
        };

        int[] mst = primMST(graph);

        System.out.println("Minimum Cost Spanning Tree:");
        printMST(mst, graph);
    }

    public static int[] primMST(int[][] graph) {
        int n = graph.length;
        int[] parent = new int[n];
        int[] key = new int[n];
        boolean[] mstSet = new boolean[n];

        Arrays.fill(key, Integer.MAX_VALUE);
        Arrays.fill(mstSet, false);

        key[0] = 0;
        parent[0] = -1;

        for (int count = 0; count < n - 1; count++) {
            int u = minKey(key, mstSet);
            mstSet[u] = true;

            for (int v = 0; v < n; v++) {
                if (graph[u][v] != 0 && !mstSet[v] && graph[u][v] < key[v]) {
                    parent[v] = u;
                    key[v] = graph[u][v];
                }
            }
        }

        return parent;
    }

    public static int minKey(int[] key, boolean[] mstSet) {
        int min = Integer.MAX_VALUE;
        int minIndex = -1;

        for (int v = 0; v < key.length; v++) {
            if (!mstSet[v] && key[v] < min) {
                min = key[v];
                minIndex = v;
            }
        }

        return minIndex;
    }

    public static void printMST(int[] parent, int[][] graph) {
        for (int i = 1; i < parent.length; i++) {
            System.out.println("Edge: " + parent[i] + " - " + i + ", Weight: " + graph[i][parent[i]]);
        }
    }
}
Write a program to implement a Merge Sort algorithm to sort a given set of elements and determine the time required to sort the elements
import java.util.Arrays;

public class MergeSortTime {
    public static void main(String[] args) {
        int[] arr = {5, 2, 9, 1, 5, 6}; // Input array

        System.out.println("Original array:");
        System.out.println(Arrays.toString(arr));

        long startTime = System.nanoTime(); // Start time
        mergeSort(arr, 0, arr.length - 1); // Sort the array using Merge Sort
        long endTime = System.nanoTime(); // End time

        System.out.println("Sorted array:");
        System.out.println(Arrays.toString(arr));

        long duration = (endTime - startTime) / 1000000; // Calculate duration in milliseconds
        System.out.println("Time taken to sort: " + duration + " milliseconds");
    }

    // Merge Sort algorithm
    public static void mergeSort(int[] arr, int low, int high) {
        if (low < high) {
            int mid = (low + high) / 2;
            mergeSort(arr, low, mid); // Sort left half
            mergeSort(arr, mid + 1, high); // Sort right half
            merge(arr, low, mid, high); // Merge the sorted halves
        }
    }

    // Merge two sorted subarrays into one sorted array
    public static void merge(int[] arr, int low, int mid, int high) {
        int n1 = mid - low + 1;
        int n2 = high - mid;

        int[] leftArr = new int[n1];
        int[] rightArr = new int[n2];

        // Copy data to temporary arrays
        for (int i = 0; i < n1; ++i)
            leftArr[i] = arr[low + i];
        for (int j = 0; j < n2; ++j)
            rightArr[j] = arr[mid + 1 + j];

        // Merge the temporary arrays
        int i = 0, j = 0;
        int k = low;
        while (i < n1 && j < n2) {
            if (leftArr[i] <= rightArr[j]) {
                arr[k] = leftArr[i];
                i++;
            } else {
                arr[k] = rightArr[j];
                j++;
            }
            k++;
        }

        // Copy remaining elements of leftArr[], if any
        while (i < n1) {
            arr[k] = leftArr[i];
            i++;
            k++;
        }

        // Copy remaining elements of rightArr[], if any
        while (j < n2) {
            arr[k] = rightArr[j];
            j++;
            k++;
        }
    }
}
Write a program to implement Knapsack problems using Greedy method
import java.util.Arrays;
import java.util.Comparator;

class Item {
    int weight, value;

    Item(int weight, int value) {
        this.weight = weight;
        this.value = value;
    }
}

public class FractionalKnapsackGreedy {
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;

        double maxValue = getMaxValue(values, weights, capacity);
        System.out.println("Maximum value obtained: " + maxValue);
    }

    public static double getMaxValue(int[] values, int[] weights, int capacity) {
        int n = values.length;
        Item[] items = new Item[n];

        for (int i = 0; i < n; i++) {
            items[i] = new Item(weights[i], values[i]);
        }

        Arrays.sort(items, Comparator.comparingDouble((Item i) -> (double) i.value / i.weight).reversed());

        double totalValue = 0;
        for (Item item : items) {
            int currWeight = item.weight;
            int currValue = item.value;

            if (capacity - currWeight >= 0) {
                capacity -= currWeight;
                totalValue += currValue;
            } else {
                double fraction = (double) capacity / currWeight;
                totalValue += currValue * fraction;
                break;
            }
        }

        return totalValue;
    }
}
Write a program for the Implementation of Kruskals algorithm to find minimum cost spanning tree .
import java.util.*;

class Edge implements Comparable<Edge> {
    int src, dest, weight;

    public Edge(int src, int dest, int weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }

    @Override
    public int compareTo(Edge otherEdge) {
        return this.weight - otherEdge.weight;
    }
}

public class KruskalMinimumSpanningTree {
    public static void main(String[] args) {
        int V = 4; // Number of vertices
        int E = 5; // Number of edges

        List<Edge> edges = new ArrayList<>();
        edges.add(new Edge(0, 1, 10));
        edges.add(new Edge(0, 2, 6));
        edges.add(new Edge(0, 3, 5));
        edges.add(new Edge(1, 3, 15));
        edges.add(new Edge(2, 3, 4));

        List<Edge> mst = kruskalMST(edges, V, E);

        System.out.println("Minimum Cost Spanning Tree:");
        for (Edge edge : mst) {
            System.out.println(edge.src + " - " + edge.dest + ", Weight: " + edge.weight);
        }
    }

    public static List<Edge> kruskalMST(List<Edge> edges, int V, int E) {
        List<Edge> mst = new ArrayList<>();

        // Sort edges by weight
        Collections.sort(edges);

        int[] parent = new int[V];
        Arrays.fill(parent, -1);

        int edgeCount = 0;
        int index = 0;
        while (edgeCount < V - 1) {
            Edge nextEdge = edges.get(index++);
            int x = find(parent, nextEdge.src);
            int y = find(parent, nextEdge.dest);

            if (x != y) {
                mst.add(nextEdge);
                union(parent, x, y);
                edgeCount++;
            }
        }

        return mst;
    }

    public static int find(int[] parent, int i) {
        if (parent[i] == -1)
            return i;
        return find(parent, parent[i]);
    }

    public static void union(int[] parent, int x, int y) {
        int xSet = find(parent, x);
        int ySet = find(parent, y);
        parent[xSet] = ySet;
    }
}
Write a program to implement huffman Code using greedy methods and also calculate the best case and worst case complexity.
import java.util.*;

class HuffmanNode implements Comparable<HuffmanNode> {
    char data;
    int frequency;
    HuffmanNode left, right;

    HuffmanNode(char data, int frequency) {
        this.data = data;
        this.frequency = frequency;
    }

    @Override
    public int compareTo(HuffmanNode other) {
        return this.frequency - other.frequency;
    }
}

public class HuffmanCoding {
    public static void main(String[] args) {
        String input = "huffman coding example";

        System.out.println("Input String: " + input);

        Map<Character, Integer> frequencyMap = buildFrequencyMap(input);

        HuffmanNode root = buildHuffmanTree(frequencyMap);

        Map<Character, String> huffmanCodes = generateHuffmanCodes(root);

        System.out.println("Huffman Codes:");
        for (Map.Entry<Character, String> entry : huffmanCodes.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }

        int bestCaseComplexity = calculateBestCaseComplexity(frequencyMap);
        int worstCaseComplexity = calculateWorstCaseComplexity(frequencyMap);

        System.out.println("Best-case Complexity: " + bestCaseComplexity);
        System.out.println("Worst-case Complexity: " + worstCaseComplexity);
    }

    public static Map<Character, Integer> buildFrequencyMap(String input) {
        Map<Character, Integer> frequencyMap = new HashMap<>();

        for (char c : input.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }

        return frequencyMap;
    }

    public static HuffmanNode buildHuffmanTree(Map<Character, Integer> frequencyMap) {
        PriorityQueue<HuffmanNode> pq = new PriorityQueue<>();

        for (Map.Entry<Character, Integer> entry : frequencyMap.entrySet()) {
            pq.offer(new HuffmanNode(entry.getKey(), entry.getValue()));
        }

        while (pq.size() > 1) {
            HuffmanNode left = pq.poll();
            HuffmanNode right = pq.poll();

            HuffmanNode parent = new HuffmanNode('\0', left.frequency + right.frequency);
            parent.left = left;
            parent.right = right;

            pq.offer(parent);
        }

        return pq.poll();
    }

    public static Map<Character, String> generateHuffmanCodes(HuffmanNode root) {
        Map<Character, String> huffmanCodes = new HashMap<>();
        generateCodesRecursive(root, "", huffmanCodes);
        return huffmanCodes;
    }

    private static void generateCodesRecursive(HuffmanNode node, String code, Map<Character, String> huffmanCodes) {
        if (node == null) return;

        if (node.left == null && node.right == null) {
            huffmanCodes.put(node.data, code);
        }

        generateCodesRecursive(node.left, code + "0", huffmanCodes);
        generateCodesRecursive(node.right, code + "1", huffmanCodes);
    }

    public static int calculateBestCaseComplexity(Map<Character, Integer> frequencyMap) {
        // Best-case complexity: O(n log n) where n is the number of distinct characters
        return frequencyMap.size() * (int) Math.ceil(Math.log(frequencyMap.size()));
    }

    public static int calculateWorstCaseComplexity(Map<Character, Integer> frequencyMap) {
        // Worst-case complexity: O(n log n) where n is the total number of characters
        int totalFrequency = frequencyMap.values().stream().mapToInt(Integer::intValue).sum();
        return totalFrequency * (int) Math.ceil(Math.log(totalFrequency));
    }
}
Write a program for the Implementation of Prim’s algorithm to find minimum cost spanning tree
import java.util.*;

class Edge implements Comparable<Edge> {
    int src, dest, weight;

    public Edge(int src, int dest, int weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }

    @Override
    public int compareTo(Edge otherEdge) {
        return this.weight - otherEdge.weight;
    }
}

public class PrimMinimumSpanningTree {
    public static void main(String[] args) {
        int V = 5; // Number of vertices
        int E = 7; // Number of edges

        List<List<Edge>> adj = new ArrayList<>();
        for (int i = 0; i < V; i++) {
            adj.add(new ArrayList<>());
        }

        // Add edges to the graph
        addEdge(adj, 0, 1, 2);
        addEdge(adj, 0, 3, 6);
        addEdge(adj, 1, 2, 3);
        addEdge(adj, 1, 3, 8);
        addEdge(adj, 1, 4, 5);
        addEdge(adj, 2, 4, 7);
        addEdge(adj, 3, 4, 9);

        List<Edge> mst = primMST(adj, V);

        System.out.println("Minimum Cost Spanning Tree:");
        for (Edge edge : mst) {
            System.out.println(edge.src + " - " + edge.dest + ", Weight: " + edge.weight);
        }
    }

    public static void addEdge(List<List<Edge>> adj, int src, int dest, int weight) {
        adj.get(src).add(new Edge(src, dest, weight));
        adj.get(dest).add(new Edge(dest, src, weight)); // for undirected graph
    }

    public static List<Edge> primMST(List<List<Edge>> adj, int V) {
        List<Edge> mst = new ArrayList<>();
        PriorityQueue<Edge> pq = new PriorityQueue<>();

        boolean[] visited = new boolean[V];
        Arrays.fill(visited, false);

        pq.offer(new Edge(-1, 0, 0)); // Start from vertex 0

        while (!pq.isEmpty()) {
            Edge minEdge = pq.poll();

            if (visited[minEdge.dest]) continue;

            visited[minEdge.dest] = true;
            if (minEdge.src != -1) {
                mst.add(minEdge);
            }

            for (Edge neighbor : adj.get(minEdge.dest)) {
                if (!visited[neighbor.dest]) {
                    pq.offer(neighbor);
                }
            }
        }

        return mst;
    }
}

Write a Program to find only length of Longest Common Subsequence.
public class LongestCommonSubsequenceLength {
    public static void main(String[] args) {
        String str1 = "AGGTAB";
        String str2 = "GXTXAYB";

        int length = findLCSLength(str1, str2);
        System.out.println("Length of Longest Common Subsequence: " + length);
    }

    public static int findLCSLength(String str1, String str2) {
        int m = str1.length();
        int n = str2.length();

        int[][] dp = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0 || j == 0)
                    dp[i][j] = 0;
                else if (str1.charAt(i - 1) == str2.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                else
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        return dp[m][n];
    }
}
)Write a program for the Implementation of Dijkstra’s algorithm to find shortest path to other vertices
import java.util.*;

public class DijkstraShortestPath {
    public static void main(String[] args) {
        int V = 5; // Number of vertices
        int source = 0; // Source vertex

        // Create a weighted graph
        int[][] graph = {
                {0, 4, 0, 0, 0},
                {4, 0, 8, 0, 0},
                {0, 8, 0, 7, 0},
                {0, 0, 7, 0, 9},
                {0, 0, 0, 9, 0}
        };

        int[] distances = dijkstra(graph, V, source);

        System.out.println("Shortest distances from source vertex " + source + ":");
        for (int i = 0; i < V; i++) {
            System.out.println("Vertex " + i + ": " + distances[i]);
        }
    }

    public static int[] dijkstra(int[][] graph, int V, int source) {
        int[] distances = new int[V];
        boolean[] visited = new boolean[V];

        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[source] = 0;

        for (int count = 0; count < V - 1; count++) {
            int minIndex = minDistance(distances, visited);
            visited[minIndex] = true;

            for (int v = 0; v < V; v++) {
                if (!visited[v] && graph[minIndex][v] != 0 && distances[minIndex] != Integer.MAX_VALUE
                        && distances[minIndex] + graph[minIndex][v] < distances[v]) {
                    distances[v] = distances[minIndex] + graph[minIndex][v];
                }
            }
        }

        return distances;
    }

    public static int minDistance(int[] distances, boolean[] visited) {
        int min = Integer.MAX_VALUE;
        int minIndex = -1;

        for (int v = 0; v < distances.length; v++) {
            if (!visited[v] && distances[v] <= min) {
                min = distances[v];
                minIndex = v;
            }
        }

        return minIndex;
    }
}

Write a program for finding Topological sorting for Directed Acyclic Graph (DAG)

import java.util.*;

public class TopologicalSorting {
    public static void main(String[] args) {
        int V = 6; // Number of vertices

        // Create a directed acyclic graph (DAG)
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < V; i++) {
            graph.add(new ArrayList<>());
        }
        addEdge(graph, 5, 2);
        addEdge(graph, 5, 0);
        addEdge(graph, 4, 0);
        addEdge(graph, 4, 1);
        addEdge(graph, 2, 3);
        addEdge(graph, 3, 1);

        System.out.println("Topological Sorting:");
        List<Integer> result = topologicalSort(graph, V);
        for (int vertex : result) {
            System.out.print(vertex + " ");
        }
    }

    public static void addEdge(List<List<Integer>> graph, int src, int dest) {
        graph.get(src).add(dest);
    }

    public static List<Integer> topologicalSort(List<List<Integer>> graph, int V) {
        Stack<Integer> stack = new Stack<>();
        boolean[] visited = new boolean[V];

        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                topologicalSortUtil(graph, i, visited, stack);
            }
        }

        List<Integer> result = new ArrayList<>();
        while (!stack.isEmpty()) {
            result.add(stack.pop());
        }

        return result;
    }

    public static void topologicalSortUtil(List<List<Integer>> graph, int v, boolean[] visited, Stack<Integer> stack) {
        visited[v] = true;

        for (int neighbor : graph.get(v)) {
            if (!visited[neighbor]) {
                topologicalSortUtil(graph, neighbor, visited, stack);
            }
        }

        stack.push(v);
    }
}
Write a program to implement Fractional Knapsack problems using Greedy Method
import java.util.*;

class Item {
    int weight, value;

    Item(int weight, int value) {
        this.weight = weight;
        this.value = value;
    }
}

public class FractionalKnapsackGreedy {
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;

        double maxValue = getMaxValue(values, weights, capacity);
        System.out.println("Maximum value obtained: " + maxValue);
    }

    public static double getMaxValue(int[] values, int[] weights, int capacity) {
        int n = values.length;
        Item[] items = new Item[n];

        for (int i = 0; i < n; i++) {
            items[i] = new Item(weights[i], values[i]);
        }

        Arrays.sort(items, Comparator.comparingDouble((Item i) -> (double) i.value / i.weight).reversed());

        double totalValue = 0;
        for (Item item : items) {
            int currWeight = item.weight;
            int currValue = item.value;

            if (capacity - currWeight >= 0) {
                capacity -= currWeight;
                totalValue += currValue;
            } else {
                double fraction = (double) capacity / currWeight;
                totalValue += currValue * fraction;
                break;
            }
        }

        return totalValue;
    }
}
Write Program to implement Traveling Salesman Problem using nearest neighbor algorithm
import java.util.*;

public class TravelingSalesmanNearestNeighbor {
    public static void main(String[] args) {
        int[][] graph = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };

        int[] tour = nearestNeighborTSP(graph);
        int totalCost = calculateTourCost(graph, tour);

        System.out.println("Tour: " + Arrays.toString(tour));
        System.out.println("Total Cost: " + totalCost);
    }

    public static int[] nearestNeighborTSP(int[][] graph) {
        int n = graph.length;
        int[] tour = new int[n];
        boolean[] visited = new boolean[n];

        // Start with the first city as the initial city
        tour[0] = 0;
        visited[0] = true;

        for (int i = 1; i < n; i++) {
            int currentCity = tour[i - 1];
            int minDistance = Integer.MAX_VALUE;
            int nearestCity = -1;

            for (int j = 0; j < n; j++) {
                if (!visited[j] && graph[currentCity][j] < minDistance) {
                    minDistance = graph[currentCity][j];
                    nearestCity = j;
                }
            }

            tour[i] = nearestCity;
            visited[nearestCity] = true;
        }

        // Return to the starting city to complete the tour
        tour[n - 1] = 0;

        return tour;
    }

    public static int calculateTourCost(int[][] graph, int[] tour) {
        int cost = 0;
        int n = graph.length;

        for (int i = 0; i < n - 1; i++) {
            int from = tour[i];
            int to = tour[i + 1];
            cost += graph[from][to];
        }

        // Add the cost of returning to the starting city
        cost += graph[tour[n - 1]][tour[0]];

        return cost;
    }
}
Write a program to implement optimal binary search tree and also calculate the best case complexity.
public class OptimalBinarySearchTree {
    public static void main(String[] args) {
        // Keys and their frequencies
        int[] keys = {10, 12, 20};
        int[] freq = {34, 8, 50};

        int n = keys.length;
        double bestCaseComplexity = optimalBST(keys, freq, n);
        System.out.println("Best-case complexity of optimal binary search tree: " + bestCaseComplexity);
    }

    public static double optimalBST(int[] keys, int[] freq, int n) {
        // Create a 2D array to store costs of subtree ranges
        double[][] cost = new double[n + 1][n + 1];

        // Base case: single keys have the same cost as their frequency
        for (int i = 0; i < n; i++) {
            cost[i][i] = freq[i];
        }

        // Fill the cost array diagonally
        for (int L = 2; L <= n; L++) {
            for (int i = 0; i <= n - L + 1; i++) {
                int j = i + L - 1;
                cost[i][j] = Double.MAX_VALUE;

                // Try making all keys in subarray keys[i..j] the root
                for (int r = i; r <= j; r++) {
                    double c = ((r > i) ? cost[i][r - 1] : 0) +
                            ((r < j) ? cost[r + 1][j] : 0) +
                            sum(freq, i, j);
                    if (c < cost[i][j]) {
                        cost[i][j] = c;
                    }
                }
            }
        }

        return cost[0][n - 1];
    }

    public static int sum(int[] freq, int i, int j) {
        int sum = 0;
        for (int k = i; k <= j; k++) {
            sum += freq[k];
        }
        return sum;
    }
}

Write a program to implement Sum of Subset by Backtracking
import java.util.*;

public class SubsetSumBacktracking {
    public static void main(String[] args) {
        int[] set = {10, 7, 5, 18, 12, 20, 15};
        int sum = 35;

        System.out.println("Original Set: " + Arrays.toString(set));
        System.out.println("Target Sum: " + sum);
        System.out.println("Subsets with sum equal to " + sum + ":");
        findSubsetsWithSum(set, sum);
    }

    public static void findSubsetsWithSum(int[] set, int sum) {
        List<Integer> subset = new ArrayList<>();
        findSubsets(set, sum, subset, 0, 0);
    }

    public static void findSubsets(int[] set, int sum, List<Integer> subset, int index, int currentSum) {
        if (currentSum == sum) {
            System.out.println(subset);
            return;
        }

        for (int i = index; i < set.length; i++) {
            if (currentSum + set[i] <= sum) {
                subset.add(set[i]);
                findSubsets(set, sum, subset, i + 1, currentSum + set[i]);
                subset.remove(subset.size() - 1);
            }
        }
    }
}
Write a program to implement huffman Code using greedy methods

import java.util.*;

class HuffmanNode implements Comparable<HuffmanNode> {
    char data;
    int frequency;
    HuffmanNode left, right;

    HuffmanNode(char data, int frequency) {
        this.data = data;
        this.frequency = frequency;
    }

    @Override
    public int compareTo(HuffmanNode other) {
        return this.frequency - other.frequency;
    }
}

public class HuffmanCodingGreedy {
    public static void main(String[] args) {
        String input = "huffman coding example";

        System.out.println("Input String: " + input);

        Map<Character, Integer> frequencyMap = buildFrequencyMap(input);

        HuffmanNode root = buildHuffmanTree(frequencyMap);

        Map<Character, String> huffmanCodes = generateHuffmanCodes(root);

        System.out.println("Huffman Codes:");
        for (Map.Entry<Character, String> entry : huffmanCodes.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    public static Map<Character, Integer> buildFrequencyMap(String input) {
        Map<Character, Integer> frequencyMap = new HashMap<>();

        for (char c : input.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }

        return frequencyMap;
    }

    public static HuffmanNode buildHuffmanTree(Map<Character, Integer> frequencyMap) {
        PriorityQueue<HuffmanNode> pq = new PriorityQueue<>();

        for (Map.Entry<Character, Integer> entry : frequencyMap.entrySet()) {
            pq.offer(new HuffmanNode(entry.getKey(), entry.getValue()));
        }

        while (pq.size() > 1) {
            HuffmanNode left = pq.poll();
            HuffmanNode right = pq.poll();

            HuffmanNode parent = new HuffmanNode('\0', left.frequency + right.frequency);
            parent.left = left;
            parent.right = right;

            pq.offer(parent);
        }

        return pq.poll();
    }

    public static Map<Character, String> generateHuffmanCodes(HuffmanNode root) {
        Map<Character, String> huffmanCodes = new HashMap<>();
        generateCodesRecursive(root, "", huffmanCodes);
        return huffmanCodes;
    }

    private static void generateCodesRecursive(HuffmanNode node, String code, Map<Character, String> huffmanCodes) {
        if (node == null) return;

        if (node.left == null && node.right == null) {
            huffmanCodes.put(node.data, code);
        }

        generateCodesRecursive(node.left, code + "0", huffmanCodes);
        generateCodesRecursive(node.right, code + "1", huffmanCodes);
    }
}

Write a program to solve 4 Queens Problem using Backtracking
public class NQueensBacktracking {
    final static int N = 4;

    public static void main(String[] args) {
        solveNQueens();
    }

    public static void solveNQueens() {
        int[][] board = new int[N][N];
        if (solveNQueensUtil(board, 0)) {
            printSolution(board);
        } else {
            System.out.println("Solution does not exist");
        }
    }

    public static boolean solveNQueensUtil(int[][] board, int col) {
        if (col >= N) {
            return true;
        }

        for (int i = 0; i < N; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;

                if (solveNQueensUtil(board, col + 1)) {
                    return true;
                }

                board[i][col] = 0; // Backtrack
            }
        }

        return false;
    }

    public static boolean isSafe(int[][] board, int row, int col) {
        int i, j;

        // Check this row on left side
        for (i = 0; i < col; i++) {
            if (board[row][i] == 1) {
                return false;
            }
        }

        // Check upper diagonal on left side
        for (i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 1) {
                return false;
            }
        }

        // Check lower diagonal on left side
        for (i = row, j = col; j >= 0 && i < N; i++, j--) {
            if (board[i][j] == 1) {
                return false;
            }
        }

        return true;
    }

    public static void printSolution(int[][] board) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(" " + board[i][j] + " ");
            }
            System.out.println();
        }
    }
}
Write programs to implement DFS (Depth First Search) and determine the time complexity for the same.
import java.util.*;

public class DepthFirstSearch {
    static class Graph {
        int V;
        LinkedList<Integer>[] adjList;

        Graph(int V) {
            this.V = V;
            adjList = new LinkedList[V];
            for (int i = 0; i < V; i++) {
                adjList[i] = new LinkedList<>();
            }
        }

        void addEdge(int v, int w) {
            adjList[v].add(w);
        }

        void DFSUtil(int v, boolean[] visited) {
            visited[v] = true;
            System.out.print(v + " ");

            for (int neighbor : adjList[v]) {
                if (!visited[neighbor]) {
                    DFSUtil(neighbor, visited);
                }
            }
        }

        void DFS(int v) {
            boolean[] visited = new boolean[V];
            DFSUtil(v, visited);
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(4);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(2, 0);
        graph.addEdge(2, 3);
        graph.addEdge(3, 3);

        System.out.println("Depth First Traversal starting from vertex 2:");
        graph.DFS(2);
    }
}
Write a program in C/C++/ Java to find shortest paths from a given vertex in a weighted connected graph, to other vertices using Dijikstra’s algorithm.
import java.util.*;

public class DijkstraShortestPath {
    static class Graph {
        int V;
        List<List<Edge>> adj;

        Graph(int V) {
            this.V = V;
            adj = new ArrayList<>(V);
            for (int i = 0; i < V; i++) {
                adj.add(new ArrayList<>());
            }
        }

        void addEdge(int src, int dest, int weight) {
            adj.get(src).add(new Edge(dest, weight));
            adj.get(dest).add(new Edge(src, weight)); // For undirected graph
        }

        void dijkstra(int src) {
            PriorityQueue<Node> pq = new PriorityQueue<>(V, new Node());
            int[] dist = new int[V];
            Arrays.fill(dist, Integer.MAX_VALUE);
            dist[src] = 0;
            pq.add(new Node(src, 0));

            while (!pq.isEmpty()) {
                int u = pq.poll().node;

                for (Edge e : adj.get(u)) {
                    int v = e.dest;
                    int weight = e.weight;

                    if (dist[v] > dist[u] + weight) {
                        dist[v] = dist[u] + weight;
                        pq.add(new Node(v, dist[v]));
                    }
                }
            }

            printShortestPaths(src, dist);
        }

        void printShortestPaths(int src, int[] dist) {
            System.out.println("Shortest paths from vertex " + src + " to:");
            for (int i = 0; i < V; i++) {
                System.out.println("Vertex " + i + ": " + dist[i]);
            }
        }
    }

    static class Edge {
        int dest, weight;

        Edge(int dest, int weight) {
            this.dest = dest;
            this.weight = weight;
        }
    }

    static class Node implements Comparator<Node> {
        int node, cost;

        Node() {
        }

        Node(int node, int cost) {
            this.node = node;
            this.cost = cost;
        }

        @Override
        public int compare(Node n1, Node n2) {
            if (n1.cost < n2.cost) return -1;
            if (n1.cost > n2.cost) return 1;
            return 0;
        }
    }

    public static void main(String[] args) {
        int V = 5; // Number of vertices
        int src = 0; // Source vertex

        Graph graph = new Graph(V);
        graph.addEdge(0, 1, 2);
        graph.addEdge(0, 3, 6);
        graph.addEdge(1, 2, 3);
        graph.addEdge(1, 3, 8);
        graph.addEdge(1, 4, 5);
        graph.addEdge(2, 4, 7);
        graph.addEdge(3, 4, 9);

        graph.dijkstra(src);
    }
}
Write programs to implement BFS (Breadth complexity for the same. First Search) and determine the time
import java.util.*;

public class BreadthFirstSearch {
    static class Graph {
        int V;
        List<List<Integer>> adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList<>(V);
            for (int i = 0; i < V; i++) {
                adjList.add(new ArrayList<>());
            }
        }

        void addEdge(int src, int dest) {
            adjList.get(src).add(dest);
            adjList.get(dest).add(src); // For undirected graph
        }

        void BFS(int start) {
            boolean[] visited = new boolean[V];
            Queue<Integer> queue = new LinkedList<>();

            visited[start] = true;
            queue.offer(start);

            while (!queue.isEmpty()) {
                int current = queue.poll();
                System.out.print(current + " ");

                for (int neighbor : adjList.get(current)) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.offer(neighbor);
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        int V = 6; // Number of vertices
        Graph graph = new Graph(V);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 4);
        graph.addEdge(3, 4);
        graph.addEdge(3, 5);
        graph.addEdge(4, 5);

        System.out.println("BFS Traversal starting from vertex 0:");
        graph.BFS(0);
    }
}
Write a program in C/C++/ Java to sort a given set of elements using the Selection sort method and determine the time required to sort the elements.
import java.util.*;

public class SelectionSort {
    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        System.out.println("Original Array: " + Arrays.toString(arr));

        long startTime = System.nanoTime();
        selectionSort(arr);
        long endTime = System.nanoTime();

        long duration = (endTime - startTime);
        System.out.println("Sorted Array: " + Arrays.toString(arr));
        System.out.println("Time taken: " + duration + " nanoseconds");
    }

    public static void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            // Swap the found minimum element with the first element
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
}
Write a program to find minimum number of multiplications in Matrix Chain Multiplication.
public class MatrixChainMultiplication {
    public static void main(String[] args) {
        int[] dimensions = {10, 20, 30, 40, 30};
        int minMultiplications = matrixChainOrder(dimensions);
        System.out.println("Minimum number of multiplications: " + minMultiplications);
    }

    public static int matrixChainOrder(int[] dimensions) {
        int n = dimensions.length - 1; // Number of matrices
        int[][] dp = new int[n][n];

        // dp[i][j] stores the minimum number of scalar multiplications needed to compute the matrix product A[i] * A[i+1] * ... * A[j]
        // Chain length l ranges from 2 to n (inclusive)
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i < n - l + 1; i++) {
                int j = i + l - 1;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                    dp[i][j] = Math.min(dp[i][j], cost);
                }
            }
        }

        return dp[0][n - 1];
    }
}
Write a program in C/C++/ Java to implement an optimal binary search tree and also calculate the best case and worst case complexity.
public class OptimalBinarySearchTree {
    public static void main(String[] args) {
        int[] keys = {10, 20, 30, 40, 50};
        int[] freq = {4, 2, 6, 3, 1};
        int n = keys.length;

        System.out.println("Cost of optimal BST: " + optimalBST(keys, freq, n));
    }

    public static int optimalBST(int[] keys, int[] freq, int n) {
        int[][] cost = new int[n + 1][n + 1];

        for (int i = 0; i < n; i++) {
            cost[i][i] = freq[i];
        }

        for (int L = 2; L <= n; L++) {
            for (int i = 0; i <= n - L + 1; i++) {
                int j = i + L - 1;
                cost[i][j] = Integer.MAX_VALUE;
                int sum = getSum(freq, i, j);

                for (int r = i; r <= j; r++) {
                    int c = ((r > i) ? cost[i][r - 1] : 0) +
                            ((r < j) ? cost[r + 1][j] : 0) +
                            sum;

                    if (c < cost[i][j]) {
                        cost[i][j] = c;
                    }
                }
            }
        }

        return cost[0][n - 1];
    }

    public static int getSum(int[] freq, int i, int j) {
        int sum = 0;
        for (int k = i; k <= j; k++) {
            sum += freq[k];
        }
        return sum;
    }
}
Write programs to implement the Job Sequencing with deadlines using greedy methods.
import java.util.*;

public class JobSequencing {
    static class Job {
        char id;
        int deadline;
        int profit;

        Job(char id, int deadline, int profit) {
            this.id = id;
            this.deadline = deadline;
            this.profit = profit;
        }
    }

    public static void main(String[] args) {
        Job[] jobs = {
                new Job('a', 2, 100),
                new Job('b', 1, 19),
                new Job('c', 2, 27),
                new Job('d', 1, 25),
                new Job('e', 3, 15)
        };

        int maxDeadline = Arrays.stream(jobs).mapToInt(job -> job.deadline).max().orElse(0);
        char[] sequence = new char[maxDeadline];
        int totalProfit = scheduleJobs(jobs, sequence);

        System.out.println("Sequence of jobs: " + Arrays.toString(sequence));
        System.out.println("Total profit: " + totalProfit);
    }

    public static int scheduleJobs(Job[] jobs, char[] sequence) {
        Arrays.sort(jobs, (a, b) -> b.profit - a.profit); // Sort jobs by profit in descending order

        boolean[] slot = new boolean[jobs.length];
        int totalProfit = 0;

        for (Job job : jobs) {
            for (int i = Math.min(job.deadline, jobs.length) - 1; i >= 0; i--) {
                if (!slot[i]) {
                    sequence[i] = job.id;
                    slot[i] = true;
                    totalProfit += job.profit;
                    break;
                }
            }
        }

        return totalProfit;
    }
}
Write a program in C/C++/ Java to implement DFS and BFS. Compare the time complexity
import java.util.*;

public class GraphTraversal {
    static class Graph {
        int V;
        LinkedList<Integer>[] adjList;

        Graph(int V) {
            this.V = V;
            adjList = new LinkedList[V];
            for (int i = 0; i < V; i++) {
                adjList[i] = new LinkedList<>();
            }
        }

        void addEdge(int src, int dest) {
            adjList[src].add(dest);
            adjList[dest].add(src); // For undirected graph
        }

        void DFS(int start) {
            boolean[] visited = new boolean[V];
            System.out.println("DFS Traversal starting from vertex " + start + ":");
            DFSUtil(start, visited);
        }

        void DFSUtil(int v, boolean[] visited) {
            visited[v] = true;
            System.out.print(v + " ");

            for (int neighbor : adjList[v]) {
                if (!visited[neighbor]) {
                    DFSUtil(neighbor, visited);
                }
            }
        }

        void BFS(int start) {
            boolean[] visited = new boolean[V];
            Queue<Integer> queue = new LinkedList<>();
            System.out.println("\nBFS Traversal starting from vertex " + start + ":");

            visited[start] = true;
            queue.offer(start);

            while (!queue.isEmpty()) {
                int current = queue.poll();
                System.out.print(current + " ");

                for (int neighbor : adjList[current]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.offer(neighbor);
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 4);

        graph.DFS(0);
        graph.BFS(0);
    }
}

Write programs to implement to find out solution for 0/1 knapsack problem using LCBB(Least Cost Branch and Bound).
public class KnapsackLCBB {
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;
        int n = values.length;

        int maxValue = knapsackLCBB(values, weights, capacity, n);
        System.out.println("Maximum value that can be obtained: " + maxValue);
    }

    public static int knapsackLCBB(int[] values, int[] weights, int capacity, int n) {
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        return dp[n][capacity];
    }
}
Write a program in C/C++/ Java a to implement Graph Coloring Algorithm
import java.util.*;

public class GraphColoring {
    static class Graph {
        int V;
        List<List<Integer>> adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList<>(V);
            for (int i = 0; i < V; i++) {
                adjList.add(new ArrayList<>());
            }
        }

        void addEdge(int src, int dest) {
            adjList.get(src).add(dest);
            adjList.get(dest).add(src); // For undirected graph
        }

        void colorGraph() {
            int[] result = new int[V];
            Arrays.fill(result, -1); // Initialize all vertices with no color

            boolean[] availableColors = new boolean[V];
            Arrays.fill(availableColors, true); // All colors are initially available

            // Assign colors to remaining V-1 vertices
            for (int u = 1; u < V; u++) {
                for (int neighbor : adjList.get(u)) {
                    if (result[neighbor] != -1) {
                        availableColors[result[neighbor]] = false; // Mark the color of neighbor as unavailable
                    }
                }

                // Find the first available color
                int color;
                for (color = 0; color < V; color++) {
                    if (availableColors[color]) {
                        break;
                    }
                }

                result[u] = color; // Assign the found color to vertex u

                // Reset the availability of colors for the next vertex
                Arrays.fill(availableColors, true);
            }

            // Print the result
            System.out.println("Vertex \t Color");
            for (int u = 0; u < V; u++) {
                System.out.println(u + " \t " + result[u]);
            }
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);

        System.out.println("Coloring of graph vertices:");
        graph.colorGraph();
    }
}
Write programs to implement to find out solution for 0/1 knapsack problem using dynamic programming.
public class KnapsackDynamicProgramming {
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;
        int n = values.length;

        int maxValue = knapsackDynamicProgramming(values, weights, capacity, n);
        System.out.println("Maximum value that can be obtained: " + maxValue);
    }

    public static int knapsackDynamicProgramming(int[] values, int[] weights, int capacity, int n) {
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        return dp[n][capacity];
    }
}
Write a program in C/C++/ Java to determine if a given graph is a Hamiltonian cycle or not
import java.util.*;

public class HamiltonianCycle {
    static class Graph {
        int V;
        List<Integer>[] adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList[V];
            for (int i = 0; i < V; i++) {
                adjList[i] = new ArrayList<>();
            }
        }

        void addEdge(int src, int dest) {
            adjList[src].add(dest);
            adjList[dest].add(src); // For undirected graph
        }

        boolean isHamiltonianCycle() {
            int[] path = new int[V];
            Arrays.fill(path, -1);
            boolean[] visited = new boolean[V];

            path[0] = 0; // Start from the first vertex
            if (hamiltonianCycleUtil(path, visited, 1)) {
                System.out.print("Hamiltonian cycle: ");
                for (int vertex : path) {
                    System.out.print(vertex + " ");
                }
                System.out.println(path[0]);
                return true;
            }

            System.out.println("No Hamiltonian cycle exists");
            return false;
        }

        boolean hamiltonianCycleUtil(int[] path, boolean[] visited, int pos) {
            if (pos == V) {
                return adjList[path[pos - 1]].contains(path[0]); // Check if the last vertex is adjacent to the first vertex
            }

            for (int v : adjList[path[pos - 1]]) {
                if (!visited[v]) {
                    visited[v] = true;
                    path[pos] = v;

                    if (hamiltonianCycleUtil(path, visited, pos + 1)) {
                        return true;
                    }

                    visited[v] = false;
                    path[pos] = -1;
                }
            }

            return false;
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 4);
        graph.addEdge(2, 4);
        graph.addEdge(3, 4);

        graph.isHamiltonianCycle();
    }
}
Write programs to implement solve ‘N’ Queens Problem using Backtracking.
import java.util.*;

public class NQueens {
    public static void main(String[] args) {
        int n = 4; // Number of queens
        solveNQueens(n);
    }

    public static void solveNQueens(int n) {
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.'); // Initialize the board with empty spaces
        }

        List<List<String>> solutions = new ArrayList<>();
        solveNQueensUtil(board, 0, solutions);
        
        // Print all solutions
        for (List<String> solution : solutions) {
            for (String row : solution) {
                System.out.println(row);
            }
            System.out.println();
        }
    }

    public static void solveNQueensUtil(char[][] board, int col, List<List<String>> solutions) {
        int n = board.length;
        if (col == n) {
            solutions.add(constructSolution(board));
            return;
        }

        for (int row = 0; row < n; row++) {
            if (isSafe(board, row, col)) {
                board[row][col] = 'Q'; // Place the queen
                solveNQueensUtil(board, col + 1, solutions); // Recur for next column
                board[row][col] = '.'; // Backtrack
            }
        }
    }

    public static boolean isSafe(char[][] board, int row, int col) {
        int n = board.length;

        // Check if there is a queen in the same row
        for (int i = 0; i < col; i++) {
            if (board[row][i] == 'Q') {
                return false;
            }
        }

        // Check upper diagonal on left side
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        // Check lower diagonal on left side
        for (int i = row, j = col; i < n && j >= 0; i++, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        return true;
    }

    public static List<String> constructSolution(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(String.valueOf(row));
        }
        return solution;
    }
}
Write a program in C/C++/ Java to find out solution for 0/1 knapsack problem.
public class KnapsackDynamicProgramming {
    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;
        int n = values.length;

        int maxValue = knapsackDynamicProgramming(values, weights, capacity, n);
        System.out.println("Maximum value that can be obtained: " + maxValue);
    }

    public static int knapsackDynamicProgramming(int[] values, int[] weights, int capacity, int n) {
        int[][] dp = new int[n + 1][capacity + 1];

        // Build the dp table using dynamic programming
        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (i == 0 || w == 0) {
                    dp[i][w] = 0;
                } else if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        return dp[n][capacity];
    }
}
Write programs to implement Graph Coloring Algorithm.
import java.util.*;

public class GraphColoring {
    static class Graph {
        int V;
        List<Integer>[] adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList[V];
            for (int i = 0; i < V; i++) {
                adjList[i] = new ArrayList<>();
            }
        }

        void addEdge(int src, int dest) {
            adjList[src].add(dest);
            adjList[dest].add(src); // For undirected graph
        }

        void colorGraph() {
            int[] result = new int[V];
            Arrays.fill(result, -1); // Initialize all vertices with no color

            boolean[] availableColors = new boolean[V];
            Arrays.fill(availableColors, true); // All colors are initially available

            // Assign colors to remaining V-1 vertices
            for (int u = 1; u < V; u++) {
                for (int neighbor : adjList[u]) {
                    if (result[neighbor] != -1) {
                        availableColors[result[neighbor]] = false; // Mark the color of neighbor as unavailable
                    }
                }

                // Find the first available color
                int color;
                for (color = 0; color < V; color++) {
                    if (availableColors[color]) {
                        break;
                    }
                }

                result[u] = color; // Assign the found color to vertex u

                // Reset the availability of colors for the next vertex
                Arrays.fill(availableColors, true);
            }

            // Print the result
            System.out.println("Vertex \t Color");
            for (int u = 0; u < V; u++) {
                System.out.println(u + " \t " + result[u]);
            }
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);

        System.out.println("Coloring of graph vertices:");
        graph.colorGraph();
    }
}

Write a program in C/C++/ Java to find out live node, E node and dead node from a given graph.
import java.util.*;

public class NodeClassification {
    static class Graph {
        int V;
        List<List<Integer>> adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList<>(V);
            for (int i = 0; i < V; i++) {
                adjList.add(new ArrayList<>());
            }
        }

        void addEdge(int src, int dest) {
            adjList.get(src).add(dest);
            adjList.get(dest).add(src); // For undirected graph
        }

        void classifyNodes() {
            boolean[] visited = new boolean[V];
            int[] degree = new int[V];

            for (int i = 0; i < V; i++) {
                degree[i] = adjList.get(i).size();
            }

            System.out.println("Node \t Classification");
            for (int i = 0; i < V; i++) {
                if (degree[i] > 0) {
                    System.out.print(i + " \t Live node");
                    if (degree[i] == 1) {
                        System.out.println(", E-node");
                    } else {
                        System.out.println();
                    }
                } else {
                    System.out.println(i + " \t Dead node");
                }
            }
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);

        System.out.println("Classification of nodes:");
        graph.classifyNodes();
    }
}


Write programs to determine if a given graph is a Hamiltonian cycle or Not.

import java.util.*;

public class HamiltonianCycle {
    static class Graph {
        int V;
        List<Integer>[] adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList[V];
            for (int i = 0; i < V; i++) {
                adjList[i] = new ArrayList<>();
            }
        }

        void addEdge(int src, int dest) {
            adjList[src].add(dest);
            adjList[dest].add(src); // For undirected graph
        }

        boolean isHamiltonianCycle() {
            boolean[] visited = new boolean[V];
            List<Integer> path = new ArrayList<>();

            // Try each vertex as a starting point
            for (int v = 0; v < V; v++) {
                Arrays.fill(visited, false);
                path.clear();
                if (isHamiltonianCycleUtil(v, visited, path, 1)) {
                    System.out.println("Hamiltonian cycle found:");
                    for (int vertex : path) {
                        System.out.print(vertex + " ");
                    }
                    System.out.println(path.get(0)); // Complete the cycle
                    return true;
                }
            }
            System.out.println("No Hamiltonian cycle exists.");
            return false;
        }

        boolean isHamiltonianCycleUtil(int v, boolean[] visited, List<Integer> path, int count) {
            visited[v] = true;
            path.add(v);

            if (count == V) {
                // Check if the last vertex is adjacent to the first vertex
                if (adjList[v].contains(path.get(0))) {
                    return true;
                } else {
                    path.remove(path.size() - 1);
                    visited[v] = false;
                    return false;
                }
            }

            for (int neighbor : adjList[v]) {
                if (!visited[neighbor]) {
                    if (isHamiltonianCycleUtil(neighbor, visited, path, count + 1)) {
                        return true;
                    }
                }
            }

            // Backtrack
            path.remove(path.size() - 1);
            visited[v] = false;
            return false;
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(5);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);

        System.out.println("Checking if the graph contains a Hamiltonian cycle:");
        graph.isHamiltonianCycle();
    }
}
Write a program in C/C++/ Java to show board configuration of 4 queens problem.
public class NQueens {
    public static void main(String[] args) {
        int n = 4;
        int[][] board = new int[n][n];
        solveNQueens(board, 0);
        printBoard(board);
    }

    public static boolean solveNQueens(int[][] board, int col) {
        if (col >= board.length) {
            return true;
        }

        for (int i = 0; i < board.length; i++) {
            if (isSafe(board, i, col)) {
                board[i][col] = 1;
                if (solveNQueens(board, col + 1)) {
                    return true;
                }
                board[i][col] = 0; // Backtrack
            }
        }
        return false;
    }

    public static boolean isSafe(int[][] board, int row, int col) {
        for (int i = 0; i < col; i++) {
            if (board[row][i] == 1) {
                return false; // Queen is present in the same row
            }
        }

        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 1) {
                return false; // Queen is present in the upper left diagonal
            }
        }

        for (int i = row, j = col; i < board.length && j >= 0; i++, j--) {
            if (board[i][j] == 1) {
                return false; // Queen is present in the lower left diagonal
            }
        }

        return true;
    }

    public static void printBoard(int[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                System.out.print((board[i][j] == 1) ? "Q " : "- ");
            }
            System.out.println();
        }
    }
}
Write programs to implement for finding Topological sorting and determine the time complexity for the same.
import java.util.*;

public class TopologicalSort {
    static class Graph {
        int V;
        List<List<Integer>> adjList;

        Graph(int V) {
            this.V = V;
            adjList = new ArrayList<>(V);
            for (int i = 0; i < V; i++) {
                adjList.add(new ArrayList<>());
            }
        }

        void addEdge(int src, int dest) {
            adjList.get(src).add(dest);
        }

        Stack<Integer> topologicalSort() {
            Stack<Integer> stack = new Stack<>();
            boolean[] visited = new boolean[V];

            for (int i = 0; i < V; i++) {
                if (!visited[i]) {
                    dfs(i, visited, stack);
                }
            }

            return stack;
        }

        void dfs(int v, boolean[] visited, Stack<Integer> stack) {
            visited[v] = true;

            for (int neighbor : adjList.get(v)) {
                if (!visited[neighbor]) {
                    dfs(neighbor, visited, stack);
                }
            }

            stack.push(v);
        }
    }

    public static void main(String[] args) {
        Graph graph = new Graph(6);
        graph.addEdge(5, 2);
        graph.addEdge(5, 0);
        graph.addEdge(4, 0);
        graph.addEdge(4, 1);
        graph.addEdge(2, 3);
        graph.addEdge(3, 1);

        System.out.println("Topological sorting:");
        Stack<Integer> result = graph.topologicalSort();
        while (!result.isEmpty()) {
            System.out.print(result.pop() + " ");
        }
    }
}
Write a program in C/C++/ Java to solve N Queens Problem using Backtracking.
import java.util.*;

public class NQueens {
    public static void main(String[] args) {
        int n = 4; // Change this to the desired board size
        solveNQueens(n);
    }

    public static void solveNQueens(int n) {
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.'); // Initialize the board with empty spaces
        }

        List<List<String>> solutions = new ArrayList<>();
        solveNQueensUtil(board, 0, solutions);
        
        // Print all solutions
        for (List<String> solution : solutions) {
            for (String row : solution) {
                System.out.println(row);
            }
            System.out.println();
        }
    }

    public static void solveNQueensUtil(char[][] board, int col, List<List<String>> solutions) {
        int n = board.length;
        if (col == n) {
            solutions.add(constructSolution(board));
            return;
        }

        for (int row = 0; row < n; row++) {
            if (isSafe(board, row, col)) {
                board[row][col] = 'Q'; // Place the queen
                solveNQueensUtil(board, col + 1, solutions); // Recur for next column
                board[row][col] = '.'; // Backtrack
            }
        }
    }

    public static boolean isSafe(char[][] board, int row, int col) {
        int n = board.length;

        // Check if there is a queen in the same row
        for (int i = 0; i < col; i++) {
            if (board[row][i] == 'Q') {
                return false;
            }
        }

        // Check upper diagonal on left side
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        // Check lower diagonal on left side
        for (int i = row, j = col; i < n && j >= 0; i++, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        return true;
    }

    public static List<String> constructSolution(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(String.valueOf(row));
        }
        return solution;
    }
}

