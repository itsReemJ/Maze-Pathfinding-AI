import cv2
import numpy as np
import queue
import math

# Define movement directions
MOVEMENT_DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# Function to check if a position is within the map boundaries
def is_within_bounds(position, map_shape):
    row, col = position
    num_rows, num_cols = map_shape
    return 0 <= row < num_rows and 0 <= col < num_cols

# Function to check if a position is unoccupied (white)
def is_unoccupied(position, map_image):
    row, col = position
    return map_image[row, col] == 255

# Function to get the neighbors of a position
def get_neighbors(position, map_shape):
    neighbors = []
    for direction in MOVEMENT_DIRECTIONS:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if is_within_bounds(neighbor, map_shape):
            neighbors.append(neighbor)
    return neighbors

# Breadth First Search (BFS)
def bfs(map_image, start_position, target_position):
    map_shape = map_image.shape[:2] 
    visited = np.zeros(map_shape, dtype=bool)
    parent = {}
    queue = []

    queue.append(start_position)
    visited[start_position] = True #mark as visited

    while queue:
        current_position = queue.pop(0)
        if current_position == target_position:
            break

        neighbors = get_neighbors(current_position, map_shape)
        for neighbor in neighbors:
            if not visited[neighbor] and is_unoccupied(neighbor, map_image):
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current_position

    path = []
    current_position = target_position
    while current_position != start_position:
        path.append(current_position)
        current_position = parent[current_position]
    path.append(start_position)
    path.reverse()

    return path

# Depth First Search (DFS)
def dfs(map_image, start_position, target_position):
    map_shape = map_image.shape[:2]
    visited = np.zeros(map_shape, dtype=bool)
    parent = {}
    stack = []

    stack.append(start_position)

    while stack:
        current_position = stack.pop()
        if current_position == target_position:
            break

        if not visited[current_position]:
            visited[current_position] = True

            neighbors = get_neighbors(current_position, map_shape)
            for neighbor in neighbors:
                if not visited[neighbor] and is_unoccupied(neighbor, map_image):
                    stack.append(neighbor)
                    parent[neighbor] = current_position

    path = []
    current_position = target_position
    while current_position != start_position:
        path.append(current_position)
        current_position = parent[current_position]
    path.append(start_position)
    path.reverse()

    return path

# Greedy Best-first Search
def greedy_best_first_search(map_image, start_position, target_position, heuristic):
    map_shape = map_image.shape[:2]
    visited = np.zeros(map_shape, dtype=bool)
    parent = {}
    priority_queue = queue.PriorityQueue()

    priority_queue.put((0, start_position))
    visited[start_position] = True

    while not priority_queue.empty():
        _, current_position = priority_queue.get()
        if current_position == target_position:
            break

        neighbors = get_neighbors(current_position, map_shape)
        for neighbor in neighbors:
            if not visited[neighbor] and is_unoccupied(neighbor, map_image):
                visited[neighbor] = True
                priority = heuristic(neighbor, target_position)
                priority_queue.put((priority, neighbor))
                parent[neighbor] = current_position

    path = []
    current_position = target_position
    while current_position != start_position:
        path.append(current_position)
        current_position = parent[current_position]
    path.append(start_position)
    path.reverse()

    return path

# A* Search
def a_star_search(map_image, start_position, target_position, heuristic):
    map_shape = map_image.shape[:2]#2D Map (hxw)
    visited = np.zeros(map_shape, dtype=bool)#keep track of visited positions
    parent = {}
    cost = {}

    priority_queue = queue.PriorityQueue() #store positions based on their priority

    cost[start_position] = 0
    priority_queue.put((0, start_position)) #start priority of 0
    visited[start_position] = True

    while not priority_queue.empty():
        _, current_position = priority_queue.get() #_,  placeholder
        if current_position == target_position:
            break #reached

        neighbors = get_neighbors(current_position, map_shape)
        for neighbor in neighbors: #Iterates through each neighbor and checks whether it is unvisited
            if not visited[neighbor] and is_unoccupied(neighbor, map_image):
                new_cost = cost[current_position] + 1
                
                if neighbor not in cost or new_cost < cost[neighbor]: #new cost is lower than the existing cost
                    cost[neighbor] = new_cost #update
                    priority = new_cost + heuristic(neighbor, target_position) #calculate
                    priority_queue.put((priority, neighbor)) 
                    visited[neighbor] = True
                    parent[neighbor] = current_position #reconstructing the path

    path = []
    current_position = target_position #loop that backtracks
    while current_position != start_position:
        path.append(current_position)
        current_position = parent[current_position]
    path.append(start_position) #complete the path, reverse order
    path.reverse() #corect order

    return path



# Manhattan distance heuristic
def manhattan_distance(position, target_position): #sum of the absolute differences of their coordinates
    return abs(position[0] - target_position[0]) + abs(position[1] - target_position[1])

# Euclidean distance heuristic
def euclidean_distance(position, target_position):
    return math.sqrt((position[0] - target_position[0])**2 + (position[1] - target_position[1])**2)

# Load the map image
map_image = cv2.imread("MAZE.jfif", cv2.IMREAD_GRAYSCALE)
map_image = map_image.astype(np.uint8)

# Function to let the user choose the search method
def choose_search_method():
    print("Choose a search method:")
    print("1. BFS (Breadth First Search)")
    print("2. DFS (Depth First Search)")
    print("3. GBFS (Greedy Best-first Search)")
    print("4. A* (A-star Search)")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    if choice == '1':
        return 'BFS'
    elif choice == '2':
        return 'DFS'
    elif choice == '3':
        return 'GBFS'
    elif choice == '4':
        return 'A*'
    else:
        print("Invalid choice. Defaulting to BFS.")
        return 'BFS'

# Get user input for start and target positions
start_row = int(input("Enter the starting row: "))
start_col = int(input("Enter the starting column: "))
target_row = int(input("Enter the target row: "))
target_col = int(input("Enter the target column: "))

# Set the start and target positions
start_position = (start_row, start_col)
target_position = (target_row, target_col)

# Get user choice for search method
search_method = choose_search_method()

# Find the optimal path based on the user's choice
if search_method == 'BFS':
    optimal_path = bfs(map_image, start_position, target_position)
elif search_method == 'DFS':
    optimal_path = dfs(map_image, start_position, target_position)
elif search_method == 'GBFS':
    optimal_path = greedy_best_first_search(map_image, start_position, target_position, manhattan_distance)
elif search_method == 'A*':
    optimal_path = a_star_search(map_image, start_position, target_position, manhattan_distance)
else:
    print("Invalid search method. Defaulting to BFS.")
    optimal_path = bfs(map_image, start_position, target_position)

print(f"{search_method} Path:", optimal_path)

# Convert the map image to RGB format
map_image_rgb = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)

# Draw the path on the map image in red and slightly thicker
for i in range(len(optimal_path) - 1):
    start_point = (optimal_path[i][1], optimal_path[i][0])
    end_point = (optimal_path[i + 1][1], optimal_path[i + 1][0])
    cv2.line(map_image_rgb, start_point, end_point, (0, 0, 255), 2)  # Adjust thickness here
    cv2.circle(map_image_rgb, start_point, 0, (0, 255, 0), -1)  # Set radius to 0 for a point

# Draw the start and target positions
cv2.circle(map_image_rgb, (start_position[1], start_position[0]), 3, (0, 255, 0), -1)  # Set radius to 0 for a point
cv2.circle(map_image_rgb, (target_position[1], target_position[0]), 3, (0, 0, 255), -1)  # Set radius to 0 for a point

# Display the map image with the path, start, and target positions
cv2.imshow("Map with Path", map_image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
