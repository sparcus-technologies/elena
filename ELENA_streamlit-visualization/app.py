import streamlit as st
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
import json
import folium
from streamlit_folium import st_folium

class Individual:
    def __init__(self, route=None):
        self.route = route
        self._fitness = None

    def __lt__(self, other):
        return self.fitness()[0] < other.fitness()[0]

    def __gt__(self, other):
        return self.fitness()[0] > other.fitness()[0]

    def __eq__(self, other):
        return self.fitness()[0] == other.fitness()[0]

    def fitness(self):
        if self._fitness is None:
            distance = sum(dist_matrix[self.route[i]][self.route[(i + 1) % len(self.route)]]
                         for i in range(len(self.route)))
            self._fitness = (distance, self.route)
        return self._fitness

    def crossover(self, other):
        size = len(self.route)
        child1_route = [-1] * size
        child2_route = [-1] * size

        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)

        child1_route[start:end + 1] = self.route[start:end + 1]
        child2_route[start:end + 1] = other.route[start:end + 1]

        self._fill_route(child1_route, other.route, start, end)
        self._fill_route(child2_route, self.route, start, end)

        return Individual(child1_route), Individual(child2_route)

    def _fill_route(self, child_route, parent_route, start, end):
        size = len(child_route)
        used = set(child_route[start:end + 1])
        pos = (end + 1) % size

        for i in range(size):
            if child_route[pos] == -1:
                for city in parent_route[pos:] + parent_route[:pos]:
                    if city not in used:
                        child_route[pos] = city
                        used.add(city)
                        break
            pos = (pos + 1) % size

    def mutate(self, mutation_rate=0.01):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(self.route)), 2)
            self.route[i], self.route[j] = self.route[j], self.route[i]
            self._fitness = None

def calculate_route(points):
    global dist_matrix
    points_array = np.array(points)
    
    # Function to calculate distance in kilometers between two lat/long points
    def haversine_distance(point1, point2):
        lat1, lon1 = point1
        lat2, lon2 = point2
        R = 6371  # Earth's radius in kilometers

        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance

    # Create distance matrix using haversine distance
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = haversine_distance(points[i], points[j])

    best_solution = evolve_population(
        pop_size=50,
        generations=100,
        early_stopping_generations=10,
        num_subpops=3
    )

    distance, route = best_solution.fitness()
    return route, distance

def create_population(pop_size, route_length):
    population = []
    for _ in range(pop_size):
        route = list(range(route_length))
        random.shuffle(route)
        population.append(Individual(route))
    return population

def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return min(tournament)

def evolve_population(pop_size=50, generations=100, early_stopping_generations=10,
                     num_subpops=3, migration_interval=10, migration_size=2):
    route_length = len(dist_matrix)
    subpops = [create_population(pop_size // num_subpops, route_length) for _ in range(num_subpops)]
    best_fitness = float('inf')
    generations_without_improvement = 0
    best_individual = None

    for gen in range(generations):
        for i in range(num_subpops):
            new_population = []
            elite = min(subpops[i])
            new_population.append(deepcopy(elite))

            while len(new_population) < len(subpops[i]):
                parent1 = tournament_selection(subpops[i])
                parent2 = tournament_selection(subpops[i])
                child1, child2 = parent1.crossover(parent2)
                child1.mutate()
                child2.mutate()
                new_population.extend([child1, child2])

            subpops[i] = new_population[:len(subpops[i])]

        current_best = min(min(pop) for pop in subpops)
        current_fitness = current_best.fitness()[0]

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= early_stopping_generations:
            break

    return best_individual

def create_map_with_points(points, route=None, center=None):
    # Set map center to the first point or default location
    if center:
        m = folium.Map(location=center, zoom_start=13)
    elif points:
        m = folium.Map(location=points[0], zoom_start=13)
    else:
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # Add markers for points
    for i, point in enumerate(points):
        folium.CircleMarker(
            point,
            radius=8,
            popup=f"Point {i + 1}",
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)
    
    # Draw route if it exists
    if route is not None and len(route) > 0:
        route_points = []
        for i in route:
            route_points.append(points[i])
        route_points.append(points[route[0]])  # Close the loop
        folium.PolyLine(
            route_points,
            weight=3,
            color='blue',
            opacity=0.8
        ).add_to(m)
    
    return m

def main():
    st.set_page_config(page_title="TSP Solver", layout="wide")
    st.title("ELENA Implementation")

    # Initialize session state
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'route' not in st.session_state:
        st.session_state.route = None
    if 'distance' not in st.session_state:
        st.session_state.distance = None
    if 'map_center' not in st.session_state:
        st.session_state.map_center = None
    if 'map_key' not in st.session_state:
        st.session_state.map_key = 0

    # Create two columns
    col1, col2 = st.columns([3, 1])

    with col2:
        st.write(f"Points marked: {len(st.session_state.points)}/20")
        st.write("Instructions:")
        st.write("1. Click on the map to add points")
        st.write("2. Add at least 3 points")
        st.write("3. Click 'Calculate Route' when ready")
        
        if st.button("Calculate Route", disabled=len(st.session_state.points) < 3):
            with st.spinner("Calculating optimal route..."):
                st.session_state.route, st.session_state.distance = calculate_route(st.session_state.points)
                st.success("Route calculated!")
                st.write(f"Total distance: {st.session_state.distance:.2f} km")

        if st.button("Reset Map"):
            st.session_state.points = []
            st.session_state.route = None
            st.session_state.distance = None
            st.session_state.map_center = None
            st.session_state.map_key += 1

        # Display current distance if it exists
        if st.session_state.distance is not None:
            st.write("---")
            st.write(f"Current route distance: {st.session_state.distance:.2f} km")

    with col1:
        # Create map
        m = create_map_with_points(
            st.session_state.points, 
            st.session_state.route,
            st.session_state.map_center
        )
        
        # Display map with reduced update frequency
        map_data = st_folium(
            m,
            height=600,
            width=None,
            key=f"map_{st.session_state.map_key}",
            returned_objects=["last_clicked"]
        )

        # Handle map clicks
        if map_data["last_clicked"]:
            click_lat = map_data["last_clicked"]["lat"]
            click_lng = map_data["last_clicked"]["lng"]
            new_point = [click_lat, click_lng]
            
            # Check if this is a new point
            if len(st.session_state.points) < 20:
                if not st.session_state.points or new_point != st.session_state.points[-1]:
                    st.session_state.points.append(new_point)
                    st.session_state.map_center = new_point
                    st.session_state.map_key += 1

if __name__ == "__main__":
    main()