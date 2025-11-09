import streamlit as st
import random
from typing import List, Tuple, Dict, Any

# --- CONFIGURATION ---
MAX_ONES_TARGET = 50
MAX_FITNESS_RETURN = 80.0
DEFAULT_LENGTH = 80
DEFAULT_POP_SIZE = 300
DEFAULT_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.01

# --- GENETIC ALGORITHM CORE FUNCTIONS ---

def calculate_fitness(individual: List[int], target_ones: int, max_fitness_value: float, length: int) -> float:
    num_ones = sum(individual)
    if num_ones == target_ones:
        return max_fitness_value
    deviation = abs(num_ones - target_ones)
    fitness = max_fitness_value * (1 - (deviation / length) ** 2)
    return max(0.0, fitness)

def create_individual(length: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(length)]

def initialize_population(size: int, length: int) -> List[List[int]]:
    return [create_individual(length) for _ in range(size)]

def selection(population: List[List[int]], fitnesses: List[float], pop_size: int, elite_count: int) -> List[List[int]]:
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    next_population = [ind for ind, fit in sorted_population[:elite_count]]
    tournament_size = 3
    for _ in range(pop_size - elite_count):
        candidates = random.sample(sorted_population, tournament_size)
        winner = max(candidates, key=lambda x: x[1])[0]
        next_population.append(winner)
    return next_population

def crossover(parent1: List[int], parent2: List[int], length: int) -> Tuple[List[int], List[int]]:
    point = random.randint(1, length - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual: List[int], mutation_rate: float) -> List[int]:
    return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

def run_ga(pop_size: int, length: int, target_ones: int, max_return: float, generations: int, mutation_rate: float) -> Dict[str, Any]:
    random.seed(42)
    population = initialize_population(pop_size, length)
    best_individual = []
    best_fitness = -1.0
    elite_count = max(1, int(0.05 * pop_size))

    for generation in range(generations):
        fitnesses = [calculate_fitness(ind, target_ones, max_return, length) for ind in population]
        max_fit = max(fitnesses)
        best_index = fitnesses.index(max_fit)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_individual = population[best_index]
        if best_fitness >= max_return:
            break
        mating_pool = selection(population, fitnesses, pop_size, elite_count)
        new_population = []
        for i in range(0, pop_size, 2):
            p1 = mating_pool[i]
            p2 = mating_pool[i + 1] if i + 1 < pop_size else mating_pool[i]
            child1, child2 = crossover(p1, p2, length)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, mutation_rate))
        population = new_population

    return {
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'final_generation': generation + 1,
        'target_reached': best_fitness >= max_return
    }

# --- STREAMLIT UI ---

st.set_page_config(page_title="Genetic Algorithm Bit Pattern Search", page_icon="🧬", layout="wide")

st.title("🧬 Genetic Algorithm Bit Pattern Search")

st.markdown("""
Genetic Algorithms (GAs) are heuristic search techniques inspired by natural selection. 
They solve optimization problems through five key phases:
- **Initialisation**: Generate a population of random solutions.
- **Fitness Assignment**: Evaluate how good each solution is.
- **Selection**: Choose the best candidates for reproduction.
- **Reproduction**: Create new solutions via crossover and mutation.
- **Termination**: Stop when the optimal solution is found or after a set number of generations.
""")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ GA Parameters")
    st.markdown("---")
    pop_size = st.number_input("Population Size", min_value=10, value=DEFAULT_POP_SIZE, step=10)
    length = st.number_input("Chromosome Length", min_value=10, value=DEFAULT_LENGTH, step=10)
    target_ones = st.number_input("Target # of Ones", min_value=1, value=MAX_ONES_TARGET, step=1)
    max_return = st.number_input("Max Fitness Value", min_value=1.0, value=MAX_FITNESS_RETURN)
    generations = st.number_input("Number of Generations", min_value=1, value=DEFAULT_GENERATIONS, step=1)
    mutation_rate = st.slider("Mutation Rate", min_value=0.001, max_value=0.1, value=DEFAULT_MUTATION_RATE, step=0.001)
    run_btn = st.button("▶️ Run Genetic Algorithm")

# --- EXECUTION ---
if run_btn:
    if target_ones > length:
        st.error("Target # of Ones cannot exceed Chromosome Length.")
    else:
        with st.spinner(f"Running {generations} Generations..."):
            results = run_ga(pop_size, length, target_ones, max_return, generations, mutation_rate)

        st.success("Algorithm Complete!")

        final_ones = sum(results['best_individual'])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target # of Ones", MAX_ONES_TARGET)
        col2.metric("Best Fitness Score", f"{results['best_fitness']:.2f}")
        col3.metric("Ones Found", final_ones, delta=final_ones - MAX_ONES_TARGET)
        col4.metric("Generations", results['final_generation'])

        if results['target_reached']:
            st.balloons()
            st.success(f"🎯 Goal Achieved! Found an individual with {MAX_ONES_TARGET} ones.")

        st.markdown("#### Optimal Bit Pattern")
        st.code("".join(map(str, results['best_individual'])))
else:
    st.info("Adjust parameters in the sidebar and click ▶️ to begin.")