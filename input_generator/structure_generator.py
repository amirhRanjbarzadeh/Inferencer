import os
import random
from pathlib import Path
import pandas as pd
import numpy as np

# --- Constants ---
NUMBER_OF_NODES = 50
EDGE_PROBABILITY = 0.3
MAX_PARENTS = 10
MIN_STATES = 2
MAX_STATES = 4
NETWORK_NAME = 'out_structure'
BASE_DATA_DIR = Path("../../data")


def generate_cpt(num_states: int, num_parent_states: int) -> pd.DataFrame:
    cpt_values = np.random.dirichlet(np.ones(num_states), size=num_parent_states).T
    state_names = [f"State{i + 1}" for i in range(num_states)]
    parent_state_cols = [f"P(ParentComb_{i + 1})" for i in range(num_parent_states)]
    df = pd.DataFrame(cpt_values, columns=parent_state_cols)
    df.insert(0, 'State', state_names)
    return df


def create_random_dag(num_nodes: int, edge_prob: float, max_parents: int):
    nodes_data = []
    connected_nodes = set()

    for i in range(num_nodes):
        node_id = i + 1
        num_states = np.random.randint(MIN_STATES, MAX_STATES + 1)

        potential_parents = [p_id for p_id in range(1, node_id) if np.random.uniform(0, 1) < edge_prob]

        if len(potential_parents) > max_parents:
            parents = random.sample(potential_parents, k=max_parents)
        else:
            parents = potential_parents

        if parents:
            connected_nodes.add(node_id)
            for p_id in parents:
                connected_nodes.add(p_id)

        nodes_data.append({
            "node_id": node_id,
            "node_name": f"Node{node_id:0>3}",
            "parents_id": ";".join(map(str, sorted(parents))),
            "state_count": num_states
        })

    all_node_ids = list(range(1, num_nodes + 1))
    for i in range(num_nodes):
        if nodes_data[i]['node_id'] not in connected_nodes:
            possible_parents = all_node_ids[:i]
            if possible_parents:
                parent_id = random.choice(possible_parents)
                nodes_data[i]['parents_id'] = str(parent_id)

    return pd.DataFrame(nodes_data)


def main():
    cpt_directory = BASE_DATA_DIR / "CPTs" / NETWORK_NAME
    cpt_directory.mkdir(parents=True, exist_ok=True)

    print("Generating network structure...")
    network_df = create_random_dag(NUMBER_OF_NODES, EDGE_PROBABILITY, MAX_PARENTS)

    print("Generating CPTs...")
    state_count_map = pd.Series(network_df.state_count.values, index=network_df.node_id).to_dict()

    for row in network_df.itertuples():
        parent_ids_str = row.parents_id

        if not parent_ids_str:
            num_parent_states = 1
        else:
            parent_ids = [int(p) for p in parent_ids_str.split(';')]
            parent_state_counts = [state_count_map[p_id] for p_id in parent_ids]
            num_parent_states = np.prod(parent_state_counts)

        cpt = generate_cpt(row.state_count, int(num_parent_states))
        cpt_path = cpt_directory / f"{row.node_id}_CPT.csv"
        cpt.to_csv(cpt_path, index=False, header=False)

    print(f"Generated {len(network_df)} CPTs in '{cpt_directory}'")

    structure_path = BASE_DATA_DIR / f"{NETWORK_NAME}.csv"
    network_df.to_csv(structure_path, index=False)
    print(f"Network structure saved to '{structure_path}'")


if __name__ == "__main__":
    main()
