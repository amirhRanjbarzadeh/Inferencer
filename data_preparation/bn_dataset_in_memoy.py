import torch
import pysmile
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Dict, Tuple


class PrecomputedBayesianNetworkDataset(InMemoryDataset):

    def __init__(self, root: str, bn_network: pysmile.Network, max_state_count: int,
                 num_samples: int, evidence_strategy: str = "random",
                 min_num_of_evidence: int = 0, max_num_of_evidence: int = 3,
                 transform=None, pre_transform=None):

        self.bn_network = bn_network
        self.max_state_count = max_state_count
        self.num_samples = num_samples
        self.evidence_strategy = evidence_strategy
        self.min_num_of_evidence = min_num_of_evidence
        self.max_num_of_evidence = max_num_of_evidence

        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        print(f"Processing: Generating {self.num_samples} unique samples for the first time...")

        self.bn_network.set_bayesian_algorithm(pysmile.BayesianAlgorithmType(8))
        self.node_handles = self.bn_network.get_all_nodes()
        self.num_nodes = len(self.node_handles)
        self.handle_to_idx = {handle: i for i, handle in enumerate(self.node_handles)}
        self.edge_index = self._get_edge_index()
        self.node_state_counts = torch.tensor(
            [self.bn_network.get_outcome_count(h) for h in self.node_handles],
            dtype=torch.long
        )

        # --- 2. Evidence Sampling Strategy Setup ---
        if self.evidence_strategy == 'degree_weighted':
            degrees = [len(self.bn_network.get_children(h) + self.bn_network.get_parents(h)) for h in self.node_handles]
            total_weight = sum(degrees)
            if total_weight == 0 or self.num_nodes == 0:
                self.sampling_weights = [1 / self.num_nodes] * self.num_nodes if self.num_nodes > 0 else []
            else:
                self.sampling_weights = [w / total_weight for w in degrees]

        self.priors = self._get_all_priors()

        data_list = []
        seen_evidence_signatures = set()
        pbar = tqdm(total=self.num_samples, desc="Generating unique samples")
        max_attempts = self.num_samples * 20
        attempts = 0

        while len(data_list) < self.num_samples and attempts < max_attempts:
            data_sample, evidence_dict = self._generate_sample_internal()
            signature = frozenset(evidence_dict.items())

            if signature not in seen_evidence_signatures:
                seen_evidence_signatures.add(signature)
                data_list.append(data_sample)
                pbar.update(1)
            attempts += 1
        pbar.close()

        if len(data_list) < self.num_samples:
            print(f"\nWarning: Could only generate {len(data_list)} unique samples.")

        self.save(data_list, self.processed_paths[0])
        print(f"Dataset processed and saved to {self.processed_paths[0]}")

    def _get_all_priors(self) -> torch.Tensor:
        """Calculates the prior probability distribution for every node."""
        self.bn_network.clear_all_evidence()
        self.bn_network.update_beliefs()
        all_priors = torch.zeros((self.num_nodes, self.max_state_count), dtype=torch.float)
        for i, handle in enumerate(self.node_handles):
            beliefs = self.bn_network.get_node_value(handle)
            num_states = self.node_state_counts[i].item()
            if num_states > 0:
                all_priors[i, :num_states] = torch.tensor(beliefs, dtype=torch.float)
        return all_priors

    def _generate_sample_internal(self) -> Tuple[Data, Dict]:
        self.bn_network.clear_all_evidence()
        num_evidence_nodes = np.random.randint(self.min_num_of_evidence, self.max_num_of_evidence + 1)

        if num_evidence_nodes > 0 and self.num_nodes > 0:
            if self.evidence_strategy == 'random':
                evidence_node_indices = np.random.choice(self.num_nodes, num_evidence_nodes, replace=False)
            elif self.evidence_strategy == 'degree_weighted':
                evidence_node_indices = np.random.choice(self.num_nodes, num_evidence_nodes, p=self.sampling_weights,
                                                         replace=False)
        else:
            evidence_node_indices = []

        evidence_dict = {}
        for node_idx in evidence_node_indices:
            handle = self.node_handles[node_idx]
            state_count = self.node_state_counts[node_idx].item()
            if state_count > 0:
                evidence_state = np.random.randint(0, state_count)
                self.bn_network.set_evidence(handle, evidence_state)
                evidence_dict[node_idx] = evidence_state

        self.bn_network.update_beliefs()
        x_features = self.priors.clone()
        y_targets = torch.zeros((self.num_nodes, self.max_state_count), dtype=torch.float)

        for node_idx, evidence_state in evidence_dict.items():
            x_features[node_idx, :] = 0.0
            x_features[node_idx, evidence_state] = 1.0

        for i, handle in enumerate(self.node_handles):
            beliefs = self.bn_network.get_node_value(handle)
            num_states = self.node_state_counts[i].item()
            if num_states > 0:
                y_targets[i, :num_states] = torch.tensor(beliefs, dtype=torch.float)

        is_observed = torch.zeros(self.num_nodes, dtype=torch.bool)
        if evidence_dict:
            is_observed[list(evidence_dict.keys())] = True

        data_object = Data(x=x_features, edge_index=self.edge_index, y=y_targets,
                           is_observed=is_observed, num_states=self.node_state_counts)
        return data_object, evidence_dict

    def _get_edge_index(self) -> torch.Tensor:
        """Builds the edge_index tensor from the Bayesian network structure."""
        edge_list = []
        for parent_handle in self.node_handles:
            parent_idx = self.handle_to_idx[parent_handle]
            for child_handle in self.bn_network.get_children(parent_handle):
                child_idx = self.handle_to_idx[child_handle]
                edge_list.append([parent_idx, child_idx])
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


if __name__ == "__main__":
    from my_code.network_creator.genie_network_creator import create_network
    import pandas as pd

    structure_file = "../../data/out_structure.csv"
    cpt_directory = "../../data/CPTs/out_structure"
    root_directory = "../../data/Datasets/50"
    output_xdsl_file = "xdsl_files/test.xdsl"

    network, node_handle = create_network(structure_file, cpt_directory, output_xdsl_file, generate_input_file=True,
                                          log_flag=False)
    df = pd.read_csv(structure_file)
    max_number_of_states = df["state_count"].max()

    dataset_1 = PrecomputedBayesianNetworkDataset(
        root=root_directory,
        bn_network=network,
        max_state_count=max_number_of_states,
        num_samples=5000
    )
    print(f"Dataset 1 loaded. Length: {len(dataset_1)}")
    print(f"First sample's features shape: {dataset_1[0].x.shape}")

    print(dataset_1[0].x)
    print(dataset_1[1].x)

