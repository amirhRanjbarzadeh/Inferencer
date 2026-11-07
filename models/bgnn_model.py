import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os

try:
    from my_code.data_preparation.bn_dataset_in_memoy import PrecomputedBayesianNetworkDataset
    import pysmile
    import pysmile_license

    IMPORTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("Warning: Custom imports not found. The script requires a dummy class to run.")
    IMPORTS_AVAILABLE = False

if not IMPORTS_AVAILABLE:
    from torch_geometric.data import Dataset, Data


    class PrecomputedBayesianNetworkDataset(Dataset):
        def __init__(self, root=None, bn_network=None, num_samples=1, max_state_count=1):
            super().__init__(root)
            self.num_features = 10

        def len(self): return 100

        def get(self, idx):
            x = torch.randn(5, 10)
            y = F.softmax(torch.rand(5, 10), dim=1)
            edge_index = torch.randint(0, 5, (2, 6))
            is_observed = torch.tensor([True, True, False, True, False])
            num_states = torch.randint(2, 10, (5,))
            return Data(x=x, y=y, edge_index=edge_index, is_observed=is_observed, num_states=num_states)


    class pysmile:
        class Network: pass


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_log_sigma, -10)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_log_sigma, -10)

    def forward(self, x):
        # Sample weights and biases
        weight_eps = torch.randn_like(self.weight_log_sigma)
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_eps

        bias_eps = torch.randn_like(self.bias_log_sigma)
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_eps

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        kl_weight = -self.weight_log_sigma + 0.5 * (torch.exp(2 * self.weight_log_sigma) + self.weight_mu.pow(2)) - 0.5
        kl_bias = -self.bias_log_sigma + 0.5 * (torch.exp(2 * self.bias_log_sigma) + self.bias_mu.pow(2)) - 0.5
        return kl_weight.sum() + kl_bias.sum()


class BayesianGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BayesianGNNLayer, self).__init__()
        self.linear = BayesianLinear(dim_in, dim_out)

    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x

    def kl_divergence(self):
        return self.linear.kl_divergence()


class BayesianGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_hid_layer=2):
        super(BayesianGNN, self).__init__()
        self.hid_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dim_out = dim_out

        # Input layer
        self.hid_layers.append(BayesianGNNLayer(dim_in, dim_h))
        self.norms.append(nn.LayerNorm(dim_h))
        self.input_res_projection = nn.Linear(dim_in, dim_h) if dim_in != dim_h else None

        # Hidden layers
        for _ in range(num_hid_layer - 1):
            self.hid_layers.append(BayesianGNNLayer(dim_h, dim_h))
            self.norms.append(nn.LayerNorm(dim_h))

        # Output layer
        self.out_layer = BayesianGNNLayer(dim_h, dim_out)

    @staticmethod
    def pearson_correlation(x, y, eps=1e-8):
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        cov = (x_centered * y_centered).sum(dim=-1)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=-1))
        y_std = torch.sqrt((y_centered ** 2).sum(dim=-1))
        corr = cov / (x_std * y_std + eps)
        return corr

    def forward(self, x, edge_index, num_states):
        num_nodes = x.size(0)
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        vals = torch.ones(edge_index_with_loops.shape[1], device=x.device)
        adj = torch.sparse_coo_tensor(edge_index_with_loops, vals, (num_nodes, num_nodes))

        h_in = x
        for i, (layer, norm) in enumerate(zip(self.hid_layers, self.norms)):
            h_res = layer(h_in, adj)
            if i == 0 and self.input_res_projection:
                h_res = h_res + self.input_res_projection(h_in)
            else:
                h_res = h_res + h_in
            h_out = F.gelu(norm(h_res))
            h_in = h_out
        h = self.out_layer(h_in, adj)

        col_indices = torch.arange(self.dim_out, device=x.device).unsqueeze(0)
        mask = col_indices >= num_states.unsqueeze(1)
        h[mask] = -1e9
        return F.log_softmax(h, dim=1)

    def kl_divergence(self):
        total_kl = sum(layer.kl_divergence() for layer in self.hid_layers)
        total_kl += self.out_layer.kl_divergence()
        return total_kl

    def fit(self, tr_loader, vl_loader, epochs, kl_annealing_epochs, final_kl_beta, lr, model_path):
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

        num_batches_in_train = len(tr_loader)
        best_val_loss = float('inf')

        print("\nStarting Bayesian GNN Training...")
        for epoch in range(1, epochs + 1):
            self.train()
            total_recon_loss, total_kl_loss, total_unobserved_nodes = 0, 0, 0
            kl_weight = final_kl_beta * min(1.0, epoch / kl_annealing_epochs)

            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                unobserved_mask = ~batch.is_observed
                num_unobserved_in_batch = unobserved_mask.sum().item()
                if num_unobserved_in_batch == 0: continue

                out_log_probs = self(batch.x, batch.edge_index, batch.num_states)
                recon_loss = criterion(out_log_probs[unobserved_mask], batch.y[unobserved_mask])
                kl_loss = self.kl_divergence()
                kl_term = kl_weight * (kl_loss / num_batches_in_train)
                total_loss = recon_loss + kl_term
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_recon_loss += recon_loss.item() * num_unobserved_in_batch
                total_kl_loss += kl_loss.item()
                total_unobserved_nodes += num_unobserved_in_batch

            avg_recon_loss = total_recon_loss / total_unobserved_nodes if total_unobserved_nodes > 0 else 0

            # Validation step
            val_results = self.test(vl_loader, num_mc_samples=25, verbose=False)
            scheduler.step(val_results['loss'])

            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                torch.save(self.state_dict(), model_path)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch: {epoch:03d} | Recon Loss: {avg_recon_loss:.4f} | "
                      f"Val Loss: {val_results['loss']:.4f} | Val Cosine: {val_results['cosine_similarity']:.4f} | Pearson Correlation: {val_results['pearson_correlation']:.4f}")

    @torch.no_grad()
    def test(self, loader, num_mc_samples=100, verbose=True):
        self.eval()
        criterion_none = nn.KLDivLoss(reduction='none')
        all_mc_losses, all_mc_cosines, all_mc_pearsons = [], [], []

        for batch in loader:
            batch = batch.to(device)
            unobserved_mask = ~batch.is_observed
            if not unobserved_mask.any(): continue

            mc_log_probs = torch.stack(
                [self(batch.x, batch.edge_index, batch.num_states)[unobserved_mask] for _ in range(num_mc_samples)],
                dim=0
            )
            true_probs = batch.y[unobserved_mask]
            mc_pred_probs = torch.exp(mc_log_probs)
            loss_per_node_and_sample = criterion_none(mc_log_probs.flatten(0, 1),
                                                      true_probs.repeat(num_mc_samples, 1)).sum(dim=1).view(
                num_mc_samples, -1)
            all_mc_losses.append(loss_per_node_and_sample.cpu())
            cosine_per_node_and_sample = F.cosine_similarity(mc_pred_probs, true_probs.unsqueeze(0), dim=2)
            all_mc_cosines.append(cosine_per_node_and_sample.cpu())
            pearson_per_node_and_sample = self.pearson_correlation(mc_pred_probs, true_probs.unsqueeze(0))
            all_mc_pearsons.append(pearson_per_node_and_sample.cpu())

        if not all_mc_losses:
            if verbose: print("\n--- No unobserved nodes in the test set ---")
            return {'loss': 0, 'loss_std': 0, 'cosine_similarity': 0, 'cosine_similarity_std': 0}

        all_mc_losses = torch.cat(all_mc_losses, dim=1)
        all_mc_cosines = torch.cat(all_mc_cosines, dim=1)
        all_mc_pearsons = torch.cat(all_mc_pearsons, dim=1)
        avg_loss_per_sample = all_mc_losses.mean(dim=1)
        avg_cosine_per_sample = all_mc_cosines.mean(dim=1)
        avg_pearson_per_sample = all_mc_pearsons.mean(dim=1)
        results = {
            'loss': avg_loss_per_sample.mean().item(), 'loss_std': avg_loss_per_sample.std().item(),
            'cosine_similarity': avg_cosine_per_sample.mean().item(),
            'cosine_similarity_std': avg_cosine_per_sample.std().item(),
            'pearson_correlation': avg_pearson_per_sample.mean().item(),
            'pearson_correlation_std': avg_pearson_per_sample.std().item()
        }
        if verbose:
            print(f'\n--- Test Results ({num_mc_samples} MC Samples) ---')
            print(f'  Avg. KLDivLoss:        {results["loss"]:.4f} ± {results["loss_std"]:.4f}')
            print(
                f'  Avg. Cosine Similarity:  {results["cosine_similarity"]:.4f} ± {results["cosine_similarity_std"]:.4f}')
            print(
                f'  Avg. Pearson Correlation:  {results["pearson_correlation"]:.4f} ± {results["pearson_correlation_std"]:.4f}')
            print(f'---------------------------------------------')
        return results


def analyze_single_node_prediction(model, loader, device, num_mc_samples):
    """Loads a Bayesian model, selects one node, and prints its predictive distribution."""
    model.eval()
    try:
        test_batch = next(iter(loader)).to(device)
    except StopIteration:
        print("\nTest loader is empty. Cannot perform analysis.")
        return

    unobserved_indices_in_batch = torch.where(~test_batch.is_observed)[0]
    if len(unobserved_indices_in_batch) == 0:
        print("\nThe first batch of the test set contains no unobserved nodes. Cannot perform analysis.")
        return

    torch.manual_seed(999)
    random_selector = torch.randint(0, len(unobserved_indices_in_batch), (1,)).item()
    node_idx_in_batch = unobserved_indices_in_batch[random_selector].item()
    node_num_states = test_batch.num_states[node_idx_in_batch].item()
    true_probs_for_node = test_batch.y[node_idx_in_batch][:node_num_states]

    print(f"\n--- Analyzing Single Node Prediction (Node index {node_idx_in_batch} in batch) ---")
    print(f"This node has {node_num_states} possible states. Running {num_mc_samples} MC samples.")

    with torch.no_grad():
        mc_predictions = []
        for _ in range(num_mc_samples):
            pred_log_probs = model(test_batch.x, test_batch.edge_index, test_batch.num_states)
            pred_probs_for_node = torch.exp(pred_log_probs[node_idx_in_batch])[:node_num_states]
            mc_predictions.append(pred_probs_for_node)

        mc_predictions_tensor = torch.stack(mc_predictions)
        mean_pred_probs = mc_predictions_tensor.mean(dim=0)
        std_pred_probs = mc_predictions_tensor.std(dim=0)

    results_df = pd.DataFrame({
        "State": [f"State {i}" for i in range(node_num_states)],
        "True Probability": true_probs_for_node.cpu().numpy(),
        "Mean Predicted Probability": mean_pred_probs.cpu().numpy(),
        "Prediction Uncertainty (Std Dev)": std_pred_probs.cpu().numpy()
    })
    results_df.set_index("State", inplace=True)

    pd.options.display.float_format = '{:.4f}'.format
    print("\nComparison of Probabilities:")
    print(results_df)
    print("---------------------------------------------------\n")


if __name__ == '__main__':
    # --- Configuration ---
    NUM_EPOCHS = 150
    KL_ANNEALING_EPOCHS = 50
    FINAL_KL_BETA = 1e-4
    TEST_RATIO = 0.2
    TRAIN_RATIO = 0.6
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 64
    NUM_HID_LAYERS = 3
    NUM_MC_SAMPLES_TEST = 100
    MODEL_SAVE_PATH = "bayesian_gnn_best.pth"

    # --- Setup ---
    moc_network = pysmile.Network()
    root_directory = "../../data/Datasets/50"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    full_dataset = PrecomputedBayesianNetworkDataset(
        root=root_directory, bn_network=moc_network, num_samples=1, max_state_count=1
    )
    train_size = int(len(full_dataset) * TRAIN_RATIO)
    test_size = int(len(full_dataset) * TEST_RATIO)
    val_size = len(full_dataset) - train_size - test_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Model Instantiation ---
    model = BayesianGNN(
        dim_in=full_dataset.num_features,
        dim_h=HIDDEN_DIM,
        dim_out=full_dataset.num_features,
        num_hid_layer=NUM_HID_LAYERS
    ).to(device)
    print("\nBayesian GNN Model Architecture:")
    print(model)

    # --- Training ---
    model.fit(train_loader, val_loader, NUM_EPOCHS, KL_ANNEALING_EPOCHS, FINAL_KL_BETA, LEARNING_RATE, MODEL_SAVE_PATH)
    print(f"\nFinished training.")

    # --- Testing ---
    print(f"\nLoading best model from: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.test(test_loader, num_mc_samples=NUM_MC_SAMPLES_TEST, verbose=True)

    # --- Single Node Analysis ---
    analyze_single_node_prediction(model, test_loader, device, num_mc_samples=NUM_MC_SAMPLES_TEST)

