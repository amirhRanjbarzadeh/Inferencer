import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

# Assuming these are your custom imports
from my_code.data_preparation.bn_dataset_in_memoy import PrecomputedBayesianNetworkDataset
import pysmile
import pysmile_license


class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(VanillaGNNLayer, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)
        return x


class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_hid_layer=2, dropout_rate=0.5):
        super(VanillaGNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.hid_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dim_out = dim_out

        # Input layer
        self.hid_layers.append(VanillaGNNLayer(dim_in, dim_h))
        self.batch_norms.append(nn.BatchNorm1d(dim_h))
        self.input_res_projection = nn.Linear(dim_in, dim_h) if dim_in != dim_h else None

        # Hidden layers
        for _ in range(num_hid_layer - 1):
            self.hid_layers.append(VanillaGNNLayer(dim_h, dim_h))
            self.batch_norms.append(nn.BatchNorm1d(dim_h))

        # Output layer
        self.out_layer = VanillaGNNLayer(dim_h, dim_out)

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

        h = x
        for i, (layer, bn) in enumerate(zip(self.hid_layers, self.batch_norms)):
            h_in = h
            h = layer(h, adj)
            h = bn(h)
            if i == 0 and self.input_res_projection:
                h = h + self.input_res_projection(h_in)
            else:
                h = h + h_in
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.out_layer(h, adj)

        col_indices = torch.arange(self.dim_out, device=x.device).unsqueeze(0)
        mask = col_indices >= num_states.unsqueeze(1)
        h[mask] = -1e9
        return F.log_softmax(h, dim=1)

    def fit(self, tr_loader, vl_loader, epochs, lr, weight_decay, model_path):
        """Trains the model and saves the best version."""
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.train()
            total_train_loss, total_unobserved_nodes = 0, 0
            for batch in tr_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                unobserved_mask = ~batch.is_observed
                num_unobserved = unobserved_mask.sum().item()
                if num_unobserved == 0: continue
                out_log_probs = self(batch.x, batch.edge_index, batch.num_states)
                loss = criterion(out_log_probs[unobserved_mask], batch.y[unobserved_mask])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * num_unobserved
                total_unobserved_nodes += num_unobserved
            avg_train_loss = total_train_loss / total_unobserved_nodes if total_unobserved_nodes > 0 else 0

            val_results = self.evaluate(vl_loader, criterion)
            scheduler.step(val_results['loss'])

            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                torch.save(self.state_dict(), model_path)


            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_results['loss']:.4f}, Val Cosine: {val_results['cosine_similarity']:.4f} | Pearson Correlation: {val_results['pearson_correlation']:.4f}")

    @torch.no_grad()
    def evaluate(self, loader, criterion):
        self.eval()
        total_loss, total_cosine_sim, total_pearson_corr, total_nodes = 0, 0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            unobserved_mask = ~batch.is_observed
            num_unobserved = unobserved_mask.sum().item()
            if num_unobserved == 0: continue
            out_log_probs = self(batch.x, batch.edge_index, batch.num_states)
            loss = criterion(out_log_probs[unobserved_mask], batch.y[unobserved_mask])
            total_loss += loss.item() * num_unobserved
            pred_probs = torch.exp(out_log_probs[unobserved_mask])
            true_probs = batch.y[unobserved_mask]
            total_cosine_sim += F.cosine_similarity(pred_probs, true_probs, dim=1).sum().item()
            total_pearson_corr += self.pearson_correlation(pred_probs, true_probs).sum().item()
            total_nodes += num_unobserved
        return {
            'loss': total_loss / total_nodes if total_nodes > 0 else 0,
            'cosine_similarity': total_cosine_sim / total_nodes if total_nodes > 0 else 0,
            'pearson_correlation': total_pearson_corr / total_nodes if total_nodes > 0 else 0
        }

    def test(self, test_loader):
        criterion = nn.KLDivLoss(reduction='batchmean')
        test_results = self.evaluate(test_loader, criterion)
        print(f'\n--- Test Results ---')
        print(f"  Avg. Test KLDivLoss: {test_results['loss']:.4f}")
        print(f"  Avg. Test Cosine Similarity: {test_results['cosine_similarity']:.4f}")
        print(f"  Avg. Test Pearson Correlation: {test_results['pearson_correlation']:.4f}")
        print(f'--------------------')
        return test_results


def analyze_single_node_prediction(model, loader, device):
    """Loads a model, selects one node, and prints its probability distribution."""
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
    print(f"This node has {node_num_states} possible states.")

    with torch.no_grad():
        pred_log_probs = model(test_batch.x, test_batch.edge_index, test_batch.num_states)

    pred_probs_for_node = torch.exp(pred_log_probs[node_idx_in_batch])[:node_num_states]

    results_df = pd.DataFrame({
        "State": [f"State {i}" for i in range(node_num_states)],
        "True Probability": true_probs_for_node.cpu().numpy(),
        "Predicted Probability": pred_probs_for_node.cpu().numpy()
    })
    results_df.set_index("State", inplace=True)

    pd.options.display.float_format = '{:.4f}'.format

    print("\nComparison of Probabilities:")
    print(results_df)
    print("---------------------------------------------------\n")


if __name__ == '__main__':
    # Hyperparameters
    NUM_EPOCHS = 150
    TEST_RATIO = 0.2
    TRAIN_RATIO = 0.6
    BATCH_SIZE = 256
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.4
    HIDDEN_DIM = 64
    NUM_HID_LAYERS = 4
    MODEL_SAVE_PATH = "vanilla_gnn_best.pth"

    # Setup
    moc_network = pysmile.Network()
    root_directory = "../../data/Datasets/50"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Loading
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

    # Model Instantiation
    model = VanillaGNN(
        dim_in=full_dataset.num_features,
        dim_h=HIDDEN_DIM,
        dim_out=full_dataset.num_features,
        num_hid_layer=NUM_HID_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    print("\nModel Architecture:")
    print(model)

    # --- Training ---
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    model.fit(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_SAVE_PATH)
    print(f"\nFinished training.")

    # --- Testing ---
    print(f"\nLoading best model from: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.test(test_loader)

    # --- Single Node Analysis ---
    analyze_single_node_prediction(model, test_loader, device)