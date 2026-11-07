import torch
import torch.nn as nn
import torch.nn.functional as F
# Import LayerNorm for the advanced model
from torch_geometric.nn import GATv2Conv
from torch.nn import LayerNorm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from my_code.data_preparation.bn_dataset_in_memoy import PrecomputedBayesianNetworkDataset
import pysmile
import pysmile_license


class Gatv2(torch.nn.Module):


    def __init__(self, dim_in, dim_h, dim_out, num_hid_layers=3, dropout_rate=0.5, heads=8):
        super(Gatv2, self).__init__()
        self.dropout_rate = dropout_rate
        self.gat_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dim_out = dim_out

        # Input layer
        self.gat_layers.append(GATv2Conv(dim_in, dim_h, heads=heads, concat=True, dropout=dropout_rate))
        self.norms.append(LayerNorm(dim_h * heads))
        self.res_projection1 = nn.Linear(dim_in, dim_h * heads) if dim_in != (dim_h * heads) else None

        # Hidden layers
        for _ in range(num_hid_layers - 1):
            self.gat_layers.append(GATv2Conv(dim_h * heads, dim_h, heads=heads, concat=True, dropout=dropout_rate))
            self.norms.append(LayerNorm(dim_h * heads))

        # Final MLP prediction head
        self.final_classifier = nn.Sequential(
            nn.Linear(dim_h * heads, dim_h),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_h, dim_out)
        )

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
        h = x
        for i, (layer, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_in = h
            h = layer(h, edge_index)
            if i == 0 and self.res_projection1:
                h = h + self.res_projection1(h_in)
            else:
                h = h + h_in
            h = norm(h)
            h = F.leaky_relu(h, 0.01)
        h = self.final_classifier(h)

        col_indices = torch.arange(self.dim_out, device=x.device).unsqueeze(0)
        mask = col_indices >= num_states.unsqueeze(1)
        h[mask] = -1e9
        return F.log_softmax(h, dim=1)

    def fit(self, tr_loader, vl_loader, epochs, lr, weight_decay, model_path):
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)

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

            if epoch % 10 == 0:
                print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_results['loss']:.4f}, Val Cosine: {val_results['cosine_similarity']:.4f}")

    @torch.no_grad()
    def evaluate(self, loader, criterion):
        """Evaluates the model on a given data loader."""
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
            'pearson_correlation': total_pearson_corr / total_nodes if total_nodes > 0 else 0,
        }

    def test(self, test_loader):
        criterion = nn.KLDivLoss(reduction='batchmean')
        test_results = self.evaluate(test_loader, criterion)
        print(f'\n--- Test Results ---')
        print(f'  Avg. Test KLDivLoss: {test_results["loss"]:.4f}')
        print(f'  Avg. Test Cosine Similarity: {test_results["cosine_similarity"]:.4f}')
        print(f'  Avg. Test Pearson Correlation: {test_results["pearson_correlation"]:.4f}')
        print(f'--------------------')
        return test_results


def analyze_single_node_prediction(model, loader, device):
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
    NUM_EPOCHS = 150
    TEST_RATIO = 0.2
    TRAIN_RATIO = 0.6
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.6
    HIDDEN_DIM = 64
    NUM_HID_LAYERS = 3
    NUM_HEADS = 8
    MODEL_SAVE_PATH = "gatv2_best.pth"

    # Setup
    moc_network = pysmile.Network()
    root_directory = "../../data/Datasets/50"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Data loading
        full_dataset = PrecomputedBayesianNetworkDataset(
            root=root_directory, bn_network=moc_network, num_samples=1, max_state_count=1
        )
        full_dataset = full_dataset[:500]
        num_of_samples = len(full_dataset)

        train_size = int(num_of_samples * TRAIN_RATIO)
        test_size = int(num_of_samples * TEST_RATIO)
        val_size = num_of_samples - test_size - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Model Instantiation
        model = Gatv2(
            dim_in=full_dataset.num_features,
            dim_h=HIDDEN_DIM,
            dim_out=full_dataset.num_features,
            num_hid_layers=NUM_HID_LAYERS,
            dropout_rate=DROPOUT_RATE,
            heads=NUM_HEADS
        ).to(device)

        print("\nModel Architecture:")
        print(model)

        # --- Training ---
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        model.fit(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_SAVE_PATH)
        print("\nFinished training.")

        # --- Testing ---
        print(f"\nLoading best model from: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.test(test_loader)

        # --- Single Node Analysis ---
        analyze_single_node_prediction(model, test_loader, device)

    except (NameError, FileNotFoundError, Exception) as e:
        print("\n---")
        print("Could not run the full script due to a missing dependency or error.")
        print(f"Error: {e}")
        print("Please ensure all libraries are installed and data paths are correct.")
        print("---")