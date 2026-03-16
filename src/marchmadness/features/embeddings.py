"""Team embeddings learned from game-level data using a neural network.

Approach: Train a small neural network to predict game outcomes from
team pair IDs + game context. The team embedding layer learns latent
representations that capture team strength, style, and dynamics.

Then extract the learned embeddings as features for the tournament model.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TeamEmbeddingModel(nn.Module):
    """Learn team embeddings from game outcomes."""

    def __init__(self, n_teams: int, embedding_dim: int = 32,
                 n_context_features: int = 0):
        super().__init__()
        self.team_embedding = nn.Embedding(n_teams, embedding_dim)
        # Input: concat of two team embeddings + optional context features
        input_dim = 2 * embedding_dim + n_context_features
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, team_a_idx, team_b_idx, context=None):
        emb_a = self.team_embedding(team_a_idx)
        emb_b = self.team_embedding(team_b_idx)
        x = torch.cat([emb_a, emb_b], dim=-1)
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.head(x).squeeze(-1)

    def get_embeddings(self):
        """Return the learned embedding matrix as numpy array."""
        return self.team_embedding.weight.detach().cpu().numpy()


def build_game_dataset(data: dict[str, pd.DataFrame], seasons: list[int],
                       gender: str = "M") -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Build training data for the embedding model from regular season games.

    Returns:
        team_id_map: dict mapping TeamID -> index
        team_a_indices: array of team A indices
        team_b_indices: array of team B indices
        labels: array of outcomes (1 if team A won, 0 otherwise)
    """
    results_key = f"{gender}RegularSeasonCompactResults"
    results = data[results_key]

    # Filter to requested seasons
    games = results[results["Season"].isin(seasons)]

    # Build team ID -> index mapping
    all_teams = sorted(set(games["WTeamID"].tolist() + games["LTeamID"].tolist()))
    team_id_map = {tid: idx for idx, tid in enumerate(all_teams)}

    team_a_indices = []
    team_b_indices = []
    labels = []

    for _, game in games.iterrows():
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        # Always order as (lower ID, higher ID) for consistency
        a = min(w_id, l_id)
        b = max(w_id, l_id)
        label = 1 if w_id == a else 0

        team_a_indices.append(team_id_map[a])
        team_b_indices.append(team_id_map[b])
        labels.append(label)

    return (
        team_id_map,
        np.array(team_a_indices),
        np.array(team_b_indices),
        np.array(labels, dtype=np.float32),
    )


def train_embeddings(data: dict[str, pd.DataFrame], season: int,
                     gender: str = "M", embedding_dim: int = 32,
                     n_epochs: int = 50, lr: float = 0.001,
                     n_recent_seasons: int = 5) -> tuple[dict, np.ndarray]:
    """Train team embeddings on recent regular season games.

    Args:
        data: Loaded datasets
        season: Target season (use this season's games for training)
        gender: M or W
        embedding_dim: Size of team embedding vectors
        n_epochs: Training epochs
        lr: Learning rate
        n_recent_seasons: How many recent seasons to include

    Returns:
        team_id_map: dict mapping TeamID -> embedding index
        embeddings: numpy array of shape (n_teams, embedding_dim)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for embeddings. Install with: pip install torch")

    seasons = list(range(season - n_recent_seasons + 1, season + 1))
    team_id_map, a_idx, b_idx, labels = build_game_dataset(data, seasons, gender)
    n_teams = len(team_id_map)

    print(f"  Training embeddings: {n_teams} teams, {len(labels)} games, "
          f"seasons {seasons[0]}-{seasons[-1]}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeamEmbeddingModel(n_teams, embedding_dim).to(device)

    # Create dataset
    a_tensor = torch.LongTensor(a_idx).to(device)
    b_tensor = torch.LongTensor(b_idx).to(device)
    y_tensor = torch.FloatTensor(labels).to(device)

    dataset = TensorDataset(a_tensor, b_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Train
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        for batch_a, batch_b, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_a, batch_b)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    embeddings = model.get_embeddings()
    return team_id_map, embeddings


def compute(data: dict[str, pd.DataFrame], season: int,
            gender: str = "M", embedding_dim: int = 32) -> pd.DataFrame:
    """Compute team embedding features for a given season.

    Returns DataFrame with columns: [TeamID, Emb_0, Emb_1, ..., Emb_{dim-1}]
    """
    if not HAS_TORCH:
        return pd.DataFrame(columns=["TeamID"])

    try:
        team_id_map, embeddings = train_embeddings(
            data, season, gender, embedding_dim=embedding_dim
        )
    except Exception as e:
        print(f"  Embedding training failed: {e}")
        return pd.DataFrame(columns=["TeamID"])

    # Build DataFrame
    rows = []
    for team_id, idx in team_id_map.items():
        row = {"TeamID": team_id}
        for i in range(embedding_dim):
            row[f"Emb_{i}"] = embeddings[idx, i]
        rows.append(row)

    return pd.DataFrame(rows)
