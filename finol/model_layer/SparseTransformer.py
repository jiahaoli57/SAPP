import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SparseAttention(nn.Module):
    """
    Implements the Strided Sparse Attention mechanism from Child et al. (2019).
    This version only includes the strided pattern.
    """

    def __init__(self, d_model, n_head, seq_len, stride, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.seq_len = seq_len
        self.stride = stride

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        mask = self._create_sparse_mask(seq_len, stride)
        self.register_buffer("mask", mask)

    def _create_sparse_mask(self, seq_len, stride):
        """Creates a symmetric strided sparse attention mask."""
        mask = torch.full((seq_len, seq_len), float('-inf'))

        for i in range(seq_len):
            for j in range(0, i + 1, stride):
                # Symmetrically allow attention
                mask[i, j] = 0
                mask[j, i] = 0
            # Ensure self-attention is always allowed
            mask[i, i] = 0

        # torch.set_printoptions(threshold=1000000)
        # print(mask)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, v):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out_linear(context)


class SparseTransformerLayer(nn.Module):
    """
    A single layer of the Sparse Transformer (for time-series feature extraction).
    """

    def __init__(self, d_model, n_head, d_ff, seq_len, stride, dropout=0.1):
        super().__init__()
        self.self_attn = SparseAttention(d_model, n_head, seq_len, stride, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class CIDM(nn.Module):
    """
    This class implements the Correlation Information Decision Module (CIDM).
    It models the inter-asset relationships using a full self-attention mechanism.
    This is the user's provided implementation.
    """

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        hidden_size = self.model_params["HIDDEN_SIZE"]
        # In the paper, the feedforward layer size for scoring is q_s. We use hidden_size*2 for simplicity.
        q_s = int(hidden_size * 2)

        # Layer normalization for the input asset representations
        self.rep_ln = nn.LayerNorm(hidden_size)

        # Linear layers to generate Query, Key, and Value for attention
        self.linear_query = nn.Linear(hidden_size, hidden_size)
        self.linear_key = nn.Linear(hidden_size, hidden_size)
        self.linear_value = nn.Linear(hidden_size, hidden_size)

        # Feed-forward network to produce the final scores (s_i)
        # The paper uses two weight matrices for scoring, W1 and W2.
        # Here we use a two-layer MLP which is equivalent.
        self.ff1 = nn.Linear(hidden_size, q_s)
        self.ff2 = nn.Linear(q_s, 1)

    def quick_gelu(self, x):
        """A fast approximation of the GELU activation function."""
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x):
        """
        Forward pass for the CIDM module.

        :param x: Input tensor R of shape `(batch_size, num_assets, HIDDEN_SIZE)`.
        :return: Output tensor final_scores of shape `(batch_size, num_assets)`.
        """
        # Apply layer normalization
        x = self.rep_ln(x)

        # Generate Query, Key, Value matrices
        query = self.linear_query(x)  # Shape: (batch_size, num_assets, HIDDEN_SIZE)
        key = self.linear_key(x)  # Shape: (batch_size, num_assets, HIDDEN_SIZE)
        value = self.linear_value(x)  # Shape: (batch_size, num_assets, HIDDEN_SIZE)

        # Calculate attention scores
        beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)  # Shape: (batch_size, num_assets, num_assets, 1)

        # Apply attention to the value matrix
        x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # Shape: (batch_size, num_assets, HIDDEN_SIZE)

        # Pass through the feed-forward network to get final_scores
        final_scores = self.quick_gelu(self.ff1(x))
        final_scores = self.ff2(final_scores).squeeze(-1)  # Shape: (batch_size, num_assets)

        return final_scores


class SparseTransformer(nn.Module):
    """
    Sparse Transformer (Child et al., 2019) baseline model for portfolio management.
    This model now uses the user's exact CIDM implementation for fair comparison.
    """

    def __init__(self, model_args, model_params):
        super().__init__()
        self.model_args = model_args
        self.model_params = model_params

        # --- Shared Parameters ---
        d_model = self.model_params["HIDDEN_SIZE"]
        dropout = self.model_params.get("DROPOUT", 0.1)

        # --- Sparse Transformer Configuration ---
        n_head_st = self.model_params["NUM_HEADS"]
        n_layers_st = self.model_params["NUM_LAYERS"]
        d_ff_st = self.model_params["HIDDEN_SIZE"]
        window_size = self.model_args["window_size"]
        stride = self.model_params.get("SPARSITY_STRIDE", 32)

        # STB Layers: Temporal Feature Extraction
        self.token_emb = nn.Linear(model_args["num_features_original"], d_model)
        self.pos_emb_stb = nn.Embedding(window_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.stb_layers = nn.ModuleList([
            SparseTransformerLayer(d_model, n_head_st, d_ff_st, window_size, stride, dropout)
            for _ in range(n_layers_st)
        ])

        # --- CIDM (Correlation Information Decision Module) ---
        # Instantiating the user's provided CIDM class
        self.cidm = CIDM(self.model_params)

    def forward(self, x):
        """
        Forward pass of the Sparse Transformer Baseline model.

        :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
                  `num_features_augmented` is `window_size * num_features_original`.
        :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape
        # --- Input Transformation ---
        # Reshape input to separate window_size and features
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
        # Combine batch and asset dimensions for STB processing
        x = rearrange(x, "b m n d -> (b m) n d")

        token_embeddings = self.token_emb(x)
        positions = torch.arange(0, self.model_args["window_size"], device=x.device).unsqueeze(0)
        positional_embeddings = self.pos_emb_stb(positions)

        x_emb = token_embeddings + positional_embeddings
        x_emb = self.emb_dropout(x_emb)

        for layer in self.stb_layers:
            x_emb = layer(x_emb)

        # Use the representation of the last time step
        asset_representation = x_emb[:, -1, :]

        # Reshape for CIDM: (batch_size, num_assets, d_model)
        asset_representation = asset_representation.view(batch_size, num_assets, -1)

        # --- 2. CIDM Stage (Cross-Asset Correlation Modeling) ---
        final_scores = self.cidm(asset_representation)

        return final_scores

