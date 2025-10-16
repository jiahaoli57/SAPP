import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ==========================================================================================
# 1. CIDM Module (As provided in your reference code)
# This module remains unchanged.
# ==========================================================================================
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


# ==========================================================================================
# 2. DRSA Temporal Feature Extractor
# This section remains unchanged.
# ==========================================================================================
class RoutingAttention(nn.Module):
    """Standard Multi-Head Attention Mechanism."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, P, K, H_prime, W, C = q.shape
        _, _, _, H_prime_kv, _, _ = k.shape  # k and v have same shape

        q = self.W_q(q).view(B, P, K, H_prime, W, self.n_heads, self.d_k).permute(0, 5, 1, 2, 3, 4, 6)
        k = self.W_k(k).view(B, P, K, H_prime_kv, W, self.n_heads, self.d_k).permute(0, 5, 1, 2, 3, 4, 6)
        v = self.W_v(v).view(B, P, K, H_prime_kv, W, self.n_heads, self.d_k).permute(0, 5, 1, 2, 3, 4, 6)

        q = rearrange(q, 'b h p k hp w dk -> b h p (k hp w) dk')
        k = rearrange(k, 'b h p k hp w dk -> b h p (k hp w) dk')
        v = rearrange(v, 'b h p k hp w dk -> b h p (k hp w) dk')

        scores = torch.einsum('bhpqd, bhpkd -> bhpqk', q, k) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.einsum('bhpqk, bhpkd -> bhpqd', attn_weights, v)
        context = rearrange(context, 'b h p (k hp w) dk -> b p k hp w (h dk)', h=self.n_heads, k=K, hp=H_prime, w=W)

        return self.W_o(context)


class DRSATemporalBlock(nn.Module):
    """
    Implements the core logic of DRSA: Dimension Conversion, Blocking, and
    the Dynamic Routing Filter (DRF) module.
    """

    def __init__(self, d_model, n_heads, P, K, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.P = P  # Number of partitions (blocks)
        self.K = K  # Top-K for routing

        self.W_q_proj = nn.Linear(d_model, d_model)
        self.W_k_proj = nn.Linear(d_model, d_model)
        self.W_v_proj = nn.Linear(d_model, d_model)

        self.routing_attention = RoutingAttention(d_model, n_heads, dropout)
        self.global_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_2d):
        # x_2d shape: (B, H, W, C) where B = batch_size * num_assets, C = d_model
        B, H, W, C = x_2d.shape
        H_prime = H // self.P  # Height of each partition

        # 1. Blocking Mechanism (by rows)
        # Reshape to (B, P, H/P, W, C)
        x_p = x_2d.view(B, self.P, H_prime, W, C)

        # 2. Dynamic Routing Filter (DRF) Module
        # 2.1. Dynamic Routing Filtering Mechanism
        q_p = self.W_q_proj(x_p)
        k_p = self.W_k_proj(x_p)
        v_p = self.W_v_proj(x_p)

        # Compute local-level representations by averaging
        q_local = q_p.mean(dim=(2, 3))  # Shape: (B, P, C)
        k_local = k_p.mean(dim=(2, 3))  # Shape: (B, P, C)

        # Compute local relevance matrix S_L
        S_local = torch.matmul(q_local, k_local.transpose(-2, -1))  # Shape: (B, P, P)

        k_val = min(self.K, self.P)

        # Get dynamic routing index matrix R_L
        _, R_local = torch.topk(S_local, k_val, dim=-1)  # Shape: (B, P, k_val)

        # Filter K and V using R_L. We gather the top K most relevant BLOCKS for each block.
        k_p_flat = rearrange(k_p, 'b p hp w c -> b p (hp w) c')  # Shape: (B, P, H'*W, C)
        v_p_flat = rearrange(v_p, 'b p hp w c -> b p (hp w) c')  # Shape: (B, P, H'*W, C)

        # Adjust shapes for gathering based on the new k_val
        R_local_expanded = R_local.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k_p_flat.shape[2], C)

        k_filter = torch.gather(k_p_flat.unsqueeze(1).expand(-1, self.P, -1, -1, -1), 2, R_local_expanded)
        v_filter = torch.gather(v_p_flat.unsqueeze(1).expand(-1, self.P, -1, -1, -1), 2, R_local_expanded)

        k_filter = k_filter.view(B, self.P, k_val, H_prime, W, C)
        v_filter = v_filter.view(B, self.P, k_val, H_prime, W, C)

        q_unfiltered = q_p.unsqueeze(2).expand(-1, -1, k_val, -1, -1, -1)

        # 2.2. Routing Attention
        routing_out = self.routing_attention(q_unfiltered, k_filter, v_filter)
        routing_out = routing_out.mean(dim=2)

        x_p = self.norm1(x_p + self.dropout(routing_out))

        # 2.3. Global Convolution Path
        kv_global = (k_p + v_p).view(B, H, W, C)
        kv_global = rearrange(kv_global, 'b h w c -> b c h w')
        global_out = self.global_conv(kv_global)
        global_out = rearrange(global_out, 'b c h w -> b h w c').view(B, self.P, H_prime, W, C)

        output = self.norm2(x_p + self.dropout(global_out))

        return output.view(B, H, W, C)


# ==========================================================================================
# 3. Main DRSA Model Class
# This class is modified to support multiple layers.
# ==========================================================================================

class DRSA(nn.Module):
    """
    Replication of the DRSA model for portfolio management, structured for fair comparison
    with SAPP and SparseTransformer baselines.

    The model consists of two main components:
    1. DRSA Temporal Block: This module first converts the 1D time series for each asset
       into a 2D representation based on its dominant period, discovered via FFT. It then
       uses a Dynamic Routing Sparse Attention mechanism to efficiently extract temporal features.
       This block can be stacked multiple times.

    2. Correlation Information Decision Module (CIDM): This module takes the asset representations
       from the DRSA block and applies a full cross-asset self-attention mechanism to model
       inter-asset correlations, producing the final scores for portfolio allocation.

    :param model_args: Dictionary of model arguments (e.g., window_size, num_features_original).
    :param model_params: Dictionary of model hyperparameters (e.g., HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS).
    """

    def __init__(self, model_args, model_params):
        super().__init__()
        self.model_args = model_args
        self.model_params = model_params

        d_model = self.model_params["HIDDEN_SIZE"]
        dropout = self.model_params.get("DROPOUT", 0.1)

        # --- DRSA Configuration ---
        n_heads = self.model_params["NUM_HEADS"]
        self.P = self.model_params.get("DRSA_PARTITIONS", 8)
        self.K = self.model_params.get("DRSA_TOP_K", 4)
        ### ADDED ###
        # Get the number of layers, default to 1 if not specified
        n_layers = self.model_params.get("NUM_LAYERS", 1)

        # --- Model Layers ---
        # Embedding layer to project input features to d_model
        self.token_emb = nn.Linear(model_args["num_features_original"], d_model)

        ### MODIFIED ###
        # Create a ModuleList of DRSA blocks to be stacked
        self.drsa_layers = nn.ModuleList([
            DRSATemporalBlock(d_model, n_heads, self.P, self.K, dropout)
            for _ in range(n_layers)
        ])

        # Final layer to produce a single feature vector from the 2D representation
        self.output_proj = nn.Linear(d_model, d_model)

        # Correlation Information Decision Module (CIDM)
        self.cidm = CIDM(self.model_params)

    def forward(self, x):
        """
        Forward pass of the DRSA model.

        :param x: Input tensor of shape `(batch_size, num_assets, num_features_augmented)`.
                  `num_features_augmented` is `window_size * num_features_original`.
        :return: Output tensor of shape `(batch_size, num_assets)` containing the predicted scores for each asset.
        """
        batch_size, num_assets, _ = x.shape
        window_size = self.model_args["window_size"]
        num_features = self.model_args["num_features_original"]
        d_model = self.model_params["HIDDEN_SIZE"]

        # --- 1. Input Transformation ---
        x = x.view(batch_size, num_assets, window_size, num_features)
        x = rearrange(x, "b m n d -> (b m) n d")
        B_prime = x.shape[0]

        x_emb = self.token_emb(x)

        # --- 2. Dimension Conversion (1D -> 2D) ---
        with torch.no_grad():
            fft_result = torch.fft.rfft(x, dim=1)
            amplitudes = torch.abs(fft_result)
            mean_amplitudes = torch.mean(amplitudes, dim=(0, 2))
            freqs = torch.fft.rfftfreq(window_size)[1:]
            mean_amplitudes = mean_amplitudes[1:]
            top_freq_idx = torch.argmax(mean_amplitudes) + 1
            if top_freq_idx > 0:
                period = math.ceil(window_size / top_freq_idx.item())
            else:
                period = window_size // 2

        W = period
        pad_len = (W - (window_size % W)) % W
        padded_x = F.pad(x_emb, (0, 0, 0, pad_len))
        L_padded = padded_x.shape[1]
        H = L_padded // W

        # This check is now inside the forward pass, which is less ideal.
        # For a cleaner implementation, H should be fixed or pre-calculated.
        # But for now, we leave it to ensure it runs.
        current_P = self.P
        if H % current_P != 0:
            valid_P = [i for i in range(1, current_P + 1) if H % i == 0]
            new_P = max(valid_P) if valid_P else 1
            for layer in self.drsa_layers:
                layer.P = new_P

        x_2d = padded_x.view(B_prime, H, W, d_model)

        # --- 3. DRSA Temporal Block Stage ---
        ### MODIFIED ###
        # Pass data through the stack of DRSA layers
        temporal_features_2d = x_2d
        for layer in self.drsa_layers:
            temporal_features_2d = layer(temporal_features_2d)

        # Flatten the 2D output and get a final representation vector
        asset_representation = temporal_features_2d.mean(dim=(1, 2))
        asset_representation = self.output_proj(asset_representation)

        # --- 4. CIDM Stage (Cross-Asset Correlation Modeling) ---
        asset_representation = asset_representation.view(batch_size, num_assets, -1)

        final_scores = self.cidm(asset_representation)

        return final_scores


# ==========================================================================================
# Example Usage
# ==========================================================================================
if __name__ == '__main__':
    # Define model arguments and hyperparameters
    model_args = {
        "window_size": 96,
        "num_features_original": 7,
    }

    model_params = {
        "HIDDEN_SIZE": 64,
        "NUM_HEADS": 8,
        ### ADDED ###
        # Add the number of layers parameter here
        "NUM_LAYERS": 2,  # Example: Stack 2 DRSA blocks
        "DROPOUT": 0.1,
        "DRSA_PARTITIONS": 8,
        "DRSA_TOP_K": 4
    }

    # Create a dummy input tensor
    batch_size = 4
    num_assets = 10
    num_features_augmented = model_args["window_size"] * model_args["num_features_original"]
    dummy_input = torch.randn(batch_size, num_assets, num_features_augmented)

    # Instantiate and run the model
    print("Instantiating DRSA model...")
    drsa_model = DRSA(model_args, model_params)
    print(f"Model instantiated with {model_params['NUM_LAYERS']} layer(s).")
    print(drsa_model)  # Print model structure

    print(f"\nInput shape: {dummy_input.shape}")
    output_scores = drsa_model(dummy_input)
    print(f"Output shape: {output_scores.shape}")

    # --- Verification ---
    print("\nVerifying output shape matches (batch_size, num_assets)...")
    assert output_scores.shape == (batch_size, num_assets)
    print("Shape verification successful!")