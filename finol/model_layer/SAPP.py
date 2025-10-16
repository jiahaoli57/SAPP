import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils.transformers.models.stb.configuration_stb import STBConfig
from .utils.transformers.models.stb.modeling_stb import STBModel

# Assuming a utility function `load_config` exists to load project configurations.
from finol.utils import load_config


class STB(nn.Module):
    """
    This class implements the Sparse Transformer Block (STB) module.
    It processes time-series data for each asset to extract temporal features.
    """

    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        # Embedding layer for input features
        self.token_emb = nn.Linear(model_args["num_features_original"], model_params["HIDDEN_SIZE"])

        # Embedding for positional information
        self.pos_emb = nn.Embedding(model_args["window_size"], model_params["HIDDEN_SIZE"])

        # Determine attention type based on the model configuration in config
        # 'SAPP' uses sparse attention, 'FAPP' (Full Attention) uses full attention
        attention_type = "original_full" if self.config["MODEL_NAME"] == "FAPP" else "block_sparse"

        # Configuration for the Sparse Transformer Block (STB) model from Hugging Face
        # Hyperparameters are mapped from the SAPP paper's Table 3.
        configuration = STBConfig(
            hidden_size=model_params["HIDDEN_SIZE"],  # d: Hidden layer size
            num_hidden_layers=model_params["NUM_LAYERS"],  # E: Number of layers
            num_attention_heads=model_params["NUM_HEADS"],  # H: Number of attention heads
            intermediate_size=int(model_params["HIDDEN_SIZE"] * 2),  # q_f: Feed-forward layer size
            hidden_act="quick_gelu",
            hidden_dropout_prob=model_params["DROPOUT"],
            attention_probs_dropout_prob=model_params["DROPOUT"],
            max_position_embeddings=model_args["window_size"],  # n: Days of information
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            use_cache=False,
            attention_type=attention_type,
            use_bias=True,
            rescale_embeddings=False,
            block_size=model_params["BLOCK_SIZE"],  # B: Block size
            num_random_blocks=2,  # Corresponds to r: Number of random periods
            classifier_dropout=model_params["DROPOUT"],
        )
        # print(configuration)

        # Instantiate the STB model
        self.stb_model = STBModel(config=configuration)

    def forward(self, x):
        """
        Forward pass for the STB module.

        :param x: Input tensor of shape `(batch_size * num_assets, window_size, num_features_original)`.
        :return: Output tensor R of shape `(batch_size * num_assets, HIDDEN_SIZE)`.
        """
        device = x.device
        n = self.model_args["window_size"]  # n: sequence length

        # Apply token embedding to the input features
        x = self.token_emb(x)

        # Create and add positional embeddings
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        # Pass the embedded input through the STB model
        # The output is the hidden state of the last layer
        R = self.stb_model(inputs_embeds=x).last_hidden_state

        # Reshape and extract the representation R from the last time step
        R = rearrange(R, 'b n d -> b d n')
        R = R[:, :, -1].squeeze(-1)

        return R


class CIDM(nn.Module):
    """
    This class implements the Correlation Information Decision Module (CIDM).
    It models the inter-asset relationships using a full self-attention mechanism.
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


class SAPP(nn.Module):
    """
    This class implements the Sparse Attention Portfolio Policy (SAPP) model.

    The SAPP model is a Transformer-based framework for portfolio management that recognizes
    patterns in financial time series. It consists of two main components:

    1. Sparse Transformer Block (STB): This module uses a sparse attention mechanism to
       efficiently capture long-range temporal dependencies within the historical data of each asset.
       This achieves near-linear complexity with respect to the sequence length.

    2. Correlation Information Decision Module (CIDM): This module takes the asset representations
       from the STB and applies a full cross-asset self-attention mechanism to model
       inter-asset correlations, producing the final scores for portfolio allocation.

    The model supports ablation studies to analyze the contribution of each component:
    - SAPP (default): Full model with both STB and CIDM.
    - FAPP: Uses a full-attention Transformer instead of a sparse one.
    - SAPP-d: Removes the STB, feeding asset features directly to a linear layer.
    - SAPP-dd: Removes the CIDM, using a simple linear layer to generate scores from STB outputs.

    :param model_args: Dictionary of model arguments (e.g., window_size, num_features_original).
    :param model_params: Dictionary of model hyperparameters (e.g., HIDDEN_SIZE, DROPOUT_PROB, MODEL_NAME).
    """

    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        # Instantiate the main components
        self.stb = STB(model_args, model_params)
        self.cidm = CIDM(model_params)
        self.dropout = nn.Dropout(p=model_params["DROPOUT"])

        # Define layers for ablation studies based on MODEL_NAME
        if self.config["MODEL_NAME"] == 'SAPP-d':
            # Linear layer to replace STB for the 'SAPP-d' ablation study
            self.ab_study_linear_d = nn.Linear(model_args["num_features_original"], model_params["HIDDEN_SIZE"])

        if self.config["MODEL_NAME"] == 'SAPP-dd':
            # Linear layer to replace CIDM for the 'SAPP-dd' ablation study
            self.ab_study_linear_dd = nn.Linear(model_params["HIDDEN_SIZE"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SAPP model.

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

        # --- Sparse Transformer Block (STB) Stage ---
        if self.config["MODEL_NAME"] == 'SAPP-d':
            # Ablation study 'SAPP-d': Bypass STB.
            # Extract features from the last time step and pass through a linear layer.
            stock_rep = x[:, -1, :]  # stock_rep: asset representation
            stock_rep = self.ab_study_linear_d(stock_rep)
        else:
            # Default behavior: Use the STB module to get asset representations R.
            stock_rep = self.stb(x)

        # Apply dropout to the asset representations
        stock_rep = self.dropout(stock_rep)

        # Reshape back to (batch_size, num_assets, HIDDEN_SIZE) for CIDM
        stock_rep = stock_rep.view(batch_size, num_assets, self.model_params["HIDDEN_SIZE"])

        # --- Correlation Information Decision Module (CIDM) Stage ---
        if self.config["MODEL_NAME"] == 'SAPP-dd':
            # Ablation study 'SAPP-dd': Bypass CIDM.
            # Use a simple linear layer to get final_scores directly from STB outputs R.
            final_scores = self.ab_study_linear_dd(stock_rep).squeeze(-1)  # final_scores: final scores
        else:
            # Default behavior: Use the CIDM module to get final_scores.
            final_scores = self.cidm(stock_rep)

        return final_scores

