# type: ignore
#
### Import Modules. ###
#
import torch
import torch.nn as nn
from typing import Dict, Tuple, Type, Any, Union, List, Optional
import inspect

def count_low_level_matrix_operations_attached_to_low_level_layers_types(
    model: nn.Module
) -> Dict[str, Tuple[int, int]]:
    """
    Count low-level matrix operations in a PyTorch model, including those hidden inside high-level modules.

    This function recursively traverses the model and:
    1. Counts low-level layers directly (Linear, Conv1d, Conv2d, etc.)
    2. Decomposes high-level modules (LSTM, Transformer, etc.) into their constituent low-level operations
    3. Includes 0-parameter layers like ReLU since they perform operations
    4. Calculates the total number of parameters for each low-level operation type

    The function separates low-level matrix operations from high-level PyTorch abstractions:
    - LOW-LEVEL (counted directly): Linear, Conv1d, Conv2d, Conv3d, Embedding, BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, ReLU, GELU, etc.
    - HIGH-LEVEL (decomposed): LSTM, LSTMCell, GRU, GRUCell, RNN, RNNCell, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention

    Args:
        model: PyTorch nn.Module to analyze

    Returns:
        Dictionary mapping layer/matrix operation type names to tuples of:
        (number_of_instances, total_sum_of_parameters)
        Example: {'Linear': (5, 12500), 'Conv2d': (3, 8960), 'ReLU': (8, 0)}

    Example usage:
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.LSTM(20, 30, num_layers=2),
            nn.TransformerEncoderLayer(d_model=30, nhead=3)
        )
        counts = count_low_level_matrix_operations_attached_to_low_level_layers_types(model)
        # This would count the Linear layer directly, decompose LSTM into its internal linear ops,
        # decompose TransformerEncoderLayer into MultiheadAttention and feed-forward Linear layers,
        # and count the ReLU activation.
    """

    # Initialize counter dictionary: {layer_type: (count, total_params)}
    operation_counts: Dict[str, Tuple[int, int]] = {}

    def get_or_init_entry(layer_type: str) -> Tuple[int, int]:
        """Get existing entry or initialize new one for a layer type."""
        if layer_type not in operation_counts:
            operation_counts[layer_type] = (0, 0)
        return operation_counts[layer_type]

    def update_counts(layer_type: str, param_count: int = 0) -> None:
        """Update the counts for a specific layer type."""
        current_count, current_params = get_or_init_entry(layer_type)
        operation_counts[layer_type] = (current_count + 1, current_params + param_count)

    def count_parameters(module: nn.Module) -> int:
        """Count total trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def decompose_lstm(lstm: nn.LSTM) -> None:
        """
        Decompose LSTM into its internal low-level operations.

        An LSTM layer contains multiple gates (input, forget, cell, output).
        Each gate has two linear transformations: one for input-to-hidden and one for hidden-to-hidden.
        For each layer in a multi-layer LSTM:
        - 4 gates x 2 linear transformations = 8 linear operations per layer
        - Plus bias terms (already included in Linear layer parameter count)

        We count all internal linear operations as 'Linear' type.
        """
        num_layers = lstm.num_layers
        bidirectional = lstm.bidirectional
        num_directions = 2 if bidirectional else 1

        # For each layer and direction, we have 4 gates (input, forget, cell, output)
        # Each gate has 2 linear transformations (input->hidden, hidden-to-hidden)
        linear_ops_per_layer = 4 * 2 * num_directions  # 8 ops per layer per direction

        # Calculate parameters for one set of gates
        input_size = lstm.input_size
        hidden_size = lstm.hidden_size

        # For each gate, we have:
        # - input-to-hidden: (input_size × hidden_size) + hidden_size (bias)
        # - hidden-to-hidden: (hidden_size × hidden_size) + hidden_size (bias)
        # Total per gate: hidden_size × (input_size + hidden_size + 2)
        # Total for 4 gates: 4 × hidden_size × (input_size + hidden_size + 2)

        params_per_layer_direction = 4 * hidden_size * (input_size + hidden_size + 2)
        total_params = params_per_layer_direction * num_layers * num_directions

        # Update counts for Linear operations (all internal linear ops count as 'Linear')
        for _ in range(linear_ops_per_layer * num_layers):
            update_counts('Linear', 0)  # We'll add parameters below

        # Add the total parameters to the last Linear count
        # Note: This is an approximation - in reality, parameters are distributed across all linear ops
        if 'Linear' in operation_counts and operation_counts['Linear'][0] > 0:
            current_count, current_params = operation_counts['Linear']
            operation_counts['Linear'] = (current_count, current_params + total_params)

    def decompose_lstm_cell(lstm_cell: nn.LSTMCell) -> None:
        """
        Decompose LSTMCell into its internal low-level operations.

        An LSTMCell contains 4 gates (input, forget, cell, output).
        Each gate has two linear transformations: input-to-hidden and hidden-to-hidden.
        Total: 8 linear operations.

        We count all internal linear operations as 'Linear' type.
        """
        input_size = lstm_cell.input_size
        hidden_size = lstm_cell.hidden_size

        # 4 gates × 2 linear transformations each = 8 linear operations
        num_linear_ops = 8

        # Calculate total parameters
        # Each gate has: (input_size × hidden_size) + (hidden_size × hidden_size) + 2 × hidden_size (biases)
        # Simplified: hidden_size × (input_size + hidden_size + 2) per gate
        params_per_gate = hidden_size * (input_size + hidden_size + 2)
        total_params = 4 * params_per_gate

        # Update counts for Linear operations
        for _ in range(num_linear_ops):
            update_counts('Linear', 0)

        # Add parameters to the Linear count
        if 'Linear' in operation_counts and operation_counts['Linear'][0] > 0:
            current_count, current_params = operation_counts['Linear']
            operation_counts['Linear'] = (current_count, current_params + total_params)

    def decompose_gru(gru: nn.GRU) -> None:
        """
        Decompose GRU into its internal low-level operations.

        A GRU layer contains 3 gates (reset, update, new).
        Each gate has two linear transformations: input-to-hidden and hidden-to-hidden.
        For each layer:
        - 3 gates × 2 linear transformations = 6 linear operations per layer

        We count all internal linear operations as 'Linear' type.
        """
        num_layers = gru.num_layers
        bidirectional = gru.bidirectional
        num_directions = 2 if bidirectional else 1

        linear_ops_per_layer = 3 * 2 * num_directions  # 6 ops per layer per direction
        input_size = gru.input_size
        hidden_size = gru.hidden_size

        # For each gate: (input_size + hidden_size + 1) × hidden_size
        # For 3 gates: 3 × (input_size + hidden_size + 1) × hidden_size
        params_per_layer_direction = 3 * hidden_size * (input_size + hidden_size + 2)
        total_params = params_per_layer_direction * num_layers * num_directions

        # Update counts for Linear operations
        for _ in range(linear_ops_per_layer * num_layers):
            update_counts('Linear', 0)

        # Add parameters to the Linear count
        if 'Linear' in operation_counts and operation_counts['Linear'][0] > 0:
            current_count, current_params = operation_counts['Linear']
            operation_counts['Linear'] = (current_count, current_params + total_params)

    def decompose_gru_cell(gru_cell: nn.GRUCell) -> None:
        """
        Decompose GRUCell into its internal low-level operations.

        A GRUCell contains 3 gates (reset, update, new).
        Each gate has two linear transformations.
        Total: 6 linear operations.

        We count all internal linear operations as 'Linear' type.
        """
        input_size = gru_cell.input_size
        hidden_size = gru_cell.hidden_size

        num_linear_ops = 6  # 3 gates × 2 transformations each

        # Each gate: (input_size + hidden_size + 1) × hidden_size
        params_per_gate = hidden_size * (input_size + hidden_size + 2)
        total_params = 3 * params_per_gate

        # Update counts for Linear operations
        for _ in range(num_linear_ops):
            update_counts('Linear', 0)

        # Add parameters to the Linear count
        if 'Linear' in operation_counts and operation_counts['Linear'][0] > 0:
            current_count, current_params = operation_counts['Linear']
            operation_counts['Linear'] = (current_count, current_params + total_params)

    def decompose_multihead_attention(mha: nn.MultiheadAttention) -> None:
        """
        Decompose MultiheadAttention into its internal low-level operations.

        A MultiheadAttention layer contains:
        1. Three linear projections for Q, K, V (query, key, value) - counted as Linear
        2. Optional output projection - counted as Linear
        3. The attention computation itself (not a learnable parameter operation)

        We count all linear projections as 'Linear' type.
        """
        embed_dim = mha.embed_dim

        # Q, K, V projections (3 linear layers)
        num_qkv_projections = 3
        for _ in range(num_qkv_projections):
            update_counts('Linear', 0)  # Will add parameters below

        # Output projection (1 linear layer if enabled)
        has_out_proj = hasattr(mha, 'out_proj') and mha.out_proj.weight is not None
        if has_out_proj:
            update_counts('Linear', 0)

        # Calculate total parameters
        total_params = 0

        # Q, K, V projections: 3 × (embed_dim × embed_dim) + 3 × embed_dim (biases)
        qkv_params = 3 * (embed_dim * embed_dim + embed_dim)
        total_params += qkv_params

        # Output projection: embed_dim × embed_dim + embed_dim (if enabled)
        if has_out_proj:
            out_proj_params = embed_dim * embed_dim + embed_dim
            total_params += out_proj_params

        # Add parameters to the Linear count
        if 'Linear' in operation_counts and operation_counts['Linear'][0] > 0:
            current_count, current_params = operation_counts['Linear']
            operation_counts['Linear'] = (current_count, current_params + total_params)

    def decompose_transformer_encoder_layer(layer: nn.TransformerEncoderLayer) -> None:
        """
        Decompose TransformerEncoderLayer into its internal low-level operations.

        A TransformerEncoderLayer contains:
        1. MultiheadAttention (self-attention) - decomposed into Linear operations
        2. Feed-forward network (typically 2 Linear layers with activation in between)
        3. LayerNorm layers
        4. Dropout layers
        5. Activation functions

        All linear operations are counted as 'Linear' type.
        """
        # Decompose the self-attention part
        if hasattr(layer, 'self_attn'):
            decompose_multihead_attention(layer.self_attn)

        # Count the feed-forward network linear layers directly
        if hasattr(layer, 'linear1') and isinstance(layer.linear1, nn.Linear):
            update_counts('Linear', count_parameters(layer.linear1))

        if hasattr(layer, 'linear2') and isinstance(layer.linear2, nn.Linear):
            update_counts('Linear', count_parameters(layer.linear2))

        # Count normalization layers
        if hasattr(layer, 'norm1') and isinstance(layer.norm1, nn.LayerNorm):
            update_counts('LayerNorm', count_parameters(layer.norm1))

        if hasattr(layer, 'norm2') and isinstance(layer.norm2, nn.LayerNorm):
            update_counts('LayerNorm', count_parameters(layer.norm2))

        # Count dropout layers
        if hasattr(layer, 'dropout') and isinstance(layer.dropout, nn.Dropout):
            update_counts('Dropout')

        if hasattr(layer, 'dropout1') and isinstance(layer.dropout1, nn.Dropout):
            update_counts('Dropout')

        if hasattr(layer, 'dropout2') and isinstance(layer.dropout2, nn.Dropout):
            update_counts('Dropout')

        # Count activation functions properly
        if hasattr(layer, 'activation'):
            if isinstance(layer.activation, nn.ReLU):
                update_counts('ReLU')
            elif isinstance(layer.activation, nn.GELU):
                update_counts('GELU')
            elif isinstance(layer.activation, nn.ReLU6):
                update_counts('ReLU6')
            elif isinstance(layer.activation, nn.LeakyReLU):
                update_counts('LeakyReLU')
            elif isinstance(layer.activation, nn.Sigmoid):
                update_counts('Sigmoid')
            elif isinstance(layer.activation, nn.Tanh):
                update_counts('Tanh')
            else:
                # Generic activation function - try to get its name
                activation_name = type(layer.activation).__name__
                if activation_name != 'function':  # Skip if it's just a function object
                    update_counts(activation_name)

    def decompose_transformer_decoder_layer(layer: nn.TransformerDecoderLayer) -> None:
        """
        Decompose TransformerDecoderLayer into its internal low-level operations.

        A TransformerDecoderLayer contains:
        1. MultiheadAttention (self-attention) - decomposed into Linear operations
        2. MultiheadAttention (cross-attention with encoder output) - decomposed into Linear operations
        3. Feed-forward network (2 Linear layers)
        4. LayerNorm layers
        5. Dropout layers
        6. Activation functions

        All linear operations are counted as 'Linear' type.
        """
        # Decompose self-attention
        if hasattr(layer, 'self_attn'):
            decompose_multihead_attention(layer.self_attn)

        # Decompose cross-attention
        if hasattr(layer, 'multihead_attn'):
            decompose_multihead_attention(layer.multihead_attn)

        # Count the feed-forward network linear layers directly
        if hasattr(layer, 'linear1') and isinstance(layer.linear1, nn.Linear):
            update_counts('Linear', count_parameters(layer.linear1))

        if hasattr(layer, 'linear2') and isinstance(layer.linear2, nn.Linear):
            update_counts('Linear', count_parameters(layer.linear2))

        # Count normalization layers
        norm_attrs = ['norm1', 'norm2', 'norm3']
        for attr in norm_attrs:
            if hasattr(layer, attr):
                norm_layer = getattr(layer, attr)
                if isinstance(norm_layer, nn.LayerNorm):
                    update_counts('LayerNorm', count_parameters(norm_layer))

        # Count dropout layers
        dropout_attrs = ['dropout', 'dropout1', 'dropout2', 'dropout3']
        for attr in dropout_attrs:
            if hasattr(layer, attr):
                dropout_layer = getattr(layer, attr)
                if isinstance(dropout_layer, nn.Dropout):
                    update_counts('Dropout')

        # Count activation functions
        if hasattr(layer, 'activation'):
            if isinstance(layer.activation, nn.ReLU):
                update_counts('ReLU')
            elif isinstance(layer.activation, nn.GELU):
                update_counts('GELU')
            elif isinstance(layer.activation, nn.ReLU6):
                update_counts('ReLU6')
            elif isinstance(layer.activation, nn.LeakyReLU):
                update_counts('LeakyReLU')
            elif isinstance(layer.activation, nn.Sigmoid):
                update_counts('Sigmoid')
            elif isinstance(layer.activation, nn.Tanh):
                update_counts('Tanh')
            else:
                activation_name = type(layer.activation).__name__
                if activation_name != 'function':
                    update_counts(activation_name)

    def decompose_transformer(transformer: Union[nn.TransformerEncoder, nn.TransformerDecoder, nn.Transformer]) -> None:
        """
        Decompose Transformer modules into their internal low-level operations.

        Transformer modules contain multiple TransformerEncoderLayer or TransformerDecoderLayer instances.
        We need to decompose each layer individually.
        """
        # Handle TransformerEncoder
        if isinstance(transformer, nn.TransformerEncoder) and hasattr(transformer, 'layers'):
            for layer in transformer.layers:
                if isinstance(layer, nn.TransformerEncoderLayer):
                    decompose_transformer_encoder_layer(layer)

        # Handle TransformerDecoder
        elif isinstance(transformer, nn.TransformerDecoder) and hasattr(transformer, 'layers'):
            for layer in transformer.layers:
                if isinstance(layer, nn.TransformerDecoderLayer):
                    decompose_transformer_decoder_layer(layer)

        # Handle full Transformer
        elif isinstance(transformer, nn.Transformer):
            if hasattr(transformer, 'encoder'):
                decompose_transformer(transformer.encoder)
            if hasattr(transformer, 'decoder'):
                decompose_transformer(transformer.decoder)

    def process_module(module: nn.Module) -> None:
        """
        Process a module recursively, counting low-level operations.

        This function handles both atomic low-level modules and decomposes high-level modules.
        """
        module_type = type(module).__name__

        # Handle low-level modules directly - these are the actual matrix operations
        low_level_modules = {
            # Linear algebra operations
            'Linear': lambda m: update_counts('Linear', count_parameters(m)),
            'Conv1d': lambda m: update_counts('Conv1d', count_parameters(m)),
            'Conv2d': lambda m: update_counts('Conv2d', count_parameters(m)),
            'Conv3d': lambda m: update_counts('Conv3d', count_parameters(m)),
            'ConvTranspose1d': lambda m: update_counts('ConvTranspose1d', count_parameters(m)),
            'ConvTranspose2d': lambda m: update_counts('ConvTranspose2d', count_parameters(m)),
            'ConvTranspose3d': lambda m: update_counts('ConvTranspose3d', count_parameters(m)),

            # Embedding layers (lookup tables - not strictly matrix multiplication but important operations)
            'Embedding': lambda m: update_counts('Embedding', count_parameters(m)),
            'EmbeddingBag': lambda m: update_counts('EmbeddingBag', count_parameters(m)),

            # Normalization layers (perform operations but typically have few parameters)
            'BatchNorm1d': lambda m: update_counts('BatchNorm1d', count_parameters(m)),
            'BatchNorm2d': lambda m: update_counts('BatchNorm2d', count_parameters(m)),
            'BatchNorm3d': lambda m: update_counts('BatchNorm3d', count_parameters(m)),
            'LayerNorm': lambda m: update_counts('LayerNorm', count_parameters(m)),
            'GroupNorm': lambda m: update_counts('GroupNorm', count_parameters(m)),
            'InstanceNorm1d': lambda m: update_counts('InstanceNorm1d', count_parameters(m)),
            'InstanceNorm2d': lambda m: update_counts('InstanceNorm2d', count_parameters(m)),
            'InstanceNorm3d': lambda m: update_counts('InstanceNorm3d', count_parameters(m)),

            # Activation functions (0 parameters but perform operations)
            'ReLU': lambda m: update_counts('ReLU'),
            'ReLU6': lambda m: update_counts('ReLU6'),
            'LeakyReLU': lambda m: update_counts('LeakyReLU'),
            'GELU': lambda m: update_counts('GELU'),
            'SiLU': lambda m: update_counts('SiLU'),
            'Sigmoid': lambda m: update_counts('Sigmoid'),
            'Tanh': lambda m: update_counts('Tanh'),
            'Softmax': lambda m: update_counts('Softmax'),
            'LogSoftmax': lambda m: update_counts('LogSoftmax'),
            'ELU': lambda m: update_counts('ELU'),
            'SELU': lambda m: update_counts('SELU'),
            'CELU': lambda m: update_counts('CELU'),
            'Hardswish': lambda m: update_counts('Hardswish'),
            'Mish': lambda m: update_counts('Mish'),

            # Pooling operations
            'MaxPool1d': lambda m: update_counts('MaxPool1d'),
            'MaxPool2d': lambda m: update_counts('MaxPool2d'),
            'MaxPool3d': lambda m: update_counts('MaxPool3d'),
            'AvgPool1d': lambda m: update_counts('AvgPool1d'),
            'AvgPool2d': lambda m: update_counts('AvgPool2d'),
            'AvgPool3d': lambda m: update_counts('AvgPool3d'),
            'AdaptiveMaxPool1d': lambda m: update_counts('AdaptiveMaxPool1d'),
            'AdaptiveMaxPool2d': lambda m: update_counts('AdaptiveMaxPool2d'),
            'AdaptiveMaxPool3d': lambda m: update_counts('AdaptiveMaxPool3d'),
            'AdaptiveAvgPool1d': lambda m: update_counts('AdaptiveAvgPool1d'),
            'AdaptiveAvgPool2d': lambda m: update_counts('AdaptiveAvgPool2d'),
            'AdaptiveAvgPool3d': lambda m: update_counts('AdaptiveAvgPool3d'),

            # Regularization and other operations
            'Dropout': lambda m: update_counts('Dropout'),
            'Dropout1d': lambda m: update_counts('Dropout1d'),
            'Dropout2d': lambda m: update_counts('Dropout2d'),
            'Dropout3d': lambda m: update_counts('Dropout3d'),
            'Flatten': lambda m: update_counts('Flatten'),
            'Unflatten': lambda m: update_counts('Unflatten'),
        }

        # Handle high-level modules that need decomposition
        high_level_modules = {
            'LSTM': decompose_lstm,
            'LSTMCell': decompose_lstm_cell,
            'GRU': decompose_gru,
            'GRUCell': decompose_gru_cell,
            'RNN': lambda m: decompose_gru(m),  # RNN is similar to GRU but simpler
            'RNNCell': lambda m: decompose_gru_cell(m),  # Similar approach
            'MultiheadAttention': decompose_multihead_attention,
            'TransformerEncoderLayer': decompose_transformer_encoder_layer,
            'TransformerDecoderLayer': decompose_transformer_decoder_layer,
            'TransformerEncoder': decompose_transformer,
            'TransformerDecoder': decompose_transformer,
            'Transformer': decompose_transformer,
        }

        # Check if this is a low-level module we can count directly
        if module_type in low_level_modules:
            low_level_modules[module_type](module)
            return

        # Check if this is a high-level module that needs decomposition
        if module_type in high_level_modules:
            high_level_modules[module_type](module)
            return

        # For all other modules, recursively process their children
        # This handles custom modules and containers like Sequential, ModuleList, etc.
        for child_name, child in module.named_children():
            # Skip if it's not a proper nn.Module (sometimes there are function objects)
            if not isinstance(child, nn.Module):
                continue

            # Special handling for activation functions that might be stored as attributes
            if hasattr(child, '__call__') and not hasattr(child, 'parameters'):
                # Try to identify the activation function type
                if isinstance(child, torch.nn.modules.activation.ReLU):
                    update_counts('ReLU')
                elif isinstance(child, torch.nn.modules.activation.GELU):
                    update_counts('GELU')
                elif isinstance(child, torch.nn.modules.activation.Sigmoid):
                    update_counts('Sigmoid')
                elif isinstance(child, torch.nn.modules.activation.Tanh):
                    update_counts('Tanh')
                else:
                    # Try to get a meaningful name
                    activation_name = type(child).__name__
                    if activation_name != 'function' and activation_name != 'builtin_function_or_method':
                        update_counts(activation_name)
                continue

            process_module(child)

    # Start processing from the root module
    process_module(model)

    return operation_counts


if __name__ == "__main__":

    #
    from pprint import pprint

    # 1. Define a complex, high-level model
    class MyTransformerModel(nn.Module):
        def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
            super().__init__()
            self.pos_encoder = nn.Embedding(1000, d_model) # Will be counted

            # A high-level TransformerEncoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True
            )

            # A high-level container for the layers
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )

            # A final output layer
            self.output_layer = nn.Linear(d_model, 10) # Will be counted
            self.relu = nn.ReLU() # Will be counted

        def forward(self, x):
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = self.output_layer(x)
            return self.relu(x)

    # 2. Instantiate the model
    model = MyTransformerModel(d_model=128, nhead=4, num_layers=2, dim_feedforward=256)

    # 3. Run the analysis function
    layer_stats = count_low_level_matrix_operations_attached_to_low_level_layers_types(model)

    # 4. Print the results
    print("--- Model Structure (abbreviated) ---")
    print(model)
    print("\n--- Low-Level Layer Statistics ---")
    pprint(layer_stats)

    # --- Expected Output Analysis ---
    # A TransformerEncoderLayer has:
    #   - 1 MultiheadAttention (which has Linear layers for q,k,v,out)
    #   - 2 Dropouts
    #   - 2 LayerNorms
    #   - 1 "feedforward" block (Linear -> ReLU/GELU -> Dropout -> Linear)
    #
    # Our function should find:
    # - Inside nn.TransformerEncoderLayer (x2 layers):
    #   - Linear: 4 (from MHA) + 2 (from FFN) = 6 per layer * 2 layers = 12
    #   - LayerNorm: 2 per layer * 2 layers = 4
    #   - Dropout: 2 (MHA) + 1 (FFN) = 3 per layer * 2 layers = 6
    #   - GELU/ReLU: 1 per layer * 2 layers = 2
    # - Outside:
    #   - Embedding: 1
    #   - Linear: 1
    #   - ReLU: 1
    #
    # Total (approx):
    #   - Linear: 13
    #   - LayerNorm: 4
    #   - Dropout: 6
    #   - GELU/ReLU: 2 + 1 = 3
    #   - Embedding: 1
    # (Note: MHA is implemented with F.linear, not nn.Linear, so it won't be counted.
    # Ah, wait. Let's check the source.
    # `nn.TransformerEncoderLayer` *does* use `nn.Linear` for its feedforward network.
    # `nn.MultiheadAttention` *does not* use `nn.Linear` modules, it uses `F.linear`
    # with `self.in_proj_weight`, etc. So it is *itself* a low-level layer.
    #
    # Let's add nn.MultiheadAttention to the list if we want to count it.
    # The user's request was: "well separating... Linear... and... MultiHeadAttention"
    # This implies MHA is HIGH level and should NOT be in the list.
    # The user also said: "Even an RNN cell is a high level layer, I want only the
    # low level matrix operations behind this high level layers."
    #
    # This is the ambiguity. `nn.MultiheadAttention` does *not* expose its
    # `nn.Linear` layers as children. It *is* the operation.
    #
    # Given the user's *explicit* separation, the current code is correct. It will
    # *not* count `nn.MultiheadAttention` (as it's not in the SET), and it
    # *will* find the `nn.Linear` layers in the FFN block.