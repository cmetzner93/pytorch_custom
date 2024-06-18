"""
This file contains source code for various model architectures.
    @author:          Christoph S. Metzner
    @date created:    06/17/2024
    @last modified:   06/17/2024
"""

# Load libraries
# built-in
from typing import List, Union

# installed
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    This class creates a convolutional neural network following the
    implementation of Kim et al. (2014)

    Parameters
    ----------
    num_classes : int
        Number of classes the model is predicting on.
    hidden_size : Union[int, List[int]]
        Number of filters for the parallel convolution layers; can either be an
        integer when all layers are the same or explicitly provided as a list.
    window_sizes : List[int]
        List indicating the window size for each convolutional layer.
    dropout_prob : float
        Dropout probability for model regularization to prevent overfitting.
    logits_mechanism : str
        Argument indicating which mechanism should be used to generate the
        logits, can either be max-pooling layer or an attention mechanism.
    device : str
        Indicates accelerator
    word_embedding_dim : int
        Dimension of word embeddings 
    word_embedding_matrix : np.array, default = None
        Pretrained word embedding matrix
    vocab_size : int, default = None
        Size of vocabulary to initialize embedding layer

    """
 
    def __init__(
        self,
        num_classes: int,
        hidden_size: Union[int, List[int]],
        window_sizes: List[int],
        dropout_prob: float,
        logits_mechanism: str,
        device: str
        word_embedding_dim: int,
        word_embedding_matrix: np.array = None,
        vocab_size: int = None
    ) -> None:

        self.num_classes = num_classes
        self.window_sizes = window_sizes
        if is_instance(hidden_size, int):
            self.hidden_sizes = [hidden_size] * len(window_sizes)
        self.dropout_prob = dropout_prob
        self.logits_mechanism = logits_mechanism
        self.device = device
        self.word_embedding_dim = word_embedding_dim

        assert len(self.hidden_sizes) == len(self.window_sizes) 
 
        if word_embedding_matrix is None:
            self.embedding_layer = nn.Embedding(
                num_embedding=vocab_size,
                embedding_dim=word_embedding_dim,
                padding_idx=0
            )
        else:
            word_embedding_matrix -= word_embedding_matrix.mean(axis=0)
            word_embedding_matrix /= word_embedding_matrix.std()
            word_embedding_matrix[0] = 0.0  # padding token
            self.embedding_layer = nn.Embedding.from_pretrained(
                torch.tensor(word_embedding_matrix, dtype=torch.float),
                freeze=False,
                padding_idx=0
            )

        # Initialize dropout layer applied to convolution output
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
       
        # Initialize three parallel convolution layers
        self.conv_layers = nn.ModuleList()
        for window_size, hidden_size in zip(self.window_sizes, self.hidden_sizes):
            conv_layer = nn.Conv1d(
                in_channels=self.word_embedding_dim,
                out_channels=hidden_size,
                padding='same',
                bias=True,
                kernel_size=window_size
            )
            nn.init.xavier_uniform_(conv_layer.weight)
            nn.init.xavier_uniform_(conv_layer.bias)
            self.conv_layers.append(conv_layer)

        self.output_layer = nn.Linear(
            in_features=torch.sum(self.hidden_sizes),
            out_features=self.num_classes
        )
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN object
        
        Parameter
        ---------
        input_ids : torch.Tensor(dtype=torch.long)
            Integer-indexed text sequences

        Return
        ------
        torch.Tensor
            Logits with shape [batch_size, num_classes]
           
        """
        
        # Truncates batch to length of longest sequence 
        mask_tokens = (input_ids != 0)
        tokens_per_sequence = mask_tokens.sum(-1)
        max_tokens = tokens_per_sequence.max()
        max_tokens = max(max_tokens, max(self.window_sizes))
       
        mask_tokens = torch.unsqueeze(mask_tokens[:, :max_tokens], -1)
        input_ids_reduced = input_ids[:, :max_tokens]

        token_embeddings = self.embedding_layer(input_ids_reduced)
        token_embeddings = torch.mul(
            token_embeddings, 
            mask.tokens.type(token_embeddings.dtype)
        )
        token_embeddings = token_embeds.permute(0, 2, 1)

        conv_outs = []
        for layer in self.conv_layers:
            conv_out = F.relu(layer(token_embeddings))
            conv_outs.append(conv_out)
        H = torch.cat(conv_outs, 1)
        H = self.dropout_layer(H)

        if self.logits_mechanism == 'baseline':
            logits = self.output_layer(H.permute(0, 2, 1)).permute(0, 2, 1)
            logits = F.adaptive_max_pool1d(logits, 1)
            logits = torch.flatten(logits, start_dim=1)

        return logits 
         
