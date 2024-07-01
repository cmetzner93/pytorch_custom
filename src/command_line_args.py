"""
This file contains source code handling the command line arguments controlling
the usage of the ModelSuite class.
    @author: Christoph S. Metzner
    @date created:  06/22/2024
    @last modified: 07/01/2024
"""

# Import libraries
# built-in
import argparse

# custom
from utils import eval_bool_command


def create_args_parser():
    parser = argparse.ArgumentParser(description='ModelSuite Parser')

    # Required command line arguments
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Name of set of experiments that you are doing'
    )
    parser.add_argument(
        '--model_description',
        type=str,
        required=True,
        help='Description that you are giving your model, e.g., special run'
    )

    parser.add_argument(
        '-m',
        '--model_type',
        type=str,
        required=True,
        choices=['CNN']
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=True,
        choices=['dummy']
    )

    # These command line args control model training/evaluation flow
    parser.add_argument(
        '-tm',
        '--train_model',
        type=eval_bool_command,
        required=True
    )
    parser.add_argument(
        '-em',
        '--eval_model',
        default=False,
        type=eval_bool_command,
        required=False
    )
    # Optional command line arguments
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=42
    )
    parser.add_argument(
        '--debugging',
        type=eval_bool_command,
        required=False,
        default='False'
    )

    parser.add_argument(
        '--inference_data',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='flag indicating split for model inference/testing'
    )

    # Optional training kwargs
    parser.add_argument(
        '-dml',
        '--doc_max_len',
        type=int,
        help='Maximum document length'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        help='Batch size - if DDP batch size is per GPU otherwise total'
    )
    parser.add_argument(
        '--patience',
        type=int,
        help='Maximum number of patience until early stopping is applied.'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help='Learning rate used for model training'
    )
    # Optional model kwargs
    parser.add_argument(
        '-wed',
        '--word_embedding_dim',
        type=int,
        help='Dimension of word embeddings - use in conventional models'
    )
    parser.add_argument(
        '-hs',
        '--hidden_size',
        type=int,
        help='Hidden dimension of text-encoder output'
    )
    parser.add_argument(
        '-dp',
        '--dropout_prob',
        type=float,
        help='Dropout probability'
    )

    parser.add_argument(
        '--logits_mechanism',
        type=str,
        help='Flag indicating how final context-vector / logits are computed'
    )

    parser.add_argument(
        '--path_trained_model',
        type=str,
        help='Path to trained model - make sure that its model_config file \
              is in the same directory'
    )
    parser.add_argument(
        '--from_checkpoint',
        type=str,
        help='Absolute path to model checkpoint - make sure that its \
        model_config file is in the smae directory.'
    )
    parser.add_argument(
        '--store_scores',
        type=eval_bool_command,
        default=False,
        help='Flag to indicate whether output (predictions/probabilities) \
              need to be stored.'
    )
    parser.add_argument(
        '--store_performance_scores',
        type=eval_bool_command,
        default=True,
        help='Flag to indicate whether performance scores are stored.'
    )
    return parser

