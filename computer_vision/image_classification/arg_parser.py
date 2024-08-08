import argparse

def get_common_parser():
    parser = argparse.ArgumentParser(description='Common argument parser')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--execution_mode', type=str, choices=['fast_local_mode', 'aws_training'], required=True, help='Execution mode: fast_local_mode or aws_training')
    parser.add_argument('--subsample', action='store_true', help='Whether to subsample the dataset')
    parser.add_argument('--subsample_rate', type=float, default=0.1, help='Rate at which to subsample the dataset')
    parser.add_argument('--bucket_name', type=str, default='image-detection-trained-model', help='S3 bucket name for saving the model')
    return parser