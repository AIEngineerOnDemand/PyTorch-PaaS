import os
import tempfile
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import sys
import argparse

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import SAGEMAKER_ROLE

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run training script.')
parser.add_argument('--execution_mode', type=str, choices=['fast_local_mode', 'aws_training'], required=True, help='Execution mode: fast_local_mode or aws_training')
args = parser.parse_args()

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# Create a temporary directory for dummy data
tmpdirname = tempfile.mkdtemp()

# Generate dummy data and save it to the temporary directory
dummy_data_path = os.path.join(tmpdirname, 'dummy_data.txt')
with open(dummy_data_path, 'w') as f:
    f.write('dummy data')

# Dynamically construct the source_dir path
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'computer_vision', 'image_classification'))

# Create an S3 bucket to save artifacts if in aws_training mode
if args.execution_mode == 'aws_training':
    s3_client = boto3.client('s3')
    bucket_name = 'image-detection-trained-model'
    s3_client.create_bucket(Bucket=bucket_name)
    s3_client.upload_file(dummy_data_path, bucket_name, 'input/dummy_data.txt')

# Create the PyTorch estimator
estimator = PyTorch(
    entry_point='train_models.py',
    source_dir=source_dir,
    role=SAGEMAKER_ROLE,
    framework_version='2.0.0',
    py_version='py310',
    script_mode=True,
    instance_count=1,
    instance_type='local' if args.execution_mode == 'fast_local_mode' else 'ml.m5.large',
    hyperparameters={
        'model_name': 'MobileNet',
        'epochs': 10,
        'execution_mode': args.execution_mode
    },
    output_path=f's3://{bucket_name}/output' if args.execution_mode == 'aws_training' else None
)

# Fit the estimator
if args.execution_mode == 'fast_local_mode':
    estimator.fit(f'file://{tmpdirname}')
else:
    estimator.fit(f's3://{bucket_name}/input')