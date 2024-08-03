import os
import tempfile
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import sys

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import SAGEMAKER_ROLE
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

# Create an S3 bucket to save artifacts
s3_client = boto3.client('s3')
import os
import tempfile
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import sys

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import SAGEMAKER_ROLE
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

# Create an S3 bucket to save artifacts
s3_client = boto3.client('s3')
bucket_name = 'image-detection-trained-model'
s3_client.create_bucket(Bucket=bucket_name)

# Create the PyTorch estimator
estimator = PyTorch(
    entry_point='train_models.py',
    source_dir=source_dir,
    role=SAGEMAKER_ROLE,
    framework_version='2.0.0',
    py_version='py310',
    script_mode=True,
    instance_count=1,
    instance_type='local',
    hyperparameters={
        'model_name': 'MobileNet',
        'epochs': 10,
        'fast_local_mode': ''
    },
    output_path=f's3://{bucket_name}/output'
)

# Fit the estimator using local mode
estimator.fit(f'file://{tmpdirname}')
s3_client.create_bucket(Bucket=bucket_name)

# Create the PyTorch estimator
estimator = PyTorch(
    entry_point='train_models.py',
    source_dir=source_dir,
    role=SAGEMAKER_ROLE,
    framework_version='2.0.0',
    py_version='py310',
    script_mode=True,
    instance_count=1,
    instance_type='local',
    hyperparameters={
        'model_name': 'MobileNet',
        'epochs': 10,
        'fast_local_mode': ''
    },
    output_path=f's3://{bucket_name}/output'
)

# Fit the estimator using local mode
estimator.fit(f'file://{tmpdirname}')