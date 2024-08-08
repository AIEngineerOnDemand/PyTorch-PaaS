import os
import tempfile
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import sys
from arg_parser import get_common_parser
import logging

# Parse arguments
parser = get_common_parser()
args = parser.parse_args()

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


if args.execution_mode == 'aws_training':

    # Initialize S3 client
    s3 = boto3.client('s3')

    def create_bucket_if_not_exists(bucket_name):
        try:
            # Check if the bucket exists
            s3.head_bucket(Bucket=bucket_name)
            logging.info(f"Bucket {bucket_name} already exists.")
        except s3.exceptions.ClientError as e:
            # If a ClientError is thrown, then the bucket does not exist.
            if e.response['Error']['Code'] == '404':
                # Create the bucket
                s3.create_bucket(Bucket=bucket_name)
                logging.info(f"Bucket {bucket_name} created.")
            else:
                # If the error is not a 404, re-raise the exception
                raise
    bucket_name = args.bucket_name
    create_bucket_if_not_exists(bucket_name)


# Create an S3 bucket to save artifacts if in aws_training mode
if args.execution_mode == 'aws_training':
    s3_client = boto3.client('s3')
    bucket_name = args.bucket_name
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
    instance_type='local' if args.execution_mode == 'fast_local_mode' else  'ml.m5.xlarge',
    use_spot_instances=True,  # Enable Spot Instances
    max_wait=3600,  # Maximum wait time for Spot Instances (in seconds)
    max_run=3600,  # Maximum runtime for the training job
    hyperparameters={
        'model_name': 'MobileNet',
        'epochs': 10,
        'execution_mode': args.execution_mode,
        'bucket_name': bucket_name if args.execution_mode == 'aws_training' else ''
    },
    output_path=f's3://{bucket_name}/output' if args.execution_mode == 'aws_training' else None
)

# Fit the estimator
if args.execution_mode == 'fast_local_mode':
    estimator.fit(f'file://{tmpdirname}')
else:
    estimator.fit(f's3://{bucket_name}/input')
    
# # Delete the SageMaker model if in aws_training mode
# if args.execution_mode == 'aws_training':
#     sagemaker_client = boto3.client('sagemaker')
#     sagemaker_client.delete_model(ModelName=estimator.latest_training_job.name)    