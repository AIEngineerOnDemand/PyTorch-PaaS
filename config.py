# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the IAM role from environment variable
SAGEMAKER_ROLE = os.getenv('SAGEMAKER_ROLE')

if not SAGEMAKER_ROLE:
	raise ValueError("SAGEMAKER_ROLE environment variable not set")