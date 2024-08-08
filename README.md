# PyTorch PaaS

This repository is dedicated to showcasing the main use cases of PyTorch, a popular open-source machine learning library, with a focus on training and deploying models using AWS SageMaker. PyTorch PaaS integrates MLOps practices to provide a seamless experience for developing, training, and deploying machine learning models.

## Use Cases

### [ Computer Vision](computer_vision/image_classification/README.md)


PyTorch's flexibility and efficient tensor computations make it suitable for tasks like image classification, object detection, and image generation.

### Natural Language Processing (NLP)

Companies use PyTorch for tasks like text classification, sentiment analysis, machine translation, and more.

*Note: This section is coming up.*

### Reinforcement Learning

PyTorch is used to develop models that can learn to make decisions from their experiences.

*Note: This section is coming up.*

### Generative Models

PyTorch is used in creating generative models like GANs (Generative Adversarial Networks) which can generate new content.

*Note: This section is coming up.*

### Research

Due to its dynamic computation graph and efficient memory usage, PyTorch is also a popular choice in the research community.

*Note: This section is coming up.*

## Development Approach

In our project, we prioritize efficient development practices to ensure rapid iteration and robust code quality, even on machines with limited resources. Here's how we achieve this:

### Fast Iterations and Local Testing

We understand the importance of being able to quickly test changes without the need for extensive setup or reliance on high-resource environments. To support this, we've implemented functionality that allows for the simulation or mocking of complex operations. This enables developers to run tests locally, ensuring that the code behaves as expected before moving to more resource-intensive testing environments. This approach significantly reduces the development cycle time and improves productivity.

### Separation of Concerns

Our codebase is structured around the principle of separation of concerns, ensuring that different parts of the application are modular and independent. This design philosophy allows for easier maintenance, testing, and scalability. By separating the application into distinct sections based on functionality, developers can work on individual components without the risk of causing unintended side effects in unrelated parts of the application.

This modular approach also facilitates the implementation of mock models and data for local testing, as developers can easily swap out or modify components without affecting the core functionality of the application.

### Why This Matters

By incorporating these practices into our development process, we aim to create a more efficient and error-resistant workflow. This allows our team to focus on innovation and the development of high-quality features, even when working on devices with limited computational power. It's a testament to our commitment to not only achieving our project goals but also to adhering to best practices in software development.

## Running the Code on SageMaker

This project is designed to run on AWS SageMaker using local mode. We use the SageMaker SDK and CLI to set up and execute the training jobs. Below is a brief overview of the key scripts involved:

### [`run_training.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2Frun_training.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\giopl\OneDrive\Desktop\python_projects\PyTorch-Use-Cases\computer_vision\image_classification\run_training.py")

This script initializes the SageMaker session, creates a temporary directory for dummy data, dynamically constructs the [`source_dir`](command:_github.copilot.openSymbolFromReferences?%5B%22source_dir%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5Ccomputer_vision%5C%5Cimage_classification%5C%5CREADME.md%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2FREADME.md%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2FREADME.md%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A38%2C%22character%22%3A121%7D%7D%2C%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5Ccomputer_vision%5C%5Cimage_classification%5C%5Crun_training.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2Frun_training.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2Frun_training.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A0%7D%7D%2C%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5CREADME.md%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2FREADME.md%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2FREADME.md%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A48%2C%22character%22%3A121%7D%7D%5D%5D "Go to definition") path, and creates a PyTorch estimator. The estimator is then fitted using local mode.

### [`train_models.py`](command:_github.copilot.openSymbolFromReferences?%5B%22train_models.py%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5Ccomputer_vision%5C%5Cimage_classification%5C%5CREADME.md%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2FREADME.md%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2FREADME.md%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A40%2C%22character%22%3A5%7D%7D%2C%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5Ccomputer_vision%5C%5Cimage_classification%5C%5Crun_training.py%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2Frun_training.py%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fcomputer_vision%2Fimage_classification%2Frun_training.py%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A27%2C%22character%22%3A17%7D%7D%2C%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22c%3A%5C%5CUsers%5C%5Cgiopl%5C%5COneDrive%5C%5CDesktop%5C%5Cpython_projects%5C%5CPyTorch-Use-Cases%5C%5CREADME.md%22%2C%22_sep%22%3A1%2C%22external%22%3A%22file%3A%2F%2F%2Fc%253A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2FREADME.md%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2FREADME.md%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A50%2C%22character%22%3A5%7D%7D%5D%5D "Go to definition")

This script installs dependencies, configures logging, and defines the training process. It includes functions to get the model by name, criterion, and optimizer. The main training loop handles the training process, including loading the dataset, moving data to the appropriate device (CPU/GPU), and logging the training progress.

### Example Usage

To run the training script, execute the following command:

```sh
python run_training.py
```

## Using SageMaker Role Across All Folders

To ensure that the same SageMaker role is used across all folders, define the role in a common configuration file and reference it in each script. For example, create a [`config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fconfig.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\giopl\OneDrive\Desktop\python_projects\PyTorch-Use-Cases\config.py") file in the root directory:

```python
# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the IAM role from environment variable
SAGEMAKER_ROLE = os.getenv('SAGEMAKER_ROLE')

if not SAGEMAKER_ROLE:
    raise ValueError("SAGEMAKER_ROLE environment variable not set")
```

Then, in each script, import the role from [`config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fgiopl%2FOneDrive%2FDesktop%2Fpython_projects%2FPyTorch-Use-Cases%2Fconfig.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\giopl\OneDrive\Desktop\python_projects\PyTorch-Use-Cases\config.py"):

```python
# Example: computer_vision/image_classification/run_training.py
from sagemaker.pytorch import PyTorch
import sagemaker
import tempfile
from config import SAGEMAKER_ROLE

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Create a temporary directory for dummy data
tmpdirname = tempfile.mkdtemp()

# Generate dummy data and save it to the temporary directory
dummy_data_path = os.path.join(tmpdirname, 'dummy_data.txt')
with open(dummy_data_path, 'w') as f:
    f.write('dummy data')

source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'computer_vision', 'image_classification'))

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
    }
)
# Fit the estimator using local mode
estimator.fit(f'file://{tmpdirname}')
```

This way, you can ensure that the same SageMaker role is used consistently across all folders in your repository.

## AWS Cost Management

Managing costs effectively is crucial when using AWS services. Here are some strategies and tools we use to optimize costs:

### Free Tier Exploitation

We take advantage of AWS's free tier offerings to minimize costs during the initial stages of development and testing. This includes using free tier eligible services and resources wherever possible.

### Spot Instances

To further reduce costs, we use Spot Instances for our training jobs. Spot Instances allow us to use spare AWS compute capacity at a reduced cost. Here are the configurations we use:

```python
use_spot_instances=True,  # Enable Spot Instances
max_wait=3600,  # Maximum wait time for Spot Instances (in seconds)
max_run=3600,  # Maximum runtime for the training job
```
### Cost Monitoring with [get_aws_cost.py](get_aws_cost.py)
We use a custom script, [get_aws_cost.py](get_aws_cost.py), to monitor our AWS costs immediately after running a job. This script uses the AWS Cost Explorer API to fetch cost information.

This script helps us keep track of our spending in real-time, allowing us to make informed decisions about resource usage and cost optimization.