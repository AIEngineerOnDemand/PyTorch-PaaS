import boto3
from datetime import datetime, timezone, timedelta

# Initialize a session using Amazon Cost Explorer
client = boto3.client('ce')

# Define the time period for the cost report (last 30 days)
start = datetime.now(timezone.utc) - timedelta(days=30)
end = datetime.now(timezone.utc)

# Format the dates as required by AWS Cost Explorer
start_date = start.strftime('%Y-%m-%d')
end_date = end.strftime('%Y-%m-%d')

# Get cost and usage broken down by instance type
response = client.get_cost_and_usage(
    TimePeriod={
        'Start': start_date,
        'End': end_date
    },
    Granularity='DAILY',
    Metrics=['UsageQuantity'],
    Filter={
        'Dimensions': {
            'Key': 'INSTANCE_TYPE',
            'Values': ['ml.m5.xlarge']
        }
    },
    GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
)

# Print the usage information for ml.m5.xlarge instances
for result in response['ResultsByTime']:
    print(f"Date: {result['TimePeriod']['Start']}")
    for group in result['Groups']:
        service = group['Keys'][0]
        amount = group['Metrics']['UsageQuantity']['Amount']
        unit = group['Metrics']['UsageQuantity']['Unit']
        print(f"Service: {service}, Usage: {amount} {unit}")

        # Provide insights into potential reasons for the usage
        if service == 'Amazon SageMaker':
            print("  - Check the configuration of your training jobs and ensure they are within the expected usage limits.")