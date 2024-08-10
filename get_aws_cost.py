import boto3
from datetime import datetime, timezone, timedelta

# Initialize a session using Amazon Cost Explorer
client = boto3.client('ce')

# Define the time period for the cost report
start = datetime.now(timezone.utc) - timedelta(days=1)
end = datetime.now(timezone.utc)

# Format the dates as required by AWS Cost Explorer
start_date = start.strftime('%Y-%m-%d')
end_date = end.strftime('%Y-%m-%d')

# Get cost and usage broken down by service
response = client.get_cost_and_usage(
    TimePeriod={
        'Start': start_date,
        'End': end_date
    },
    Granularity='DAILY',
    Metrics=['UnblendedCost'],
    GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
)

# Print the cost information broken down by service
for result in response['ResultsByTime']:
    print(f"Date: {result['TimePeriod']['Start']}")
    for group in result['Groups']:
        service = group['Keys'][0]
        amount = group['Metrics']['UnblendedCost']['Amount']
        unit = group['Metrics']['UnblendedCost']['Unit']
        print(f"Service: {service}, Cost: {amount} {unit}")

        # Provide insights into potential reasons for the costs
        if service == 'Amazon EC2':
            print("  - Check if you are using Spot Instances or On-Demand Instances.")
            print("  - Ensure your usage stays within the free tier limits.")
        elif service == 'Amazon S3':
            print("  - Verify if you have exceeded the free tier limits for storage or requests.")
        elif service == 'AWS Lambda':
            print("  - Ensure your usage stays within the free tier limits for requests and compute time.")
        elif service == 'Amazon SageMaker':
            print("  - Check the configuration of your training jobs and ensure they are within the expected usage limits.")
        # Add more services and insights as needed