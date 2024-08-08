import boto3
from datetime import datetime, timezone, timedelta

# Initialize a session using Amazon Cost Explorer
client = boto3.client('ce')

# Define the time period for the cost report
start = datetime.now(timezone.utc)
end = start + timedelta(days=1)

# Format the dates as required by AWS Cost Explorer
start_date = start.strftime('%Y-%m-%d')
end_date = end.strftime('%Y-%m-%d')

# Get cost and usage
response = client.get_cost_and_usage(
    TimePeriod={
        'Start': start_date,
        'End': end_date
    },
    Granularity='DAILY',
    Metrics=['UnblendedCost']
)

# Print the cost information
for result in response['ResultsByTime']:
    print(f"Date: {result['TimePeriod']['Start']}, Cost: {result['Total']['UnblendedCost']['Amount']} {result['Total']['UnblendedCost']['Unit']}")