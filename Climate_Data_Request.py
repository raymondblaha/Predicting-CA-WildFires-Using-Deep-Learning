import requests
import csv
from datetime import datetime, timedelta
import time
import os

def fetch_data_for_month(url, headers, start_date, end_date):
    # Function to fetch data for a single month
    all_data = []
    offset = 1
    max_retries = 5  # Maximum number of retries per request
    retry_delay = 60  # Initial delay between retries in seconds
    params = {
        'datasetid': 'GHCND',
        'locationid': 'FIPS:06',
        'startdate': start_date.strftime('%Y-%m-%d'),
        'enddate': end_date.strftime('%Y-%m-%d'),
        'limit': 1000,
        'units': 'standard'
    }
    while True:
        params['offset'] = offset
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            if not results:
                break

            all_data.extend(results)
            if results:
                print(f"Successfully fetched data for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            total_count = data['metadata']['resultset']['count']
            if offset >= total_count:
                break
            offset += params['limit']

        except requests.exceptions.HTTPError as e:
            if response.status_code == 503 and max_retries > 0:
                print("Service unavailable, retrying after a delay of {} seconds...".format(retry_delay))
                max_retries -= 1
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f"HTTP error: {e}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error during requests to {url}: {e}")
            break

    return all_data

# Define API URL and headers
base_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
headers = {'token': 'YOUR_API_KEY_HERE'} # Update your API toke here. Keep in mind that there is a limit of 1000 requests. Tha will be good enoguht 

# Define the date range for the entire data fetch
end_year = 2023 # Since we can only get about 5 year and 7 months and 11 days before the request limit is reached. We can figure out how to run multiple scripts of this at one time so we can get the data faster. 

# Modify these to the year and month where you need to resume
resume_year = 2005
resume_month = 7  # Start from July 2005
resume_day = 11   # Start from 11th July 2005

for year in range(resume_year, end_year + 1):
    start_month = resume_month if year == resume_year else 1
    for month in range(start_month, 13):
        start_day = resume_day if year == resume_year and month == resume_month else 1
        start_date = datetime(year, month, start_day)
        end_date = start_date + timedelta(days=31)
        end_date = end_date.replace(day=1) - timedelta(days=1)
        print(f"Fetching data for {year}-{month:02d}")
        monthly_data = fetch_data_for_month(base_url, headers, start_date, end_date)
        
        if monthly_data:
            file_exists = os.path.isfile('climate_data.csv') and os.path.getsize('climate_data.csv') > 0
            with open('climate_data.csv', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=monthly_data[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(monthly_data)

        time.sleep(10)  # Fixed delay between each request

print("Data fetching complete.")