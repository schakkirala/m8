# src/setup_dask.py
from dask.distributed import Client

def start_dask():
    client = Client()
    print("Start Dask client with the following details:")
    print(client.dashboard_link)
    return client