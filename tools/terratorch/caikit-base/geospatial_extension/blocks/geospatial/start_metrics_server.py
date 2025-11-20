import random

from prometheus_client import start_http_server

from .metrics import process_request

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8000)

    # Generate some requests.
    while True:
        process_request(random.random())
