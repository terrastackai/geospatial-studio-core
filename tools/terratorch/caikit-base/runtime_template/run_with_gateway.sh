#!/usr/bin/env bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


################################################################################
# This script is the entrypoint for the multi-process runtime container that
# runs the REST gateway alongside the grpc runtime.
# The multiprocess management is intended to be handled by `tini`, the tiny
# but valid `init`.
################################################################################

set -e

echo '[STARTING RUNTIME]'

cd /app && python3 start_runtime.py &

RUNTIME_PORT=${SERVICE_PORT:-8085}

# If TLS enabled, make an https call, otherwise make an http call
protocol="http"
if [ "${TLS_SERVER_KEY}" != "" ] && [ "${TLS_SERVER_CERT}" != "" ]
then
    protocol="--cacert $TLS_SERVER_CERT https"
    if [ "${TLS_CLIENT_CERT}" != "" ]
    then
        protocol="-k --cert $TLS_SERVER_CERT --key $TLS_SERVER_KEY https"
    fi
fi

(
    # Wait for the Runtime to come up before starting the gateway
    sleep 3
    until $(curl --insecure --http2-prior-knowledge --output /dev/null --silent --fail ${protocol}://localhost:${RUNTIME_PORT}); do
        echo '.'
        sleep 1
    done

    echo '[STARTING GATEWAY]'
    PROXY_ENDPOINT="localhost:${RUNTIME_PORT}" SERVE_PORT=${GATEWAY_PORT:-8080} /gateway --swagger_path=/swagger --proxy_no_cert_val=true
) &

wait -n