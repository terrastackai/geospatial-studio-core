# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import sys

import alog
import caikit_health_probe.__main__ as chealth

log = alog.use_channel("PROBE")


def _http_readinesss_check():
    """Probe the http server.

    This implementations is necessary as a workaround to support the watsonx integration.
    We need a workaround to only check the readiness of the http-server even when it is not
    enabled in the config/config.yaml file.

    We currently use the http requests for the integration. Enabling that in the config/config.yaml
    file means the caikit[http] dependencies have to be installed. Those dependencies includes
    pydanticV2 which is currently not supported by lightly (a dependency of terratorch). We are
    therefore not enabling http in the config file which means the current implementation of the
    `readiness_probe` (https://github.com/caikit/caikit/blob/main/caikit_health_probe/__main__.py#L49)
    will find it as inactive and skip readiness prob for the http server.

    """
    config = chealth.get_config()
    log.debug("Checking HTTP server health")
    http_ready = chealth._http_readiness_probe(
        config.runtime.http.port,
        config.runtime.tls.server.key,
        config.runtime.tls.server.cert,
        config.runtime.tls.client.cert,
    )
    if http_ready is False:
        log.error("Runtime server(s) not ready. HTTP: %s", http_ready)
        sys.exit(1)


if __name__ == "__main__":
    _http_readinesss_check()
