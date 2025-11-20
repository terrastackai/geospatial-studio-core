# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging

import pytest

# Data Payloads
from tests.integration import api_models as ds

from .utils import redacted_response_text

log = logging.getLogger("gateway_tests")
log.setLevel(logging.INFO)

pytestmark = pytest.mark.integration


def test_create_then_delete_model(gateway, caplog):
    caplog.set_level(logging.INFO, logger="gateway_tests")

    payload = ds.SANDBOX_MODEL

    # Create
    r = gateway.post("/v2/models", json=payload)
    log.info("CREATE /v2/models (redacted):\n%s", redacted_response_text(r))
    assert r.status_code in (200, 201), r.text
    body = r.json()
    model_id = body.get("id")
    assert model_id

    # Cleanup
    rd = gateway.delete(f"/v2/models/{model_id}")
    log.info("DELETE /v2/models/%s -> %s", model_id, rd.status_code)
    assert rd.status_code in (200, 202, 204), rd.text
