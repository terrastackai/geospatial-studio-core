# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging

import pytest

# Data Payloads
from tests.integration import api_inferences as inf
from tests.integration import api_models as ds

from .utils import redacted_response_text

log = logging.getLogger("gateway_tests")
log.setLevel(logging.INFO)

pytestmark = pytest.mark.integration


def test_create_inference_request(gateway, caplog):
    caplog.set_level(logging.INFO, logger="gateway_tests")

    # --- Create Model ---
    payload = ds.SANDBOX_MODEL
    r_model = gateway.post("/v2/models", json=payload)
    log.info("CREATE /v2/models (redacted):\n%s", redacted_response_text(r_model))
    assert r_model.status_code in (200, 201), r_model.text
    body = r_model.json()
    model_id = body.get("id")
    assert model_id

    # --- Create Inference ---
    payload = inf.INFERENCE_WX_WIND_TEXAS
    r_inference = gateway.post("/v2/inference", json=payload)
    log.info(
        "CREATE /v2/inference (redacted):\n%s", redacted_response_text(r_inference)
    )
    assert r_inference.status_code in (200, 201, 202), r_inference.text
    body = r_inference.json()
    assert body is not None
    inference_id = body.get("id")
    if inference_id:
        log.info("INFERENCE ID: %s", inference_id)

    # --- Cleanup Inference ---
    rd_inference = gateway.delete(f"/v2/inference/{inference_id}")
    log.info("DELETE /v2/inference/%s -> %s", inference_id, rd_inference.status_code)
    assert rd_inference.status_code in (200, 202, 204, 404), rd_inference.text

    # --- Cleanup Model ---
    rd_model = gateway.delete(f"/v2/models/{model_id}")
    log.info("DELETE /v2/models/%s -> %s", model_id, rd_model.status_code)
    assert rd_model.status_code in (200, 202, 204, 404), rd_model.text
