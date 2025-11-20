# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging

import pytest

# Data Payloads
from tests.integration import api_data_sources as ds

from .utils import redacted_response_text

log = logging.getLogger("gateway_tests")
log.setLevel(logging.INFO)

pytestmark = pytest.mark.integration


def test_create_then_delete_data_source(gateway, caplog):
    caplog.set_level(logging.INFO, logger="gateway_tests")

    payload = ds.DATA_SOURCE_SENTINEL

    # --- Create ---
    r = gateway.post("/v2/data-sources", json=payload)
    log.info("CREATE /v2/data-sources (redacted):\n%s", redacted_response_text(r))
    assert r.status_code in (200, 201), r.text
    body = r.json()
    ds_id = body.get("id")
    assert ds_id

    # --- Cleanup ---
    rd = gateway.delete(f"/v2/data-sources/{ds_id}")
    log.info("DELETE /v2/data-sources/%s -> %s", ds_id, rd.status_code)
    assert rd.status_code in (200, 202, 204, 404), rd.text
