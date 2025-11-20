# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import logging

import pytest

from .utils import mask_secret_string, redacted_response_text

log = logging.getLogger("gateway_tests")
log.setLevel(logging.INFO)

pytestmark = pytest.mark.integration


@pytest.fixture
def minted_api_key(gateway, caplog):
    caplog.set_level(logging.INFO, logger="gateway_tests")

    # --- SETUP (create api_key) ---
    r = gateway.post("/v2/auth/api-keys", json={})
    log.info("CREATE response (redacted):\n%s", redacted_response_text(r))
    assert r.status_code in (200, 201), r.text
    assert r.headers.get("Content-Type", "").startswith(
        "application/json"
    ), r.headers.get("Content-Type")

    data = r.json()
    apikey_id = data.get("id")
    apikey_value = data.get("value")
    assert apikey_id and apikey_value, json.dumps(data, indent=2)
    log.info(
        "created key id=%s value=%s active=%s expires=%s",
        apikey_id,
        mask_secret_string(apikey_value),
        data.get("active"),
        data.get("expires_on"),
    )

    # --- USE (return to test_generate_then_delete_api_key) ---
    yield {"id": apikey_id, "value": apikey_value, "raw": data}

    # --- TEARDOWN (delete api_key) ---
    rr = gateway.delete("/v2/auth/api-keys", params={"apikey_id": apikey_id})
    log.info("DELETE -> %s (url=%s)", rr.status_code, rr.request.url)
    assert rr.status_code == 204, rr.text


def test_list_api_keys(gateway, caplog):
    caplog.set_level(logging.INFO, logger="gateway_tests")
    r = gateway.get("/v2/auth/api-keys")
    log.info("response (redacted):\n%s", redacted_response_text(r))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body and isinstance(body["results"], list)


def test_generate_then_delete_api_key(gateway, minted_api_key):
    # Sanity check: fixture works
    assert minted_api_key["id"]

    # Verify api_keys
    r = gateway.get("/v2/auth/api-keys")
    assert r.status_code == 200, r.text
    ids = [it.get("id") for it in r.json().get("results", [])]
    assert minted_api_key["id"] in ids
