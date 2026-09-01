# Rate Limiting Configuration

## Default Rate Limits

| Endpoint | Method | Requests | Window | Per Minute |
|----------|--------|----------|--------|------------|
| Default (all endpoints) | ALL | 200 | 60s | 200/min |
| `/v2/async/inference/` | POST | 6 | 300s | 1.2/min |
| `/v2/submit-tune` | POST | 6 | 300s | 1.2/min |
| `/v2/datasets/onboard` | POST | 6 | 300s | 1.2/min |
| `/health/livez` | GET | 100 | 60s | 100/min |
| `/health/readyz` | GET | 100 | 60s | 100/min |

## Environment Variables

### Enable/Disable

```bash
# Enable rate limiting (default: false)
RATELIMIT_ENABLED=true

# Redis connection (required when enabled)
REDIS_URL=redis://localhost:6379/0
```

### Adjust Default Limits

```bash
# Default limit for all endpoints
RATELIMIT_LIMIT=200          # requests
RATELIMIT_WINDOW=60          # seconds

# Sensitive resources (inference, training, datasets)
RATELIMIT_SENSITIVE_RESOURCE_LIMIT=6    # requests
RATELIMIT_SENSITIVE_RESOURCE_WINDOW=300 # seconds
```

### Customize Specific Endpoints

Set `RATE_LIMIT_CONFIG` as a JSON string in your environment:

```bash
RATE_LIMIT_CONFIG='{"default": {"/your/endpoint": {"POST": {"limit": 10, "window": 60}, "GET": {"limit": 50, "window": 60}}}}'
```

**Format:**
```json
{
  "default": {
    "/endpoint/path": {
      "METHOD": {
        "limit": <number>,
        "window": <seconds>
      }
    }
  }
}
```

**Example:** Custom limits for a new endpoint
```bash
RATE_LIMIT_CONFIG='{"default": {"/v2/my-endpoint": {"POST": {"limit": 20, "window": 120}}}}'
```

The custom config will be merged with built-in defaults. Custom endpoints take precedence.

## Quick Start

**Enable in `.env`:**
```bash
RATELIMIT_ENABLED=true
REDIS_URL=redis://localhost:6379/0
```

**Adjust defaults (optional):**
```bash
RATELIMIT_LIMIT=500                      # Increase general limit
RATELIMIT_SENSITIVE_RESOURCE_LIMIT=10   # Increase ML ops limit
```

**Add custom endpoint limits (optional):**
```bash
RATE_LIMIT_CONFIG='{"default": {"/v2/custom": {"POST": {"limit": 15, "window": 60}}}}'
```

**Restart application**

## Disable Rate Limiting

```bash
RATELIMIT_ENABLED=false
```
