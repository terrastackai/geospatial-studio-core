import logging
import os


logger = logging.getLogger(__name__)


# Define regex patterns for supported storage services
COS_PATTERN = r"^http[s]?://.*\.cloud-object-storage\.appdomain\.cloud/.*"
S3_PATTERN = r"^http[s]?://.*\.amazonaws\.com/.*"
AZURE_BLOB_PATTERN = r"^http[s]?://.*\.blob\.core\.windows\.net/.*"
BOX_PATTERN = r"^http[s]?://.*\.box\.com/.*"
GCP_STORAGE_PATTERN = (
    r"^http[s]?://(storage\.googleapis\.com|.*\.storage\.googleapis\.com)/.*"
)

# List of allowed storage patterns.
ALLOWED_STORAGE_PATTERNS = [
    COS_PATTERN,
    S3_PATTERN,
    AZURE_BLOB_PATTERN,
    BOX_PATTERN,
    GCP_STORAGE_PATTERN,
]
DATA_SOURCE_REGEX = os.getenv("ADDITIONAL_DATA_SOURCES_REGEX", None)

ENVIRONMENT = os.getenv("ENVIRONMENT", "")

if DATA_SOURCE_REGEX:
    try:
        ALLOWED_STORAGE_PATTERNS += DATA_SOURCE_REGEX.split(",")
    except Exception:
        logger.warning(
            "Error creating list of allowed data sources. Using default allowed data sources."
        )

APPLICATION_DATA_DIR = os.getenv("APPLICATION_DATA_DIR", "/app/output/")
