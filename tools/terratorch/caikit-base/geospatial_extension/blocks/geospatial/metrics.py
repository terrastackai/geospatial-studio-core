import time

from prometheus_client import Counter, Gauge, Info, Summary


METRIC_NAME_PREFIX = "infsvc_"

APP_INFO = Info('app_version', 'The version of the running application')
# TODO: get versions from a config file to all dynamic values.
APP_INFO.info({'version': '0.0.1', 'caikit_version': '0.19.0'})

# resolution of data/size of bbox
BBOX_SIZE = Gauge('data_resolution_or_bbox_size',
                  'resolution of data/size of bbox')

# TODO: In general how large a request is (i.e. pixels*timestamps)

# duration of the different aspects of the inference pipeline (data pull, inference, data push, entire pipeline)
# in this case, `(data pull, inference, data push, entire pipeline)` are different actions
# REQUEST_TIME metric tracks time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds',
                       'Time spent processing request', ['action'])

# number of requests
# the number of requests has been already met by using the `REQUEST_TIME` Summary metric above
# here is what it is called: `request_processing_seconds_count`: total number of calls made
# with this and the `action` label, we're able to get calls made for different actions.


# Metrics
RUNNING_GAUGE_LABELS =  {"user", "data_type", "step"}
RUNNING_GAUGE_METRIC = Gauge(
    name=f"{METRIC_NAME_PREFIX}_inference_inprogress",
    documentation="Total inferences started metric",
    labelnames=RUNNING_GAUGE_LABELS,
)

DURATION_SUMMARY_LABELS =  {"user", "data_type", "step"}
DURATION_SUMMARY_METRIC = Summary(
    name=f"{METRIC_NAME_PREFIX}_inference_running_seconds",
    documentation="Time spent processing an inference",
    labelnames=DURATION_SUMMARY_LABELS,
)


TOTAL_IMAGES_METRIC = Gauge(
    name=f"{METRIC_NAME_PREFIX}_request_images_total",
    documentation="Total Images in a request.",
    labelnames={"user", "data_type"}
)

TOTAL_TIMESTAMPS_METRIC = Gauge(
    name=f"{METRIC_NAME_PREFIX}_request_timestamps_total",
    documentation="Total number of timestamps in a request.",
    labelnames={"user", "data_type"}
)


EXCEPTIONS_COUNTER_LABELS =  {"user", "data_type"}
EXCEPTIONS_COUNTER = Counter(
    name=f"{METRIC_NAME_PREFIX}_exceptions_total",
    documentation="Count total exceptions encountered",
    labelnames=EXCEPTIONS_COUNTER_LABELS,
)
