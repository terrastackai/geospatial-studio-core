from unittest import TestCase

import pytest
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    REGISTRY,
)

from geospatial_extension.blocks.geospatial.metrics_utils import emit_prometheus_metrics


################################################################
# USAGE EXAMPLE
################################################################

# REGISTRY = CollectorRegistry()
BASE_LABELS = {"user", "model", "data_source", "node"}

# A metric that only uses the base/common labels
INF_RUN_COUNTER_METRIC = Counter(
    name="infsvc_inference_run_count",
    documentation="Total Inferences Counter Metric",
    labelnames=BASE_LABELS,
)

# A metric that add custom labels to the base labels
INF_ACTIVE_GAUGE_LABELS = BASE_LABELS | {"region"}
INF_ACTIVE_GAUGE_METRIC = Gauge(
    name="infsvc_inference_active_gauge",
    documentation="Active Inferences Gauge Metric",
    labelnames=INF_ACTIVE_GAUGE_LABELS,
)


@emit_prometheus_metrics([
    (lambda labels: INF_RUN_COUNTER_METRIC.labels(**labels).inc(2), BASE_LABELS),
    (lambda labels: INF_RUN_COUNTER_METRIC.labels(**labels).count_exceptions(), BASE_LABELS),
])
def example_function_counter(x, data={"user": "a.gmail.com"}):
    """
    Example function using the Counter metric decorator.
    """
    if not isinstance(data, dict):
        # Dummy exception to test .count_exceptions
        raise TypeError("expected sting, found integer type.")
    return 178, True


@emit_prometheus_metrics([
    (lambda labels: INF_ACTIVE_GAUGE_METRIC.labels(**labels).track_inprogress(), INF_ACTIVE_GAUGE_LABELS)
])
def example_function_gauge(x, data={"user": "a.gmail.com"}):
    """
    Example function using the Gauge metric decorator.
    """
    return 42

@emit_prometheus_metrics([
    (lambda labels: INF_RUN_COUNTER_METRIC.labels(**labels).inc(), BASE_LABELS),
    (lambda labels: INF_ACTIVE_GAUGE_METRIC.labels(**labels).track_inprogress(), INF_ACTIVE_GAUGE_LABELS)
])
def example_function_counter_n_gauge(x, data={"user": "a.gmail.com"}):
    """
    Example function using the Gauge metric decorator.
    """
    return "completed"

class TestEmitPrometheusMetricsDecorator(TestCase):

    def test_counter_metric_decorator(self):
        # Test with valid values, function executed once
        example_function_counter('{"user": "bb.gmail.com"}')
        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_run_count_total{data_source="None",model="None",node="None",user="bb.gmail.com"} 2.0' in metric_data

        # Test with valid values, function executed more than once
        for _ in range(4):
            example_function_counter('{"user": "aa.gmail.com"}')
        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_run_count_total{data_source="None",model="None",node="None",user="bb.gmail.com"} 2.0' in metric_data
        assert b'infsvc_inference_run_count_total{data_source="None",model="None",node="None",user="aa.gmail.com"} 8.0' in metric_data

        # Test with an exception
        with pytest.raises(Exception):
            example_function_counter('{"user": "cc.gmail.com"}', 100)
        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_run_count_total{data_source="None",model="None",node="None",user="cc.gmail.com"} 2.0' in metric_data

    def test_gauge_metric_decorator(self):
        # Test with valid values
        example_function_gauge('{"user": "aa.gmail.com"}')
        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_active_gauge{data_source="None",model="None",node="None",region="None",user="aa.gmail.com"} 0.0' in metric_data

        for _ in range(4):
            example_function_gauge('{"user": "bb.gmail.com"}')

        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_active_gauge{data_source="None",model="None",node="None",region="None",user="bb.gmail.com"} 0.0' in metric_data

    def test_counter_and_gauge_metric_decorator(self):
        example_function_counter_n_gauge('{"user": "dd.gmail.com"}')
        metric_data = generate_latest(REGISTRY)
        assert b'infsvc_inference_active_gauge{data_source="None",model="None",node="None",region="None",user="dd.gmail.com"} 0.0' in metric_data
        assert b'infsvc_inference_run_count_total{data_source="None",model="None",node="None",user="dd.gmail.com"} 1.0' in metric_data
