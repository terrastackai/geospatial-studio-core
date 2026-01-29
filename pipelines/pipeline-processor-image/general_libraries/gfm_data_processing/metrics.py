# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import time
from functools import wraps
from opentelemetry import metrics,trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter,SimpleSpanProcessor

class MetricManager:
    def __init__(self,component_name: str, service_name="inference-service-name" ):
        # Metrics Setup
        self.reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        self.provider = MeterProvider(metric_readers=[self.reader])
        metrics.set_meter_provider(self.provider)
        meter_name = f"{component_name}-meter"
        self.meter = metrics.get_meter(meter_name)
        
        # Tracing SetUp which should be on start up
        self.span_exporter = ConsoleSpanExporter()
        self.span_processor = SimpleSpanProcessor(self.span_exporter)
        self.trace_provider = TracerProvider()
        self.trace_provider.add_span_processor(self.span_processor)
        trace.set_tracer_provider(self.trace_provider)
        self.tracer = trace.get_tracer(service_name) 
        
        # Duration tracking for components
        self.duration_histogram = self.get_histogram(
            name=f"{component_name}_duration_sec",
            description=f"Duration of {component_name.replace('_',' ')} steps",
            unit="s"
        )
        
        self.failure_counter = self.get_counter(
            name=f"{component_name}_failures_total",
            description=f"Total {component_name.replace('_',' ')} failures",
            unit = "1"
        )
        

    def get_histogram(self, name, description="", unit="1"):
        return self.meter.create_histogram(name=name, description=description, unit=unit)

    def get_counter(self, name, description="", unit="1"):
        return self.meter.create_counter(name=name, description=description, unit=unit)

    def record_duration(self, histogram= None, **base_attributes):
        """Decorator to record how long a function takes to run."""
        histogram = histogram or self.duration_histogram
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.time() - start_time
                    histogram.record(duration, attributes=base_attributes)
                    print(f"{func.__name__} took {duration:.3f} seconds")
            return wrapper
        return decorator

    def count_failures(self, counter=None, span_name_prefix="", **base_attributes):
        """Decorator to increment a count when a function raises an exception AND record the failure in a span."""
        counter = counter or self.failure_counter
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                actual_span_name= f"{span_name_prefix}{func.__name__}" if span_name_prefix else func.__name__
                with self.tracer.start_as_current_span(actual_span_name) as span:
                    # Add base attributes to the span
                    for key, value in base_attributes.items():
                        span.set_attribute(key, value)

                    try:
                        result = func(*args, **kwargs)
                        span.set_status(trace.StatusCode.OK) # Set span status to OK on success
                        return result
                    except Exception as e:
                        counter.add(1, attributes=base_attributes) # Increment failure metric
                        span.record_exception(e) # Record exception in the span
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e))) # Set span status to ERROR on exception
                        raise # Re-raise the exception
            return wrapper
        return decorator
