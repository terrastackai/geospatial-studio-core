import ast
from collections.abc import Iterable
from functools import wraps


def emit_prometheus_metrics(metric_funcs):
    """
    Decorator for Prometheus metrics.

    This decorator takes a Prometheus metric (Counter, Gauge, Summary, Histogram) as the
    first argument and adds labels based on the function's key-value arguments.

    Parameters:
        metric_funcs: The Prometheus metric function or a list of functions to be called.

    Returns:
        function: The decorated function.

    """
    if not isinstance(metric_funcs, list):
        metric_funcs = [metric_funcs]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that updates the Prometheus metric based on the function's
            key-value arguments.

            Parameters:
                *args: Positional arguments.
                **kwargs: Key-value arguments.

            Returns:
                Any: The result of the original function.
            """
            # Extract labels from function arguments
            # Workaround to extract labels from the string argument in the inference service.
            if args:
                                    
                if isinstance(args, Iterable) and isinstance(args[0], str):
                    try:
                        json_input = ast.literal_eval(args[0])
                    except Exception:
                        json_input = args[0]
                elif isinstance(args, Iterable) and isinstance(args[0], object):
                    try:
                        json_input = ast.literal_eval(args[1])
                    except Exception:
                        json_input = {}
                    json_input["model"] = args[0].model
                else:
                    json_input = kwargs
            else:
                json_input = kwargs

            for metric_func, metric_labels in metric_funcs:
                function_labels = {label: json_input.get(label) for label in metric_labels}
                if "user" in function_labels and not function_labels.get("user"):
                    function_labels["user"] = "systemuser@ibm.com"
                metric_func(function_labels)

            return func(*args, **kwargs)

        return wrapper

    return decorator
