# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import warnings
from typing import Optional

import mlflow
import pandas as pd
import structlog

from gfmstudio.config import Settings, get_settings, settings

warnings.filterwarnings("ignore")

logger = logger = structlog.get_logger()


def mlflow_auth(
    insecure_tls: str = "true",
    settings_: Optional[Settings] = None,
):
    """Function to login to MLflow Client

    Parameters
    ----------
    insecure_tls : str, optional
        Whether to use insecure connections, by default "true"
    settings_ : Optional[Settings], optional
        param to grab env variables, by default None

    Returns
    -------
    MlflowClient
        authenticated instance of MlflowClient
    """

    if not settings_:
        settings_ = get_settings()
    logger.debug(f"Initialized mlflow url to {settings.MLFLOW_URL}")

    # Enable insecure tls connections
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = insecure_tls

    # Connect to Mlflow server
    mlflow.set_tracking_uri(uri=settings_.MLFLOW_URL)

    return mlflow.tracking.MlflowClient()


def get_metrics_per_epoch(metrics_dict: dict, metrics_cols: dict):
    """Function to get metrics per epoch

    Parameters
    ----------
    metrics_dict : dict
        Metrics dictionary from Mlflow
    metrics_cols : dict
        Metrics columns from Mlflow

    Returns
    -------
    pd.DataFrame
        combined Dataframe with the metrics and the columns
    """

    df_list = []
    for metric in metrics_cols:
        df = pd.DataFrame(
            [(m.step, m.value) for m in metrics_dict[metric]], columns=["epoch", metric]
        )
        df_list.append(df.to_dict())

    return df_list


def get_overall_runs_status(runs):
    statuses = [run.get("status") for run in runs]

    if "ERROR" in statuses:
        return "ERROR"
    elif "RUNNING" in statuses:
        return "RUNNING"
    elif "FINISHED" in statuses:
        return "FINISHED"
    else:
        return "NOT_FOUND"


def get_mlflow_metrics(tune_id: str):
    """Function to get Mlflow metrics

    Parameters
    ----------
    tune_id : str
        The tune id

    Returns
    -------
    tuple
        Tuple with status of tune experiment, number of epochs, metrics, error/message_detail
    """

    try:
        mlflow_client = mlflow_auth(insecure_tls="true")
        experiment_name = mlflow_client.get_experiment_by_name(name=tune_id)

        if experiment_name:
            # Get Id of experiment
            experiment_id = experiment_name.experiment_id

            # Get the list of runs associated with the experiment
            # In the current setup, one run only per experiment

            runs = mlflow_client.search_runs(experiment_ids=[experiment_id])

            if not runs:
                error_message = f"No run found for the experiment with tune_id {tune_id} found in Mlflow server. "
                logger.exception(error_message)

                status = "NOT_FOUND"
                runs = []
                detail = error_message

                return status, runs, detail

            run_output_list = []
            for run in runs:
                # Get ID of this run
                run_id = run.info.run_id
                run_name = run.info.run_name

                # Get all logged metrics for this run
                metrics = list(run.data.metrics.keys())

                metrics_dict = {}
                # For each metric, get values for all epochs
                for metric in metrics:
                    metrics_dict[f"{str(metric)}"] = mlflow_client.get_metric_history(
                        run_id, metric
                    )

                metrics_output = get_metrics_per_epoch(metrics_dict, metrics)

                # Getting length of any metric, tells how many epochs have already run.
                if metrics_dict:
                    # if train/loss and epoch in dict, drop them to avoid indexing them;
                    # If we have a large dataset and a smaller batch size defined,
                    # we will end up logging a significant number of [train/loss, epochs] per epoch.
                    metrics = [
                        metric
                        for metric in metrics
                        if metric not in ["train/loss", "epoch"]
                    ]
                    logger.debug(f"Experiment Metrics : {metrics}")
                    epochs = len(metrics_dict[metrics[0]])
                    status = run.info.status

                    run_output_list.append(
                        {
                            "name": run_name,
                            "status": status,
                            "epochs": str(epochs),
                            "metrics": metrics_output,
                        }
                    )
                else:
                    run_output_list.append(
                        {
                            "name": run_name,
                            "status": "NOT_FOUND",
                            "epochs": "",
                            "metrics": metrics_output,
                        }
                    )

            if run_output_list:
                status = get_overall_runs_status(run_output_list)
                runs = run_output_list
                detail = f"Retrieving logs returned: {status}"

                return status, runs, detail
            else:
                error_message = f"No metrics logged yet for the experiment with tune_id {tune_id} in Mlflow server. "
                logger.exception(error_message)

                status = "NOT_FOUND"
                runs = []
                detail = error_message

                return status, runs, detail

        else:
            error_message = (
                f"No experiment with tune_id {tune_id} found in Mlflow server. "
            )
            logger.exception(error_message)

            status = "NOT_FOUND"
            runs = []
            detail = error_message

            return status, runs, detail

    except Exception as e:
        logger.exception(e)

        status = "ERROR"
        runs = []
        detail = f"Something went wrong, Error details {e}"

        return status, runs, detail


def get_mlflow_experiment_id(tune_id: str):
    """Get Mlflow experiment id

    Parameters
    ----------
    tune_id : str
        Tune ID

    Returns
    -------
    int
        Experiment id
    """
    try:
        mlflow_client = mlflow_auth(insecure_tls="true")
        experiment_name = mlflow_client.get_experiment_by_name(name=tune_id)
        if experiment_name:
            # Get Id of experiment
            experiment_id = experiment_name.experiment_id
            return int(experiment_id)
        else:
            error_message = (
                f"No experiment with tune_id {tune_id} found in Mlflow server. "
            )
            logger.exception(error_message)
            return

    except Exception as e:
        logger.exception(e)


def get_mlflow_experiment(experiment_id: int):
    """Get all the Mlflow experiment details

    Parameters
    ----------
    experiment_id : int
        Experiment ID

    Returns
    -------
    Mlflow Experiment
        Mlflow experiment details for the Mlflow ID
    """
    try:
        if experiment_id:
            experiment = mlflow.get_experiment(experiment_id=experiment_id)
            return experiment
        else:
            error_message = f"No experiment with Experiment ID {experiment_id} found in Mlflow server. "
            logger.exception(error_message)
            return
    except Exception as e:
        logger.exception(e)


def get_experiment_by_tag(experiment_id: int, user):
    """Get all experiments with a tag of user(email_address)

    Parameters
    ----------
    experiment_id : int
        Experiment ID
    user : logged in user
        User to filter experiments by

    Returns
    -------
    list
        List of experiments tagged by user
    """
    try:
        mlflow_client = mlflow_auth(insecure_tls="true")

        experiments = mlflow_client.search_experiments()
        filtered_experiments = [
            exp for exp in experiments if exp.tags.get("name") == user
        ]

        if filtered_experiments:
            return filtered_experiments
        else:
            error_message = (
                f"No experiments with the tag {user} found in Mlflow server. "
            )
            logger.exception(error_message)
            return
    except Exception as e:
        logger.exception(e)
