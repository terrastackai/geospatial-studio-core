# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text

# Import event publishing
try:
    from pipelines.general_libraries.eventing import (
        create_task_ready_event,
        publish_event,
    )

    EVENTING_AVAILABLE = True
except ImportError:
    EVENTING_AVAILABLE = False

    def publish_event(*args, **kwargs):
        return False


# Configuration
process_id = os.getenv("process_id", "unknown-process")
process_exec = os.getenv("process_exec", "python process.py")
orchestrate_db_uri = os.getenv("orchestrate_db_uri", "")
inf_task_table = os.getenv("inference_task_table", "inf_task")
stop_exit_code = int(os.getenv("stop_exit_code", 177))
log_level = os.getenv("log_level", "INFO")
event_driven_mode = os.getenv("EVENT_DRIVEN_MODE", "false").lower() == "true"

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Flask app for receiving CloudEvents
app = Flask(__name__)
app.logger.setLevel(log_level)

# Database engine
engine = create_engine(orchestrate_db_uri) if orchestrate_db_uri else None


def run_and_log(task_id, process_exec, process_id, inference_folder):
    """Execute process and capture logs (same as original)."""
    std_out_log_name = f"{inference_folder}/{task_id}/{task_id}-{process_id}-stdout.log"
    std_err_log_name = f"{inference_folder}/{task_id}/{task_id}-{process_id}-stderr.log"

    try:
        with open(std_out_log_name, "a", buffering=1) as so:
            with open(std_err_log_name, "a", buffering=1) as _se:
                so.write("-----INVOKING TASK (EVENT-DRIVEN)-------------------\n")
                so.write(f"Task ID: {task_id}\n")
                so.write(f"Command: {process_exec}\n")
                so.flush()

                try:
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    env["GFM_STDOUT_LOG"] = std_out_log_name
                    env["GFM_STDERR_LOG"] = std_err_log_name
                    env["GFM_LOG_LEVEL"] = "DEBUG"

                    process = subprocess.Popen(
                        process_exec,
                        shell=True,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    if process.stdout:
                        for line in process.stdout:
                            so.write(line)
                            so.flush()

                    returncode = process.wait()
                    so.write(f"\nReturn code: {returncode}\n")
                    so.flush()

                    return returncode

                except Exception as ex:
                    error_msg = f"Task ID: {task_id} Command: {process_exec} exited with error: {ex}"
                    so.write(f"\nError: {error_msg}\n")
                    so.flush()
                    return 500

    except Exception as ex:
        logger.error(
            f"Task ID: {task_id} Command: {process_exec} exited with error: {ex}"
        )
        return 500


def update_status_and_publish_next(
    engine, process_id, inference_id, task_id, new_state
):
    """
    Update task status in database and publish event for next step.
    This is the KEY function that maintains the sequential workflow.
    """
    task_terminal_status_set = False
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Get current pipeline steps
    with engine.connect() as conn:
        task_pipeline_sql = text(
            f"SELECT pipeline_steps, inference_folder, priority FROM {inf_task_table} WHERE task_id = '{task_id}';"
        )
        result = conn.execute(task_pipeline_sql).fetchall()[0]
        pipeline_steps = result[0]
        inference_folder = result[1]
        priority = result[2] if len(result) > 2 else 5

    # Update current step status
    next_step_number = None
    for s in range(0, len(pipeline_steps)):
        if pipeline_steps[s]["process_id"] == process_id:
            pipeline_steps[s]["status"] = new_state
            pipeline_steps[s]["end_time"] = end_time
            next_step_number = pipeline_steps[s]["step_number"] + 1

    # Mark next step as READY if current step finished successfully
    next_step_process_id = None
    if new_state == "FINISHED":
        if next_step_number is not None and next_step_number < len(pipeline_steps):
            for s in range(0, len(pipeline_steps)):
                if pipeline_steps[s]["step_number"] == next_step_number:
                    pipeline_steps[s]["status"] = "READY"
                    next_step_process_id = pipeline_steps[s]["process_id"]
                    logger.info(f"Marking next step READY: {next_step_process_id}")

    # Handle STOPPED state
    if new_state == "STOPPED":
        if next_step_number is not None and next_step_number < len(pipeline_steps):
            for s in range(0, len(pipeline_steps)):
                if pipeline_steps[s]["process_id"] == process_id:
                    pipeline_steps[s]["status"] = "FINISHED"
                elif pipeline_steps[s]["step_number"] >= next_step_number:
                    pipeline_steps[s]["status"] = "STOPPED"

    # Update database
    with engine.connect() as conn:
        update_status = text(
            f"""UPDATE {inf_task_table} SET pipeline_steps = '{json.dumps(pipeline_steps)}'  WHERE task_id = '{task_id}';"""
        )
        conn.execute(update_status)
        conn.commit()

        # Check overall task status
        if all([X["status"] == "FINISHED" for X in pipeline_steps]):
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'FINISHED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True
            logger.info(f"Task {task_id} completed - all steps finished")

        if any([X["status"] == "STOPPED" for X in pipeline_steps]):
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'STOPPED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True

        if any([X["status"] == "FAILED" for X in pipeline_steps]):
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'FAILED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True

    # CRITICAL: Publish event for next step if in event-driven mode
    if event_driven_mode and next_step_process_id and new_state == "FINISHED":
        logger.info(f"Publishing event for next step: {next_step_process_id}")
        try:
            event = create_task_ready_event(
                task_id=task_id,
                inference_id=inference_id,
                process_id=next_step_process_id,
                step_number=next_step_number,
                inference_folder=inference_folder,
                pipeline_steps=pipeline_steps,
                priority=priority,
                metadata={
                    "previous_step": process_id,
                    "triggered_by": "step_completion",
                },
            )
            success = publish_event(event)
            if success:
                logger.info(f"Successfully published event for {next_step_process_id}")
            else:
                logger.error(f"Failed to publish event for {next_step_process_id}")
        except Exception as ex:
            logger.error(f"Error publishing next step event: {ex}", exc_info=True)

    return task_terminal_status_set


@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "process_id": process_id}), 200


@app.route("/readyz", methods=["GET"])
def readyz():
    """Readiness check endpoint."""
    return jsonify({"status": "ready", "process_id": process_id}), 200


@app.route("/process", methods=["POST"])
def process_event():
    """
    Handle incoming CloudEvent for task processing.
    This endpoint receives events and processes tasks just like the polling version.
    """
    try:
        # Parse CloudEvent
        event_data = request.get_json()

        if not event_data:
            return jsonify({"error": "No event data provided"}), 400

        logger.info(f"Received event: {event_data.get('type')}")

        # Extract task data from CloudEvent
        data = event_data.get("data", {})
        task_id = data.get("task_id")
        inference_id = data.get("inference_id")
        inference_folder = data.get("inference_folder")

        if not all([task_id, inference_id, inference_folder]):
            return jsonify({"error": "Missing required task data"}), 400

        logger.info(f"Processing task: {task_id}, process: {process_id}")

        # Set environment variables (same as polling version)
        os.environ["inference_id"] = inference_id
        os.environ["inference_folder"] = inference_folder
        os.environ["task_id"] = task_id

        # Execute process (same as polling version)
        return_value = run_and_log(task_id, process_exec, process_id, inference_folder)
        logger.info(f"Process returned: {return_value}")

        # Update status and publish next step event
        if return_value == 0:
            logger.info(f"Finished running code for {task_id}, updating status")
            update_status_and_publish_next(
                engine, process_id, inference_id, task_id, "FINISHED"
            )
            return jsonify({"status": "completed", "task_id": task_id}), 200

        elif return_value == stop_exit_code:
            logger.info(f"Stopped running code for {task_id}, updating status")
            update_status_and_publish_next(
                engine, process_id, inference_id, task_id, "STOPPED"
            )
            return jsonify({"status": "stopped", "task_id": task_id}), 200

        else:
            logger.error(f"Failed running code for {task_id}, updating status")
            update_status_and_publish_next(
                engine, process_id, inference_id, task_id, "FAILED"
            )
            return (
                jsonify({"status": "failed", "task_id": task_id, "code": return_value}),
                500,
            )

    except Exception as ex:
        logger.error(f"Error processing event: {ex}", exc_info=True)
        return jsonify({"error": str(ex)}), 500


def main():
    """Main entry point for event-driven handler."""
    port = int(os.getenv("PORT", 8080))
    logger.info(
        f"Starting event-driven orchestrate wrapper for {process_id} on port {port}"
    )
    logger.info(f"Event-driven mode: {event_driven_mode}")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

# Made with Bob
