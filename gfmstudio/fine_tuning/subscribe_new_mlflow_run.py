# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import select
import threading
import time
from queue import Queue

import pg8000.dbapi
import pg8000.native
import structlog

"""
Logging settings.
"""
LOG_LEVEL = getattr(logging, "INFO")
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
logger = structlog.get_logger()


def parse_connection_string(conn_string):
    """
    Parse a PostgreSQL connection string into pg8000 connection parameters.

    Args:
        conn_string (str): Connection string in format postgresql+pg8000://user:pass@host:port/dbname

    Returns:
        dict: Dictionary of connection parameters for pg8000
    """
    # Remove postgresql+pg8000:// prefix
    conn_string = conn_string.replace("postgresql+pg8000://", "")

    # Split user:pass@host:port/dbname
    if "@" in conn_string:
        auth, location = conn_string.split("@", 1)
        user, password = auth.split(":", 1) if ":" in auth else (auth, None)
    else:
        user = None
        password = None
        location = conn_string

    # Split host:port/dbname
    if "/" in location:
        host_port, database = location.split("/", 1)
    else:
        host_port = location
        database = "postgres"

    # Split host:port
    if ":" in host_port:
        host, port = host_port.split(":", 1)
        port = int(port)
    else:
        host = host_port
        port = 5432

    params = {
        "host": host,
        "port": port,
        "database": database,
    }

    if user:
        params["user"] = user
    if password:
        params["password"] = password

    return params


def listen_to_notifications(
    listen_conn_string, update_conn_string, health_check_interval=10
):
    """
    Listen to notifications from the mlflow database and update the tunes table accordingly.

    Args:
        listen_conn_string (str): The connection string for the listen database.
        update_conn_string (str): The connection string for the update database.
        health_check_interval (int, optional): The interval in seconds to check the health of the connection. Defaults to 10.

    Returns:
        None
    """
    listen_conn = None
    update_conn = None

    try:
        listen_params = parse_connection_string(listen_conn_string)
        update_params = parse_connection_string(update_conn_string)

        listen_conn = pg8000.dbapi.connect(**listen_params)
        listen_conn.autocommit = True

        update_conn = pg8000.dbapi.connect(**update_params)
        update_conn.autocommit = True

        notification_queue = Queue()

        def notification_listener():
            """
            Thread function to listen for PostgreSQL notifications.
            """
            # Create a separate native connection for the listener thread

            try:
                listener_conn = pg8000.native.Connection(**listen_params)
                listener_conn.run("LISTEN new_run_event;")
                logger.info("Listening for notifications...")

                while True:
                    # Use select to wait for notifications with timeout
                    # Get the socket from the connection
                    sock = listener_conn._usock

                    if select.select([sock], [], [], 5) == ([], [], []):
                        # Timeout - no notification received
                        continue

                    # Consume the notification by reading from socket
                    # This will trigger notification processing
                    listener_conn._flush()

                    # Check for notifications
                    notifies = listener_conn.notifications
                    while notifies:
                        notify = notifies.pop(0)
                        # notify is a tuple: (backend_pid, channel, payload)
                        notification_queue.put(
                            notify.payload if hasattr(notify, "payload") else notify[2]
                        )

            except Exception as e:
                logger.exception(f"Error in notification listener: {e}")
            finally:
                try:
                    listener_conn.close()
                except Exception:
                    pass

        def notification_handler(payload):
            """
            Handle notification payload.

            Parameters:
            payload (str): Payload containing experiment ID, experiment name, and run UUID.

            Returns:
            None
            """
            try:
                experiment_id, experiment_name, run_uuid, run_name = payload.split(",")
                logger.info(
                    f"Received notification: Experiment ID: {experiment_id}, Experiment Name: {experiment_name}, Run UUID: {run_uuid}, Run Name: {run_name}"
                )
                updated = update_run_log_queue_status(
                    listen_conn, run_uuid, status="DONE"
                )
                if updated == 1:
                    update_tasks_table(
                        update_conn, experiment_name, experiment_id, run_uuid, run_name
                    )
                    delete_run_log_queue_entry(listen_conn, experiment_id, run_uuid)
            except Exception as e:
                logger.exception(f"Error handling notification: {e}")

        # Process existing logs before starting listener
        process_existing_logs(listen_conn, update_conn)

        # Start notification listener thread
        listener_thread = threading.Thread(target=notification_listener, daemon=True)
        listener_thread.start()

        # Main loop to process notifications
        while True:
            try:
                if not notification_queue.empty():
                    payload = notification_queue.get()
                    notification_handler(payload)

                # Health check
                time.sleep(health_check_interval)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                time.sleep(5)

    except pg8000.dbapi.DatabaseError as e:
        logger.exception(f"Database error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during setup: {e}")
    finally:
        if listen_conn:
            try:
                listen_conn.close()
            except Exception:
                pass
        if update_conn:
            try:
                update_conn.close()
            except Exception:
                pass


def process_existing_logs(listen_conn, update_conn):
    """
    Process existing log entries in the database.

    Parameters:
    listen_conn (pg8000.dbapi.Connection): Database connection for listening to new log entries.
    update_conn (pg8000.dbapi.Connection): Database connection for updating log entries.

    Returns:
    None
    """
    try:
        cursor = listen_conn.cursor()
        cursor.execute(
            "SELECT experiment_id, experiment_name, run_uuid, run_name FROM public.run_log_queue"
        )
        rows = cursor.fetchall()
        cursor.close()

        if rows:
            logger.info(f"Processing {len(rows)} existing log entries...")
            for row in rows:
                experiment_id = row[0]
                experiment_name = row[1]
                run_uuid = row[2]
                run_name = row[3]
                logger.info(
                    f"Processing existing log: Experiment ID: {experiment_id}, Experiment Name: {experiment_name}, Run UUID: {run_uuid}, Run Name: {run_name}"
                )
                updated = update_run_log_queue_status(
                    listen_conn, run_uuid, status="DONE"
                )
                if updated == 1:
                    update_tasks_table(
                        update_conn,
                        experiment_name,
                        experiment_id,
                        run_uuid,
                        run_name,
                    )
                    delete_run_log_queue_entry(listen_conn, experiment_id, run_uuid)
            logger.info("Finished processing existing log entries.")
        else:
            logger.info("No existing log entries found.")
    except pg8000.dbapi.DatabaseError as e:
        logger.exception(f"Error processing existing logs: {e}")


def update_tasks_table(update_conn, experiment_name, experiment_id, run_uuid, run_name):
    """
    Update the tasks table in the public schema of the database.

    Parameters:
    update_conn (pg8000.dbapi.Connection): A pg8000 database connection.
    experiment_name (str): The name of the experiment.
    experiment_id (int): The ID of the experiment.
    run_uuid (str): The UUID of the run.
    run_name (str): The name of the run.

    Returns:
    None
    """
    try:
        cursor = update_conn.cursor()
        metrics_value = {f"{run_name}": f"/experiments/{experiment_id}/runs/{run_uuid}"}
        cursor.execute(
            """
            UPDATE public.ft_tunes
            SET metrics =
                CASE
                    WHEN metrics IS NULL OR metrics = '' THEN %s::jsonb::text
                    ELSE
                        (
                            COALESCE(jsonb_path_query_array(metrics::jsonb, '$[*]'),'[]'::jsonb) || jsonb_build_array(%s::jsonb)
                        )::text
                END
            WHERE id = %s
        """,
            (
                json.dumps([metrics_value]),
                json.dumps(metrics_value),
                experiment_name,
            ),
        )
        cursor.close()
    except pg8000.dbapi.DatabaseError as e:
        logger.exception(f"Error updating tasks table: {e}")


def update_run_log_queue_status(listen_conn, run_uuid, status):
    """
    Update the status of a run in the run_log_queue table.

    Parameters:
    listen_conn (pg8000.dbapi.Connection): A database connection.
    run_uuid (str): The unique identifier for the run.
    status (str): The new status to be set for the run.

    Returns:
    int: The number of rows affected by the update, or None if an error occurred.

    Raises:
    Exception: If an unexpected error occurs during the update.
    """
    updated = None
    try:
        cursor = listen_conn.cursor()
        cursor.execute(
            """
            UPDATE public.run_log_queue
            SET status = %s
            WHERE run_uuid = %s AND status = 'PENDING';
        """,
            (status, run_uuid),
        )
        updated = cursor.rowcount
        cursor.close()
    except pg8000.dbapi.DatabaseError as e:
        logger.warning(f"Error updating run_log_queue entry: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error when updating run_log_queue entry: {e}")
    finally:
        return updated


def delete_run_log_queue_entry(listen_conn, experiment_id, run_uuid):
    """
    Delete a specific entry from the run_log_queue table in the public schema of the database.

    Parameters:
    - listen_conn (pg8000.dbapi.Connection): A database connection.
    - experiment_id (int): An integer representing the experiment ID.
    - run_uuid (str): A string representing the run UUID.

    Returns:
    None

    Raises:
    - pg8000.dbapi.DatabaseError: If an error occurs during the deletion process.
    """
    try:
        cursor = listen_conn.cursor()
        cursor.execute(
            """
            DELETE FROM public.run_log_queue
            WHERE experiment_id = %s AND run_uuid = %s;
        """,
            (int(experiment_id), run_uuid),
        )
        cursor.close()
        logger.info(
            f"Deleted entry from run_log_queue for Experiment ID: {experiment_id}, Run UUID: {run_uuid}"
        )
    except pg8000.dbapi.DatabaseError as e:
        logger.exception(f"Error deleting run_log_queue entry: {e}")


def main():
    """
    This function listens to notifications from a database and updates another database accordingly.

    Parameters:
    listen_conn_string (str): The connection string for the database to listen to.
    update_conn_string (str): The connection string for the database to update.

    Returns:
    None
    """
    listen_conn_string = os.getenv(
        "MLFLOW_DATABASE_URI",
        "",
    )
    update_conn_string = os.getenv(
        "DATABASE_URI",
        "",
    )
    listen_to_notifications(listen_conn_string, update_conn_string)


if __name__ == "__main__":
    main()
