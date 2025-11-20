# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import json
import logging
import os

import asyncpg
import asyncpg.pool
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


async def listen_to_notifications(
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
    listen_pool = None
    update_pool = None

    try:
        listen_pool = await asyncpg.create_pool(
            listen_conn_string,
            min_size=1,
            max_size=4,
            max_inactive_connection_lifetime=60,
        )
        update_pool = await asyncpg.create_pool(
            update_conn_string,
            min_size=1,
            max_size=1,
            max_inactive_connection_lifetime=60,
        )

        async def notification_handler(connection, pid, channel, payload):
            """
            Asynchronous notification handler for a messaging system.

            Parameters:
            ) connection (object): Connection object for the messaging system.
            ) pid (int): Process ID of the notification handler.
            ) channel (str): Channel on which the notification was received.
            ) payload (str): Payload containing experiment ID, experiment name, and run UUID.

            Returns:
            ) None
            """
            experiment_id, experiment_name, run_uuid, run_name = payload.split(",")
            logger.info(
                f"Received notification: Experiment ID: {experiment_id}, Experiment Name: {experiment_name}, Run UUID: {run_uuid}, Run Name: {run_name}"
            )
            updated = await update_run_log_queue_status(
                listen_pool, run_uuid, status="DONE"
            )
            if updated == "UPDATE 1":
                await update_tasks_table(
                    update_pool, experiment_name, experiment_id, run_uuid, run_name
                )
                await delete_run_log_queue_entry(listen_pool, experiment_id, run_uuid)

        while True:
            try:
                async with listen_pool.acquire() as listen_conn:
                    await listen_conn.execute("UNLISTEN *;")
                    await listen_conn.add_listener(
                        "new_run_event", notification_handler
                    )
                    logger.info("Listening for notifications...")
                    await process_existing_logs(listen_pool, update_pool)
                    await listen_conn_pool_health_check(
                        listen_conn, health_check_interval
                    )
                    await asyncio.Future()

            except (asyncpg.PostgresConnectionError, OSError):
                logger.exception("Connection lost in listener. Retrying in 1 minute...")
                await asyncio.sleep(5)
            except Exception:
                logger.exception(
                    "An unexpected error occurred in listener. Retrying in 1 minute..."
                )
                await asyncio.sleep(5)

    except asyncpg.PostgresError:
        logger.exception("Error creating connection pools")
    except Exception:
        logger.exception("An unexpected error occurred during pool creation")
    finally:
        if listen_pool:
            await listen_pool.close()
        if update_pool:
            await update_pool.close()


async def listen_conn_pool_health_check(conn, interval: int):
    """
    Listen to the connection pool health check.

    Parameters:
    conn (object): The connection object.
    interval (int): The time interval in seconds between each health check.

    Returns:
    None
    """
    while not conn.is_closed():
        await asyncio.sleep(interval)


async def process_existing_logs(listen_pool, update_pool):
    """
    Process existing log entries in the database.

    Parameters:
    listen_pool (asyncpg.pool.Pool): Database connection pool for listening to new log entries.
    update_pool (asyncpg.pool.Pool): Database connection pool for updating log entries.

    Returns:
    None
    """
    try:
        async with listen_pool.acquire() as listen_conn:
            rows = await listen_conn.fetch(
                "SELECT experiment_id, experiment_name, run_uuid, run_name FROM public.run_log_queue"
            )
            if rows:
                logger.info(f"Processing {len(rows)} existing log entries...")
                for row in rows:
                    experiment_id = row["experiment_id"]
                    experiment_name = row["experiment_name"]
                    run_uuid = row["run_uuid"]
                    run_name = row["run_name"]
                    logger.info(
                        f"Processing existing log: Experiment ID: {experiment_id}, Experiment Name: {experiment_name}, Run UUID: {run_uuid}, Run Name: {run_name}"
                    )
                    updated = await update_run_log_queue_status(
                        listen_pool, run_uuid, status="DONE"
                    )
                    if updated == "UPDATE 1":
                        await update_tasks_table(
                            update_pool,
                            experiment_name,
                            experiment_id,
                            run_uuid,
                            run_name,
                        )
                        await delete_run_log_queue_entry(
                            listen_pool, experiment_id, run_uuid
                        )
                logger.info("Finished processing existing log entries.")
            else:
                logger.info("No existing log entries found.")
    except asyncpg.PostgresError:
        logger.exception("Error processing existing logs")


async def update_tasks_table(
    update_pool, experiment_name, experiment_id, run_uuid, run_name
):
    """
    Update the tasks table in the public schema of the database.

    Parameters:
    update_pool (asyncpg.pool.Pool): An asyncpg connection pool.
    experiment_name (str): The name of the experiment.
    experiment_id (int): The ID of the experiment.
    run_uuid (str): The UUID of the run.

    Returns:
    None
    """
    try:
        async with update_pool.acquire() as update_conn:
            metrics_value = {
                f"{run_name}": f"/experiments/{experiment_id}/runs/{run_uuid}"
            }
            await update_conn.execute(
                """
                UPDATE public.ft_tunes
                SET metrics =
                    CASE
                        WHEN metrics IS NULL OR metrics = '' THEN $1::jsonb::text
                        ELSE
                            (
                                COALESCE(jsonb_path_query_array(metrics::jsonb, '$[*]'),'[]'::jsonb) || jsonb_build_array($2::jsonb)
                            )::text
                    END
                WHERE id = $3
            """,
                json.dumps([metrics_value]),
                json.dumps(metrics_value),
                experiment_name,
            )
    except asyncpg.PostgresError:
        logger.exception("Error updating tasks table")


async def update_run_log_queue_status(listen_pool, run_uuid, status):
    """
    Update the status of a run in the run_log_queue table.

    Parameters:
    listen_pool (asyncpg.pool.Pool): An asynchronous connection pool to the database.
    run_uuid (str): The unique identifier for the run.
    status (str): The new status to be set for the run.

    Returns:
    int: The number of rows affected by the update, or None if an error occurred.

    Raises:
    Exception: If an unexpected error occurs during the update.
    """
    updated = None
    try:
        async with listen_pool.acquire() as listen_conn:
            updated = await listen_conn.execute(
                """
                UPDATE public.run_log_queue
                SET status = $1
                WHERE run_uuid = $2 AND status = 'PENDING';
            """,
                status,
                run_uuid,
            )
    except asyncpg.PostgresError as e:
        logger.warning(f"Error updating run_log_queue entry: {e}")
    except Exception:
        logger.exception("An unexpected error when updating run_log_queue entry")
    finally:
        return updated


async def delete_run_log_queue_entry(listen_pool, experiment_id, run_uuid):
    """
    Delete a specific entry from the run_log_queue table in the public schema of the database.

    Parameters:
    - listen_pool (asyncpg.pool.Pool): An asynchronous connection pool.
    - experiment_id (int): An integer representing the experiment ID.
    - run_uuid (str): A string representing the run UUID.

    Returns:
    None

    Raises:
    - asyncpg.PostgresError: If an error occurs during the deletion process.
    """
    try:
        async with listen_pool.acquire() as listen_conn:
            await listen_conn.execute(
                """
                DELETE FROM public.run_log_queue
                WHERE experiment_id = $1 AND run_uuid = $2;
            """,
                int(experiment_id),
                run_uuid,
            )
            logger.info(
                f"Deleted entry from run_log_queue for Experiment ID: {experiment_id}, Run UUID: {run_uuid}"
            )
    except asyncpg.PostgresError:
        logger.exception("Error deleting run_log_queue entry")


async def main():
    """
    This function is an asynchronous function that listens to notifications from a database and updates another database accordingly.

    Parameters:
    listen_conn_string (str): The connection string for the database to listen to.
    update_conn_string (str): The connection string for the database to update.

    Returns:
    None
    """
    listen_conn_string = os.getenv(
        "MLFLOW_DATABASE_URI",
        "",
    ).replace("postgresql+pg8000://", "postgresql://")
    update_conn_string = os.getenv(
        "DATABASE_URI",
        "",
    ).replace("postgresql+pg8000://", "postgresql://")
    await listen_to_notifications(listen_conn_string, update_conn_string)


if __name__ == "__main__":
    asyncio.run(main())
