# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import signal
import time
from typing import Optional
from urllib.parse import urlparse

import pg8000.native

try:
    from eventing import (
        create_task_ready_event,
        publish_event,
    )
except ImportError:
    from cloudevents_schema import create_task_ready_event
    from knative_events import publish_event

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DatabaseEventPublisher:
    """
    Listens to PostgreSQL NOTIFY events and publishes CloudEvents.
    """

    def __init__(
        self, db_uri: str, channel: str = "task_events", table_name: str = "inf_task"
    ):
        """
        Initialize the database event publisher.

        Args:
            db_uri: PostgreSQL connection URI
            channel: PostgreSQL NOTIFY channel name
            table_name: Name of the task table
        """
        self.db_uri = db_uri
        self.channel = channel
        self.table_name = table_name
        self.conn: Optional[pg8000.native.Connection] = None
        self.running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def connect(self):
        """Establish database connection."""
        try:
            # Parse the database URI
            parsed = urlparse(self.db_uri)

            # Extract connection parameters
            self.conn = pg8000.native.Connection(
                user=parsed.username,
                password=parsed.password,
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/"),
            )

            logger.info(f"Connected to database, listening on channel: {self.channel}")
        except Exception as ex:
            logger.error(f"Failed to connect to database: {ex}")
            raise

    def setup_trigger(self):
        """
        Create PostgreSQL trigger function and trigger for task events.
        This should be run once during setup.
        """
        trigger_function = f"""
        CREATE OR REPLACE FUNCTION notify_task_event()
        RETURNS trigger AS $$
        DECLARE
            payload json;
            step_data json;
        BEGIN
            -- Only notify for READY status changes
            IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
                -- Find the READY step in pipeline_steps
                SELECT json_array_elements(NEW.pipeline_steps::json)
                INTO step_data
                WHERE (json_array_elements(NEW.pipeline_steps::json)->>'status') = 'READY'
                LIMIT 1;

                IF step_data IS NOT NULL THEN
                    payload = json_build_object(
                        'operation', TG_OP,
                        'task_id', NEW.task_id,
                        'inference_id', NEW.inference_id,
                        'inference_folder', NEW.inference_folder,
                        'priority', NEW.priority,
                        'process_id', step_data->>'process_id',
                        'step_number', step_data->>'step_number',
                        'pipeline_steps', NEW.pipeline_steps,
                        'timestamp', EXTRACT(EPOCH FROM NOW())
                    );
                    
                    PERFORM pg_notify('{self.channel}', payload::text);
                END IF;
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """

        trigger = f"""
        DROP TRIGGER IF EXISTS task_event_trigger ON {self.table_name};
        CREATE TRIGGER task_event_trigger
        AFTER INSERT OR UPDATE ON {self.table_name}
        FOR EACH ROW
        EXECUTE FUNCTION notify_task_event();
        """

        if not self.conn:
            raise RuntimeError("Database connection not established")

        try:
            self.conn.run(trigger_function)
            self.conn.run(trigger)
            logger.info(f"Successfully created trigger on {self.table_name}")
        except Exception as ex:
            logger.error(f"Failed to create trigger: {ex}")
            raise

    def listen(self):
        """Start listening for database notifications."""
        if not self.conn:
            self.connect()

        if not self.conn:
            raise RuntimeError("Failed to establish database connection")

        try:
            self.conn.run(f"LISTEN {self.channel};")

            logger.info(f"Listening for notifications on channel: {self.channel}")
            self.running = True

            while self.running:
                # Check for notifications (notifications is a deque attribute in pg8000)
                if self.conn.notifications:
                    while self.conn.notifications:
                        notify = self.conn.notifications.popleft()
                        self._handle_notification(notify)
                else:
                    # Sleep briefly to avoid busy waiting
                    time.sleep(0.1)

        except Exception as ex:
            logger.error(f"Error in listen loop: {ex}")
            raise
        finally:
            self.close()

    def _handle_notification(self, notify):
        """
        Handle a database notification by publishing a CloudEvent.

        Args:
            notify: pg8000 notification tuple (channel, payload, pid)
        """
        try:
            # pg8000 returns notifications as tuples: (channel, payload, pid)
            channel, payload_str, pid = notify
            payload = json.loads(payload_str)
            logger.info(f"Received notification: {payload.get('task_id')}")

            # Create CloudEvent
            event = create_task_ready_event(
                task_id=payload["task_id"],
                inference_id=payload["inference_id"],
                process_id=payload["process_id"],
                step_number=int(payload["step_number"]),
                inference_folder=payload["inference_folder"],
                pipeline_steps=payload["pipeline_steps"],
                priority=int(payload.get("priority", 5)),
                metadata={
                    "operation": payload["operation"],
                    "db_timestamp": payload["timestamp"],
                },
            )

            # Publish event
            success = publish_event(event)
            if success:
                logger.info(
                    f"Published event for task: {payload['task_id']}, "
                    f"process: {payload['process_id']}"
                )
            else:
                logger.error(f"Failed to publish event for task: {payload['task_id']}")

        except Exception as ex:
            logger.error(f"Error handling notification: {ex}", exc_info=True)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main entry point for the database event publisher."""
    db_uri = os.getenv("orchestrate_db_uri", "")
    channel = os.getenv("DB_NOTIFY_CHANNEL", "task_events")
    table_name = os.getenv("inference_task_table", "inf_task")
    setup_trigger = os.getenv("SETUP_TRIGGER", "false").lower() == "true"

    publisher = DatabaseEventPublisher(
        db_uri=db_uri, channel=channel, table_name=table_name
    )

    publisher.connect()

    if setup_trigger:
        logger.info("Setting up database trigger...")
        publisher.setup_trigger()
        logger.info("Trigger setup complete. Exiting...")
        publisher.close()
        return

    logger.info("Starting event publisher...")
    publisher.listen()


if __name__ == "__main__":
    main()

# Made with Bob
