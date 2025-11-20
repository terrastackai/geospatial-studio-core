## Running Inference Tasks Asynchronously

To run inference tasks asynchronously use `POST /v1/async/inference`. Requests going through `POST /v1/inference` are synchronous.

### Request Flow

#### Sequence Diagram

![Structure](../docs/img/architecture-sequence-async-Inference.png)

#### Initial Inference Request

1. API client makes a request to `POST /v1/async/inference` to run an inference task.
2. The request details are stored in a Database and state marked as `PENDING`.
3. The requests is published to a message broker (Redis).
4. API client gets back a reponse object. The response object includes an `id` and `event_id` among other fields:

    ```json
    {
        "id": "<uuid>",
        "event_id": "<uuid>",
        ...
    }
    ```

#### Background Tasks

After step 3 in the *Initial Inference Request*, the users gets back the response in step 4 but the following steps continue in the background.

5. A celery worker retrieves the inference request object published in step 3 from the message broker.
6. The celery worker makes an inference call to the downstream inference services.
7. The inference services make their predictions and return the response to the worker.
8. The worker gets the reponse, updates the DB records that was created in step 2. with a `COMPLETED`/`ERROR` state and adds the predicted layers.
9. The worker publishes the result from the inference run to the message broker.


#### Event Listeners

After step 4 in the *Initial Inference Request*, a client could opt to use our **(SSE) Server Sent Events** endpoint to listen to changes/updates from the inference tasks.

10. Use the `event_id` from step 4 to register for events like so:

    ```
    GET /v1/async/notify/{event_id}
    ```

    This will send real time messages and tasks status untill the inference task is complete.

11. The SSE endpoint `GET /v1/async/notify/{event_id}` listens to inference messages on the channel associated with the `event_id` in the message broker.
12. The client gets notified every time an event occurs until the inference tasks in completed.
