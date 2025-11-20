# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

# import logging
# from urllib.parse import urljoin

# import backoff
# import httpx
# from fastapi import HTTPException
# from gfmstudio.config import settings


# logger = logging.getLogger(__name__)


# @backoff.on_exception(backoff.expo, (httpx.HTTPStatusError, httpx.RequestError), max_tries=3)
# async def invoke_inference_gateway(url: str, headers: dict, json_data: dict, verify: bool = False):
#     """Function to invoke inference gateway

#     Parameters
#     ----------
#     url : str
#         URL to the deployed model
#     headers : dict
#         payload headers
#     json_data : dict
#         data for the inference
#     verify : bool, optional
#         Insecure connections, by default False

#     Returns
#     -------
#     Response
#         Response from the model after inference
#     """
#     RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
#     async with httpx.AsyncClient(verify=verify) as client:
#         response = await client.post(url, headers=headers, json=json_data)
#         if response.status_code in RETRYABLE_STATUS_CODES:
#             response.raise_for_status()
#         return response


# async def try_out_model_inference(
#     auth_headers: dict,
#     try_inference_payload: dict,
# ):
#     """Function to try out model inference

#     Parameters
#     ----------
#     auth_headers : dict
#         Authentication headers
#     try_inference_payload : dict
#         Payload for the inference

#     Returns
#     -------
#     dict
#         Response from the model after inference

#     Raises
#     ------
#     HTTPException
#         500: Error occured when attempting to invoke inference gateway
#     """
#     inference_url = urljoin(settings.INFERENCE_GATEWAY_BASE_URL.replace("v1", "v2"), "inference")
#     request_headers = {
#         "accept": "application/json",
#         "Content-Type": "application/json",
#     }
#     if ("Authorization" in auth_headers) or ("X-API-Key" in auth_headers):
#         request_headers.update(auth_headers)
#     else:
#         request_headers["X-Api-Key"] = settings.INFERENCE_GATEWAY_API_KEY

#     try:
#         inference_response = await invoke_inference_gateway(
#             inference_url, request_headers, try_inference_payload
#         )
#     except (httpx.ConnectError, httpx.HTTPStatusError):
#         raise HTTPException(
#             status_code=500,
#             detail={"message": "Error occured when attempting to invoke inference gateway."},
#         )

#     # Check if the status code is not 2XX
#     if not (inference_response.status_code // 100) == 2:
#         logger.error(
#             "Failed invoking inference gateway: ",
#             inference_response.status_code,
#         )
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "message": "Error occured when attempting to invoke inference gateway.",
#                 "error": inference_response.json(),
#             },
#         )
#     return inference_response.json()


# async def deploy_model_to_inference_svc(
#     auth_headers: dict,
#     model_metadata: dict,
#     model_deployment_params: dict,
# ):
#     """Function that tries to deploy a model to the inference service

#     Parameters
#     ----------
#     auth_headers : str
#         Authentication Key
#     model_metadata : dict
#         Dict of model metadata needed for deployment
#     model_deployment_params : dict
#         Parameters of the model

#     Returns
#     -------
#     dict
#         Dictionary with successfully deployed model metadata

#     Raises
#     ------
#     HTTPException
#         500: Error occured when attempting to add model to inference gateway
#     HTTPException
#         500: Error occured when attempting to add model to inference gateway
#     HTTPException
#         500: Model was not successfully added to the inference gateway.
#     HTTPException
#         500: Model was not successfully added to the inference gateway.
#     """
#     # First request to create a model
#     create_model_url = urljoin(settings.INFERENCE_GATEWAY_BASE_URL, "models/")
#     request_headers = {
#         "accept": "application/json",
#         "Content-Type": "application/json",
#     }
#     if ("Authorization" in auth_headers) or ("X-API-Key" in auth_headers):
#         request_headers.update(auth_headers)
#     else:
#         request_headers["X-Api-Key"] = settings.INFERENCE_GATEWAY_API_KEY

#     try:
#         create_model_response = await invoke_inference_gateway(
#             create_model_url, request_headers, model_metadata
#         )
#     except (httpx.ConnectError, httpx.HTTPStatusError):
#         raise HTTPException(
#             status_code=500,
#             detail={"message": "Error occured when attempting to add model to inference gateway."},
#         )

#     # Check if the status code is not 2XX
#     created_model_data = create_model_response.json()
#     if not (create_model_response.status_code // 100) == 2:
#         logger.error(
#             "Failed creating model metatada in inference gateway: ",
#             create_model_response.status_code,
#         )
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "message": "Model was not successfully added to the inference gateway.",
#                 "error": created_model_data,
#             },
#         )

#     # Second request to deploy the model using the obtained UUID
#     inference_model_id = created_model_data["id"]
#     deploy_model_url = urljoin(
#         settings.INFERENCE_GATEWAY_BASE_URL,
#         f"models/{inference_model_id}/deploy",
#     )

#     try:
#         deploy_model_response = await invoke_inference_gateway(
#             deploy_model_url,
#             request_headers,
#             model_deployment_params,
#         )
#     except (httpx.ConnectError, httpx.HTTPStatusError):
#         logger.error(
#             "Failed invoking deploy-model in inference gateway: ",
#             create_model_response.status_code,
#         )
#         raise HTTPException(
#             status_code=500,
#             detail={"message": "Error occured when attempting to add model to inference gateway."},
#         )

#     # Check if the status code is not 2XX
#     deployed_model_data = deploy_model_response.json()
#     if not (create_model_response.status_code // 100) == 2:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "message": "Model was not successfully added to the inference gateway.",
#                 "error": created_model_data,
#             },
#         )

#     return {
#         "message": "Deployment successful started",
#         "details": deployed_model_data,
#     }
