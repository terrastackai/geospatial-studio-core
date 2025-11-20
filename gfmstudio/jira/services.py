# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import requests
from fastapi import HTTPException

from .schemas import CommentResponseSchema, IssueListResponse, IssueResponseSchema


def fetch_jira_issues(jira_api_url, user, project, fields, jira_api_key):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {jira_api_key}"}
    url = f"{jira_api_url}/search?jql=project={project} AND labels='{user}'&fields={fields}"

    response = requests.request(
        "GET",
        url,
        headers=headers,
    )

    if not response.status_code == 200:
        raise HTTPException(status_code=500, detail="Error loading issues!")

    response_data = IssueListResponse(results=response.json()["issues"])
    return response_data


def post_jira_comment(jira_api_url, user, jira_api_key, jira_issue_key, item):
    headers_issue = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jira_api_key}",
    }
    url_issue = f"{jira_api_url}/issue/{jira_issue_key}"
    response_issue = requests.request(
        "GET",
        url_issue,
        headers=headers_issue,
    )

    if response_issue.status_code != 200 or user.lower() not in map(
        str.lower, response_issue.json()["fields"]["labels"]
    ):
        raise HTTPException(status_code=500, detail="Error updating comment!")

    headers_comment = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jira_api_key}",
        "Content-Type": "application/json",
    }
    url_comment = f"{jira_api_url}/issue/{jira_issue_key}/comment"

    data = {"body": item.body}

    response_comment = requests.post(url_comment, headers=headers_comment, json=data)
    if not response_comment.status_code == 201:
        raise HTTPException(status_code=500, detail="Error loading issues!")

    response_data = CommentResponseSchema(**response_comment.json())
    return response_data


def post_jira_issue(jira_api_url, user, project, jira_api_key, item, parent_key):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jira_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{jira_api_url}/issue"

    data = {
        "fields": {
            "project": {"key": project},
            "summary": item.summary,
            "description": item.description,
            "issuetype": {"name": item.issuetype},
            "customfield_10100": parent_key,
            "labels": [user],
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if not response.status_code == 201:
        raise HTTPException(status_code=500, detail="Error creating issues!")

    jira_issue_key = response.json()["key"]
    headers_issue = {
        "Accept": "application/json",
        "Authorization": f"Bearer {jira_api_key}",
    }
    url_issue = f"{jira_api_url}/issue/{jira_issue_key}"
    response_issue = requests.request(
        "GET",
        url_issue,
        headers=headers_issue,
    )

    if not response_issue.status_code == 200:
        raise HTTPException(status_code=500, detail="Error updating comment!")

    response_data = IssueResponseSchema(**response_issue.json())
    return response_data
