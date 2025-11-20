# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from fastapi import APIRouter, Depends, HTTPException, Path, Request

from ..auth.authorizer import auth_handler
from ..config import settings
from .schemas import (
    CommentRequestSchema,
    CommentResponseSchema,
    IssueListResponse,
    IssueRequestSchema,
    IssueResponseSchema,
)
from .services import fetch_jira_issues, post_jira_comment, post_jira_issue

jira_router = APIRouter(
    tags=["Studio / Feedback"],
    dependencies=[Depends(auth_handler)],
)


@jira_router.get(
    "/issue",
    response_model=IssueListResponse,
)
async def get_issues(
    request: Request,
    auth=Depends(auth_handler),
):
    user = auth[0]
    jira_api_key = settings.JIRA_API_KEY
    jira_api_url = settings.JIRA_API_URI
    project = settings.JIRA_API_PROJECT
    fields = settings.JIRA_API_FIELDS
    issues = fetch_jira_issues(
        jira_api_url=jira_api_url,
        user=user,
        project=project,
        fields=fields,
        jira_api_key=jira_api_key,
    )
    return issues


@jira_router.post(
    "/issue/{issue_key}/comment",
    response_model=CommentResponseSchema,
    status_code=201,
)
async def post_issue_comment(
    request: Request,
    item: CommentRequestSchema,
    issue_key: str = Path(description="Key of an existing issue, e.g. WGS-xxxx"),
    auth=Depends(auth_handler),
):
    """Add comments to an existing issue. You need to provide `issue_key` as a path parameter

    **Example Schema**
    ```
    {
        "body": "This is my comment..."
    }
    ```

    """

    user = auth[0]

    # Do not save comment with empty details:
    if not item.body:
        raise HTTPException(
            status_code=412, detail="Comment with empty body not processed."
        )

    jira_api_key = settings.JIRA_API_KEY
    jira_api_url = settings.JIRA_API_URI
    issue = post_jira_comment(
        jira_api_url=jira_api_url,
        user=user,
        jira_api_key=jira_api_key,
        jira_issue_key=issue_key,
        item=item,
    )
    return issue


@jira_router.post(
    "/issue",
    response_model=IssueResponseSchema,
    status_code=201,
)
async def post_issue(
    request: Request,
    item: IssueRequestSchema,
    auth=Depends(auth_handler),
):
    """Report issues and feedback using this api.

    The `issuetype` entry has a few accepted types as discussed below…

    **Example Schemas**

    1. **Task**: Task issue type is used to track a piece of work that needs to be completed within a project, but
      doesn't necessarily correspond to a new feature or bug fix. Tasks are typically smaller pieces of work that
      contribute to the overall progress of a project, such as documentation, testing, or code refactoring.
        ```
        {
            "issuetype": "Task",
            "summary": "Summary of the task...",
            "description": "Detailed description of the task..."
        }
        ```

    2. **Bug**: The `Bug` issue type is used to track defects or problems with the software or system. A bug is a coding
      error, a mistake or a flaw in the system that causes it to behave unexpectedly or not as intended. Bugs can be
      discovered during testing, production, or even after the software has been released. Once a bug issue has been
        created, it can be assigned to a developer or team member responsible for fixing it. As the bug is worked on,
          updates can be added to the issue to track progress, such as when a fix is tested or deployed.
        ```
        {
            "issuetype": "Bug",
            "summary": "Summary of the bug...",
            "description": "Detailed description of the bug..."
        }
        ```

    3. **Feature**: New Feature issue type is used to track the development of new functionality or features that are
      being added to a software system or application. It is one of the default issue types available and is commonly
      used in software development projects to manage the development of new features or enhancements to existing
        functionality.
        ```
        {
            "issuetype": "Feature",
            "summary": "Summary of the feature...",
            "description": "Detailed description of the feature..."
        }
        ```

    4. **Story**: A Story issue type is used to describe a small, self-contained unit of work that contributes to the
      completion of an Epic or project. Stories are used to track and manage the implementation of specific features
        or requirements.
        ```
        {
            "issuetype": "Story",
            "summary": "Summary of the story...",
            "description": "Detailed description of the story..."
        }
        ```

    5. **Incident**: Reporting an incident or IT service outage.
        ```
        {
            "issuetype": "Incident",
            "summary": "Summary of the incident...",
            "description": "Detailed description of the incident..."
        }
        ```

    6. **Risk**: Tracks potential problems or uncertainties that might impact your project, allowing you to
      differentiate them from regular tasks, stories, or bugs; essentially, it's a way to categorize and manage
        potential issues that haven't yet occurred within your project using your issue tracking system.
        ```
        {
            "issuetype": "Risk",
            "summary": "Summary of the risk...",
            "description": "Detailed description of the risk..."
        }
        ```

    7. **Change Request**: The change issue type is used to represent a significant change to an existing system or
      process. It is typically used in IT service management to track change requests and approvals
        ```
        {
            "issuetype": "Change Request",
            "summary": "Summary of the change request...",
            "description": "Detailed description of the change request..."
        }
        ```

    8. **Service Ticket**: This issue type is used to represent a request for technical support from users or
      customers. It can be used to track and manage support tickets, and to ensure that support requests are responded
        to in a timely manner. e.g demo requests
        ```
        {
            "issuetype": "Service Ticket",
            "summary": "Summary of the service ticket...",
            "description": "Detailed description of the service ticket..."
        }
        ```
    """
    user = auth[0]

    # Do not save issues with empty details:
    if not item.description:
        raise HTTPException(
            status_code=412, detail="Issue with empty description not processed."
        )

    if not item.summary:
        raise HTTPException(
            status_code=412, detail="Issue with empty summary not processed."
        )

    jira_issue_types_parent = settings.JIRA_ISSUE_TYPES_PARENTS
    if not item.issuetype or item.issuetype not in jira_issue_types_parent:
        raise HTTPException(
            status_code=412, detail="Issue with empty/invalid issuetype not processed."
        )

    jira_api_key = settings.JIRA_API_KEY
    jira_api_url = settings.JIRA_API_URI
    project = settings.JIRA_API_PROJECT
    parent_key = jira_issue_types_parent[item.issuetype]
    issue = post_jira_issue(
        jira_api_url=jira_api_url,
        user=user,
        jira_api_key=jira_api_key,
        item=item,
        parent_key=parent_key,
        project=project,
    )
    return issue
