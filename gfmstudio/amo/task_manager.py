# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy.orm import Session

from gfmstudio.amo.schemas import OnboardingStatus
from gfmstudio.common.api import crud, utils
from gfmstudio.config import settings
from gfmstudio.inference.v2.models import Task


class AmoTaskManager:
    """Manager class for handling task states."""

    def __init__(self):
        self.task_crud = crud.ItemCrud(model=Task)

    def _check_availability(self, task_id: str, user: str, db: Session = None) -> None:
        """
        Check if task ID is available.

        Args:
            task_id: Task identifier
        """
        db = db or next(utils.get_db())
        task_id = task_id if task_id.startswith("amo-") else f"amo-{task_id}"
        tasks = self.task_crud.get_all(
            db=db,
            filters={"task_id": task_id},
            ignore_user_check=True,
        )
        return tasks[0] if tasks else tasks

    def set_task_status(
        self,
        task_id: str,
        status: OnboardingStatus,
        user: str = settings.DEFAULT_SYSTEM_USER,
        db: Session = None,
    ) -> None:
        """
        Set task status for a model.

        Args:
            model_id: Model identifier
            status: New status to set
        """
        db = db or next(utils.get_db())
        task_id = task_id if task_id.startswith("amo-") else f"amo-{task_id}"

        # Check if exists:
        task = self._check_availability(task_id=task_id, user=user, db=db)
        if task:  # Update the status of existing task
            self.task_crud.update(
                db=db,
                item_id=task.id,
                item={"status": str(status)},
                protected=False,
            )
        else:
            self.task_crud.create(
                db=db,
                item=Task(
                    **{
                        "task_id": task_id,
                        "status": str(status),
                        "created_by": user,
                    }
                ),
                user=user,
            )

    def get_task_status(
        self, task_id: str, user: str, db: Session = None
    ) -> OnboardingStatus:
        """
        Get task status for a model.

        Args:
            model_id: Model identifier

        Returns:
            OnboardingStatus: Current task status

        Raises:
            HTTPException: If task not found
        """
        db = db or next(utils.get_db())
        task_id = task_id if task_id.startswith("amo-") else f"amo-{task_id}"
        task = self._check_availability(task_id=task_id, user=user, db=db)
        if task:
            return task.status


amo_task_manager = AmoTaskManager()
