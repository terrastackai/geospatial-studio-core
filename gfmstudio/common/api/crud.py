# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime, timezone
from typing import List, Optional, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import or_, union_all
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from gfmstudio.config import settings

ModelType = TypeVar("ModelType", bound=BaseModel)


class ItemCrud:
    """Mixin class providing CRUD operations for a model."""

    def __init__(self, model: ModelType, custom_filter=None):
        self.model = model
        self.custom_filter = custom_filter

    def _custom_filter(self, query, filters):
        return self.custom_filter(query, filters) or query.filter()

    def _get_user_filter(self, user: str, shared: bool = False):
        user_filter = None
        if user:
            if shared and getattr(self.model, "sharable"):
                user_filter = or_(
                    self.model.created_by.in_(
                        [str(user), settings.DEFAULT_SYSTEM_USER]
                    ),
                    self.model.sharable.is_(True),
                )
            else:
                user_filter = self.model.created_by.in_(
                    [str(user), settings.DEFAULT_SYSTEM_USER]
                )
        else:
            user_filter = self.model.created_by.in_([settings.DEFAULT_SYSTEM_USER])

        return user_filter

    def get_all(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 25,
        user: str = None,
        shared: bool = False,
        filters: dict[str:str] = None,
        filter_expr=None,
        search: dict[str:str] = None,
        ignore_user_check: bool = False,
        total_count: int = False,
    ) -> List[ModelType]:
        """Get all items.

        Parameters
        ----------
        db : Session
            The database session.
        skip : int, optional
            The number of items to skip., by default 0
        limit : int, optional
            The maximum number of items to retrieve., by default 100
        ignore_user_check : bool, optional
            Ignore the default user filters
        total_count : bool, optional
            Wheather to include total count in response

        Returns
        -------
        List[ModelType]
            A list of items.

        """
        query = db.query(self.model).filter(
            or_(
                self.model.deleted.is_(False),
                self.model.deleted.is_(None),
            )
        )

        if not ignore_user_check:
            user_filter = self._get_user_filter(user=user, shared=shared)
            if user_filter is not None:
                query = query.filter(user_filter)

        if self.custom_filter:
            query = self._custom_filter(query=query, filters=filters)
        elif filters:
            query = query.filter_by(**filters)

        if filter_expr is not None:
            query = query.filter(filter_expr)

        if search:
            search_filters = [
                getattr(self.model, column_name, None).ilike(f"%{search_value}%")
                for column_name, search_value in search.items()
                if getattr(self.model, column_name, None)
            ]
            for search_filter in search_filters:
                query = query.filter(search_filter)

        if total_count:
            total_available_records = query.count()
            return (
                total_available_records,
                query.order_by(self.model.created_at.desc())
                .offset(skip)
                .limit(limit)
                .all(),
            )

        return (
            query.order_by(self.model.created_at.desc()).offset(skip).limit(limit).all()
        )

    def get_by_id(
        self,
        db: Session,
        item_id: str,
        user: str = None,
        shared: bool = False,
    ) -> Optional[ModelType]:
        """Get an item by ID.

        Parameters
        ----------
        db : Session
            The database session.
        item_id : str
            The ID of the item.
        user: str, Optional
            The logged in user, by default None
        shared: bool, Optional
            Whether the item is shared accross users, by default False

        Returns
        -------
        Optional[ModelType]
            The item if found, else None.

        """
        query = db.query(self.model).filter(
            self.model.id == item_id,
            or_(
                self.model.deleted.is_(False),
                self.model.deleted.is_(None),
            ),
        )
        if not user:
            return query.first()

        user_filter = self._get_user_filter(user=user, shared=shared)
        if user_filter is not None:
            query = query.filter(user_filter)

        return query.first()

    def create(
        self,
        db: Session,
        item: ModelType,
        user: str = None,
    ) -> ModelType:
        """Create a new item.

        Parameters
        ----------
        db : Session
             The database session.
        item : ModelType
            The item to create.
        user: str, Optional
            The logged in user, by default None

        Returns
        -------
        ModelType: The created item.

        """
        user = user or settings.DEFAULT_SYSTEM_USER
        user_fields = {
            "created_by": user,
            "updated_by": user,
        }
        try:
            db_item = (
                self.model(**user_fields, **item.model_dump())
                if hasattr(item, "dict")
                else item
            )
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
        except IntegrityError as e:
            raise HTTPException(
                status_code=409, detail=str(e.args[0]).split("DETAIL: ")[1]
            )

        return db_item

    def bulk_create(
        self,
        db: Session,
        items: List[ModelType],
        user: str = None,
    ) -> ModelType:
        """Create new items in bulk.

        Parameters
        ----------
        db : Session
             The database session.
        items : List[ModelType]
            The items to create.
        user: str, Optional
            The logged in user, by default None

        Returns
        -------
        ModelType: The created item.

        """
        user = user or settings.DEFAULT_SYSTEM_USER
        user_fields = {
            "created_by": user,
            "updated_by": user,
        }
        try:
            db_items = [self.model(**user_fields, **item.dict()) for item in items]
            db.bulk_save_objects(db_items)
            db.commit()
        except IntegrityError as e:
            raise HTTPException(
                status_code=409, detail=str(e.args[0]).split("DETAIL: ")[1]
            )

        return db_items

    def update(
        self,
        db: Session,
        item_id: str,
        item: ModelType,
        user: str = None,
        protected: bool = True,
    ) -> ModelType:
        """Update an item.

        Parameters
        ----------
        db : Session
            The database session.
        item_id : str
            The id of item to update.
        item : ModelType
            The item to update.
        user: str, Optional
            The logged in user, by default None
        protected: boolean
            Whether record can only be updated by owner.

        Returns
        -------
        ModelType: The updated item.

        Raises
        -------
        HTTPException
            If the item is not found.

        """
        update_user = user or settings.DEFAULT_SYSTEM_USER
        user_fields = {"updated_by": update_user}
        db_item = self.get_by_id(db, item_id, user=user)
        if not db_item:
            raise HTTPException(
                status_code=404, detail="Item not found or Missing permissions to edit."
            )

        if protected and db_item.created_by != user:
            raise HTTPException(
                status_code=404, detail="Missing permissions to edit this record."
            )

        if isinstance(item, dict):
            item.update(user_fields)
            for field, value in item.items():
                setattr(db_item, field, value)
                # After modifying your JSONB field, Force update
                if isinstance(value, dict) or isinstance(value, list):
                    flag_modified(db_item, field)
        else:
            for field, value in item.dict().items():
                if value:
                    setattr(db_item, field, value)
                    # After modifying your JSONB field, Force update
                    if isinstance(value, dict) or isinstance(value, list):
                        flag_modified(db_item, field)
        setattr(db_item, "updated_by", user)
        setattr(db_item, "updated_at", str(datetime.now(timezone.utc)))

        db.commit()
        db.refresh(db_item)
        return db_item

    def delete(
        self,
        db: Session,
        item_id: str,
        user: str = None,
        protected: bool = True,
    ) -> None:
        """Delete an item.

        Parameters
        ----------
        db : Session
             The database session.
        item_id : str
            The ID of the item.
        user: str, Optional
            The logged in user, by default None
        protected: boolean
            Whether record can only be updated by owner.

        Raises
        ------
        HTTPException
            If the item is not found.

        """
        db_item = self.get_by_id(db, item_id, user=user)
        if not db_item:
            raise HTTPException(
                status_code=404,
                detail="Item not found or Missing permissions to delete.",
            )

        if protected and db_item.created_by != user:
            raise HTTPException(
                status_code=404, detail="Missing permissions to delete this record."
            )

        db.delete(db_item)
        db.commit()

    def soft_delete(
        self,
        db: Session,
        item_id: str,
        user: str = None,
        protected: bool = True,
    ) -> None:
        """Soft delete an item.

        Parameters
        ----------
        db : Session
             The database session.
        item_id : str
            The ID of the item.
        user: str, Optional
            The logged in user, by default None
        protected: boolean
            Whether record can only be updated by owner.

        Raises
        ------
        HTTPException
            If the item is not found.

        """
        db_item = self.get_by_id(db, item_id, user=user)
        if not db_item:
            raise HTTPException(
                status_code=404,
                detail="Item not found or Missing permissions to delete.",
            )

        if protected and db_item.created_by != user:
            raise HTTPException(
                status_code=404, detail="Missing permissions to delete this record."
            )

        if db_item.created_by.lower() == user.lower():
            setattr(db_item, "updated_by", user)
            setattr(db_item, "deleted", True)
            setattr(db_item, "active", False)
            db.commit()
            db.refresh(db_item)
        else:
            raise HTTPException(
                status_code=403,
                detail=f"You do not have permission to delete record created by {db_item.created_by}",
            )

    def union_all(
        self,
        db: Session,
        union_config: dict = None,
        skip: int = 0,
        limit: int = 25,
        user: str = None,
        shared: bool = False,
        filters: dict[str:str] = None,
        filter_expr=None,
        search: dict[str:str] = None,
        ignore_user_check: bool = False,
        total_count: int = False,
    ) -> List[ModelType]:
        """Get all items.

        Parameters
        ----------
        db : Session
            The database session.
        union_config: Dict
            A dict of mapped columns for union
        skip : int, optional
            The number of items to skip., by default 0
        limit : int, optional
            The maximum number of items to retrieve., by default 100
        ignore_user_check : bool, optional
            Ignore the default user filters
        total_count : bool, optional
            Wheather to include total count in response

        Returns
        -------
        List[ModelType]
            A list of items.

        """
        union_queries = []
        for union_config_element in union_config:
            model = union_config_element["model"]
            self.model = model
            union_column = union_config_element["columns"]
            query = db.query(*union_column).filter(
                or_(
                    self.model.deleted.is_(False),
                    self.model.deleted.is_(None),
                )
            )

            if not ignore_user_check:
                user_filter = self._get_user_filter(user=user, shared=shared)
                if user_filter is not None:
                    query = query.filter(user_filter)

            if self.custom_filter:
                query = self._custom_filter(query=query, filters=filters)
            elif filters:
                query = query.filter_by(**filters)

            if union_config_element["filter_expr"] is not None:
                query = query.filter(union_config_element["filter_expr"])

            if search:
                search_filters = [
                    getattr(self.model, column_name, None).ilike(f"%{search_value}%")
                    for column_name, search_value in search.items()
                    if getattr(self.model, column_name, None)
                ]
                for search_filter in search_filters:
                    query = query.filter(search_filter)

            union_queries.append(query)

        union_stmt = union_all(*union_queries).alias("union_result")

        if total_count:
            total_available_records = db.query(union_stmt).count()
            return (
                total_available_records,
                db.query(union_stmt)
                .order_by(union_stmt.c.created_at_.desc())
                .offset(skip)
                .limit(limit)
                .all(),
            )

        return (
            db.query(union_stmt)
            .order_by(union_stmt.c.created_at_.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
