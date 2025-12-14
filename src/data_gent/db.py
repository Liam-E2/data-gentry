from datetime import datetime

from sqlalchemy import Sequence, Text, TIMESTAMP, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class BaseTable(DeclarativeBase):
    pass


class Documents(BaseTable):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column("id", Sequence("document_id_sequence"), primary_key=True)
    content: Mapped[str] = mapped_column("content", Text())
    created_at: Mapped[datetime] = mapped_column("created_at", TIMESTAMP(), server_default=func.current_timestamp())
