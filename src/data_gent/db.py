from sqlalchemy import Sequence, Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class BaseTable(DeclarativeBase):
    pass


class Documents(BaseTable):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column("id", Sequence("document_id_sequence"), primary_key=True)
    content: Mapped[str] = mapped_column("content", Text())

