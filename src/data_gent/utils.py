from uuid import uuid4

sanitized_uuid = lambda: f"{str(uuid4()).replace('-', '_')}"
