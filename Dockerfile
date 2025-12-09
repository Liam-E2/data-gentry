FROM python:3.14-slim as base

RUN apt update -y && apt install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /data-gent

COPY pyproject.toml .
RUN uv sync

COPY . .
RUN uv run pytest -ra
RUN uv build

FROM python:3.14-slim as final

WORKDIR /data-gent
COPY --from=base /data-gent/dist ./dist

RUN pip install ./dist/*.whl

CMD ["python3"]