FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 caster
RUN mkdir -p /sandbox && chown caster:caster /sandbox

WORKDIR /workspace

COPY miner-sdk ./miner-sdk
COPY miner ./miner

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir ./miner-sdk ./miner

USER caster

ENTRYPOINT ["caster-miner"]
CMD ["--serve"]
