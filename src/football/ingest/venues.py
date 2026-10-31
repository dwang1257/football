from football.ingest.base import IngestFn
from football.validate.base import CheckFn

SOURCES: dict[str, IngestFn] = {}
CHECKS: dict[str, CheckFn] = {}
