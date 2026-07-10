import pytest

from agentobs import init_tracing
from agentobs.tracing import get_memory_exporter

# The global OTel TracerProvider can only be installed once per process,
# so initialize the in-memory exporter for the whole session and clear
# captured spans between tests.
init_tracing("agentobs-tests", exporter="memory")


@pytest.fixture
def exporter():
    exp = get_memory_exporter()
    exp.clear()
    return exp
