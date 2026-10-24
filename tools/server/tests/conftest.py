import pytest
from utils import *


<<<<<<< HEAD
=======
@pytest.fixture(scope="session", autouse=True)
def configure_worker_port(request):
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    worker_num = 0 if worker_id == "master" else int(worker_id[2:])
    port_base = int(os.environ.get("LLAMA_SERVER_TEST_PORT_BASE", "8080"))
    os.environ["PORT"] = str(port_base + worker_num * 10)


>>>>>>> ggml-org/master
# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
