"""Verify notebook preservation, server selection, and the local launch endpoint."""

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from scripts.notebooks import NotebookLauncher
from scripts.slides import PreviewHandler


@pytest.fixture
def course(tmp_path):
    deck = tmp_path / "slides" / "01-tokenization"
    deck.mkdir(parents=True)
    (deck / "practice.ipynb").write_text(json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []
    }))
    (deck / "assets").mkdir()
    (deck / "assets" / "data.json").write_text('[1, 2, 3]')
    return tmp_path


def test_working_copy_preserves_answers_and_assets(course):
    launcher = NotebookLauncher(course, runtime_dirs=[])
    notebook = launcher.prepare_notebook("01-tokenization")
    notebook.write_text("Student's saved answers")
    assets = notebook.parent / "assets" / "data.json"
    assets.write_text("Student's data")
    launcher.prepare_notebook("01-tokenization")
    assert notebook.read_text() == "Student's saved answers"
    assert assets.read_text() == "Student's data"
    assert json.loads((course / "slides/01-tokenization/practice.ipynb").read_text())["cells"] == []


def test_named_lecture_notebook_preserves_student_work(course):
    deck = course / "slides/lecture-01"
    deck.mkdir()
    filename = "lecture-01-exercise.ipynb"
    (deck / "lecture.json").write_text(json.dumps({"notebook": filename}))
    (deck / filename).write_text((course / "slides/01-tokenization/practice.ipynb").read_text())
    launcher = NotebookLauncher(course)
    notebook = launcher.prepare_notebook("lecture-01")
    assert notebook == course / "workspace/slides/lecture-01" / filename
    assert json.loads(notebook.read_text())["metadata"]["kernelspec"]["name"] == "python3"
    notebook.write_text("Student's saved answers")
    assert launcher.prepare_notebook("lecture-01").read_text() == "Student's saved answers"


def test_additional_notebook_gets_separate_copy_and_new_assets(course):
    deck = course / "slides/01-tokenization"
    filename = "extended.ipynb"
    (deck / "lecture.json").write_text(json.dumps({"additional_notebooks": [filename]}))
    (deck / filename).write_text((deck / "practice.ipynb").read_text())
    launcher = NotebookLauncher(course)
    primary = launcher.prepare_notebook("01-tokenization")
    primary.write_text("Classroom answers")
    personal_assets = primary.parent / "assets"
    (personal_assets / "data.json").write_text("Student data")
    (deck / "assets/new.json").write_text('[4, 5]')
    extended = launcher.prepare_notebook("01-tokenization", filename)
    assert extended.name == filename
    assert primary.read_text() == "Classroom answers"
    assert (personal_assets / "data.json").read_text() == "Student data"
    assert (personal_assets / "new.json").read_text() == '[4, 5]'
    extended.write_text("Extended answers")
    assert launcher.prepare_notebook("01-tokenization", filename).read_text() == "Extended answers"


@pytest.mark.parametrize("filename", ["private.ipynb", "../private.ipynb", "/tmp/private.ipynb", [], 1])
def test_additional_notebook_must_be_listed_and_inside_lecture(course, filename):
    with pytest.raises(ValueError):
        NotebookLauncher(course).prepare_notebook("01-tokenization", filename)
    assert not (course / "workspace").exists()


@pytest.mark.parametrize("filename", ["../private.ipynb", "/tmp/private.ipynb", "nested/test.ipynb", "test.txt", None])
def test_notebook_metadata_cannot_choose_another_path(course, filename):
    (course / "slides/01-tokenization/lecture.json").write_text(json.dumps({"notebook": filename}))
    with pytest.raises(ValueError):
        NotebookLauncher(course).prepare_notebook("01-tokenization")
    assert not (course / "workspace").exists()


@pytest.mark.parametrize("lecture", [None, {}, "../../.env", "01-tokenization/../../", "template", "99-missing"])
def test_invalid_notebook_requests_do_not_write(course, lecture):
    launcher = NotebookLauncher(course)
    with pytest.raises(ValueError):
        launcher.prepare_notebook(lecture)
    assert not (course / "workspace").exists()


def test_workspace_symlink_cannot_write_outside_repository(course, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (course / "workspace").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        NotebookLauncher(course).prepare_notebook("01-tokenization")
    assert list(outside.iterdir()) == []


class Response(io.BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(json.dumps(value).encode())


def mock_responses(launcher, monkeypatch, executable):
    def request(server, route):
        if route == "api/kernelspecs":
            return Response({"kernelspecs": {"python3": {"spec": {"argv": [str(executable)]}}}})
        if route.startswith("api/contents/"):
            return Response({"type": "notebook"})
        assert route == "lab"
        return Response({})
    monkeypatch.setattr(launcher, "request", request)


def test_reuses_only_server_with_access_and_matching_environment(course, monkeypatch):
    python = course / ".venv/bin/python3"
    launcher = NotebookLauncher(course, python=python)
    notebook = launcher.prepare_notebook("01-tokenization")
    mock_responses(launcher, monkeypatch, python)
    assert launcher.compatible({"root_dir": str(course.parent)}, notebook)
    assert not launcher.compatible({"root_dir": str(course / "slides")}, notebook)
    mock_responses(launcher, monkeypatch, course / "another-env/bin/python3")
    assert not launcher.compatible({"root_dir": str(course)}, notebook)


def test_native_python_kernel_uses_existing_server_environment(course, monkeypatch):
    import scripts.notebooks as module
    from types import SimpleNamespace

    python = course / ".venv/bin/python3"
    launcher = NotebookLauncher(course, python=python)
    notebook = launcher.prepare_notebook("01-tokenization")
    mock_responses(launcher, monkeypatch, "python")
    monkeypatch.setattr(module.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: [str(python), "-m", "jupyterlab"]))
    assert launcher.compatible({"root_dir": str(course), "pid": 123}, notebook)
    monkeypatch.setattr(module.psutil, "Process", lambda pid: SimpleNamespace(cmdline=lambda: ["/another/env/bin/python3", "-m", "jupyterlab"]))
    assert not launcher.compatible({"root_dir": str(course), "pid": 123}, notebook)
    mock_responses(launcher, monkeypatch, "python")
    assert not launcher.compatible({"root_dir": str(course)}, notebook)


@pytest.mark.parametrize("url", ["http://example.com/", "file:///tmp/", "http://user:secret@localhost:8888/"])
def test_server_probes_stay_local(course, url):
    with pytest.raises(ValueError):
        NotebookLauncher(course).server_base({"url": url})


def test_concurrent_clicks_start_one_server_and_reuse_it(course, monkeypatch):
    launcher = NotebookLauncher(course, runtime_dirs=[])
    servers = []
    starts = []
    server = {"url": "http://127.0.0.1:8888/prefix/", "root_dir": str(course), "token": "test token & value"}

    def start(notebook):
        starts.append(notebook)
        servers.append(server)
        return server

    monkeypatch.setattr(launcher, "registered_servers", lambda: iter(servers))
    monkeypatch.setattr(launcher, "compatible", lambda item, notebook: True)
    monkeypatch.setattr(launcher, "start_server", start)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(launcher.open, ["01-tokenization"] * 3))
    assert len(starts) == 1
    assert sorted(result["reused"] for result in results) == [False, True, True]
    assert len({urlsplit(result["url"]).path for result in results}) == 3
    for result in results:
        parsed = urlsplit(result["url"])
        assert parsed.path.startswith("/prefix/lab/workspaces/course-")
        assert parsed.path.endswith("/tree/workspace/slides/01-tokenization/practice.ipynb")
        assert parse_qs(parsed.query)["token"] == [server["token"]]


def test_incomplete_runtime_records_are_ignored(course):
    runtime = course / "runtime"
    runtime.mkdir()
    (runtime / "jpserver-1.json").write_text("{")
    (runtime / "jpserver-2.json").write_text('{"pid": 2}')
    launcher = NotebookLauncher(course, runtime_dirs=[runtime])
    assert list(launcher.registered_servers()) == [{"pid": 2}]


def test_start_failure_stops_only_the_process_it_started(course, monkeypatch):
    import scripts.notebooks as module

    class Process:
        pid = 12345
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    launcher = NotebookLauncher(course, startup_timeout=0)
    with pytest.raises(RuntimeError, match="JupyterLab could not start"):
        launcher.start_server(launcher.prepare_notebook("01-tokenization"))
    assert process.terminated


@pytest.fixture
def preview_server(course):
    class Handler(PreviewHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=course, **kwargs)

        def log_message(self, *args):
            pass

    class Launcher:
        calls = []

        def open(self, lecture, notebook=None):
            self.calls.append((lecture, notebook))
            return {"url": "http://127.0.0.1:8888/lab", "reused": True}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.notebooks = Launcher()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def test_launch_requires_same_origin_post(preview_server):
    origin = f"http://127.0.0.1:{preview_server.server_port}"
    client = build_opener(ProxyHandler({}))
    for request_origin, marker in [("https://unrelated.example", "1"), (origin, "")]:
        request = Request(origin + "/_course/notebook", data=b'{"lecture":"01-tokenization"}', headers={
            "Origin": request_origin, "X-Course-Notebook": marker, "Content-Type": "application/json"
        })
        with pytest.raises(HTTPError) as error:
            client.open(request)
        assert error.value.code == 403
    assert preview_server.notebooks.calls == []
    request = Request(origin + "/_course/notebook", data=b'{"lecture":"01-tokenization"}', headers={
        "Origin": origin, "X-Course-Notebook": "1", "Content-Type": "application/json"
    })
    with client.open(request) as response:
        assert json.load(response)["reused"]
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Referrer-Policy"] == "no-referrer"
    assert preview_server.notebooks.calls == [("01-tokenization", None)]


def test_launch_passes_requested_notebook(preview_server):
    origin = f"http://127.0.0.1:{preview_server.server_port}"
    request = Request(origin + "/_course/notebook", data=json.dumps({
        "lecture": "lecture-01", "notebook": "lecture-01-exercise-tokenization.ipynb"
    }).encode(), headers={
        "Origin": origin, "X-Course-Notebook": "1", "Content-Type": "application/json"
    })
    with build_opener(ProxyHandler({})).open(request) as response:
        assert response.status == 200
    assert preview_server.notebooks.calls == [("lecture-01", "lecture-01-exercise-tokenization.ipynb")]
