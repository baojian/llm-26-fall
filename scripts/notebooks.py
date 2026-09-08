"""Open personal lecture notebooks in a compatible local JupyterLab server."""

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import psutil


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


class NotebookLauncher:
    def __init__(self, root, *, python=None, runtime_dirs=None, startup_timeout=30):
        self.root = Path(root).resolve()
        self.python = os.path.abspath(python or sys.executable)
        self.state = self.root / "workspace" / ".jupyter"
        self.runtime_dirs = runtime_dirs
        self.startup_timeout = startup_timeout
        self.lock = threading.Lock()
        self.process = None
        # Never forward local Jupyter credentials through a configured proxy.
        self.http = build_opener(ProxyHandler({}), NoRedirect())

    def prepare_notebook(self, lecture, notebook=None):
        if not isinstance(lecture, str) or not re.fullmatch(r"example|lecture-[0-9]{2}|[0-9]{2}-[a-z0-9-]+", lecture):
            raise ValueError("Choose a lecture from the course slides.")
        deck = self.root / "slides" / lecture
        metadata_path = deck / "lecture.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        default = metadata.get("notebook", "practice.ipynb")
        filename = default if notebook is None else notebook
        if not isinstance(filename, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*\.ipynb", filename):
            raise ValueError("The lecture notebook must be an .ipynb file in its lecture folder.")
        if filename != default and filename not in metadata.get("additional_notebooks", []):
            raise ValueError("Choose a notebook listed for this lecture.")
        source = deck / filename
        if not source.is_file() or not source.resolve().is_relative_to(self.root / "slides"):
            raise ValueError("This lecture does not have a practice notebook yet.")
        destination = self.root / "workspace" / "slides" / lecture / filename
        if not destination.resolve().is_relative_to(self.root / "workspace"):
            raise ValueError("The notebook workspace must stay inside this repository.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            notebook = json.loads(source.read_text(encoding="utf-8"))
            notebook.setdefault("metadata", {})["kernelspec"] = {
                "name": "python3", "display_name": "Python 3 (course uv)", "language": "python"
            }
            try:
                with destination.open("x", encoding="utf-8") as output:
                    json.dump(notebook, output, ensure_ascii=False, indent=2)
                    output.write("\n")
            except FileExistsError:
                pass
        assets = source.parent / "assets"
        personal_assets = destination.parent / "assets"
        if assets.is_dir():
            for asset in assets.rglob("*"):
                if not asset.is_file():
                    continue
                target = personal_assets / asset.relative_to(assets)
                if not target.resolve().is_relative_to(self.root / "workspace"):
                    raise ValueError("Notebook assets must stay inside the workspace.")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with target.open("xb") as output, asset.open("rb") as source_file:
                        shutil.copyfileobj(source_file, output)
                except FileExistsError:
                    pass
        return destination

    def registered_servers(self):
        if self.runtime_dirs is None:
            from jupyter_core.paths import jupyter_runtime_dir

            directories = [self.state / "runtime", Path(jupyter_runtime_dir())]
        else:
            directories = list(self.runtime_dirs)
        for directory in dict.fromkeys(directories):
            for pattern in ("jpserver-*.json", "nbserver-*.json"):
                for filename in Path(directory).glob(pattern):
                    try:
                        yield json.loads(filename.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue

    def server_base(self, server):
        parts = urlsplit(server.get("url", ""))
        if parts.scheme not in {"http", "https"} or parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only local Jupyter servers can open course notebooks.")
        if parts.username or parts.password:
            raise ValueError("Unsupported Jupyter URL.")
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))

    def request(self, server, route):
        headers = {"Authorization": f"token {server['token']}"} if server.get("token") else {}
        request = Request(self.server_base(server) + route, headers=headers)
        return self.http.open(request, timeout=1.5)

    def compatible(self, server, notebook):
        try:
            server_root = Path(server.get("root_dir") or server.get("notebook_dir") or "").resolve()
            if not (server.get("root_dir") or server.get("notebook_dir")) or not notebook.is_relative_to(server_root):
                return False
            with self.request(server, "api/kernelspecs") as response:
                spec = json.load(response)["kernelspecs"]["python3"]["spec"]
            executable = spec["argv"][0]
            if executable in {"python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"}:
                # Jupyter replaces these native kernel commands with its own
                # sys.executable. Inspect that server's process to identify it.
                try:
                    executable = psutil.Process(server["pid"]).cmdline()[0]
                except (psutil.Error, KeyError, IndexError, TypeError):
                    return False
            # Resolving the executable symlink would conflate different virtual environments.
            if not os.path.isabs(executable) or Path(executable).parent != Path(self.python).parent:
                return False
            if Path(executable).name not in {"python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}", Path(self.python).name}:
                return False
            relative = quote(notebook.relative_to(server_root).as_posix(), safe="/")
            with self.request(server, f"api/contents/{relative}?content=0") as response:
                if json.load(response).get("type") != "notebook":
                    return False
            with self.request(server, "lab") as response:
                return response.status == 200
        except (OSError, ValueError, KeyError, IndexError, TypeError, HTTPError, URLError):
            return False

    def start_server(self, notebook):
        if not self.state.resolve().is_relative_to(self.root / "workspace"):
            raise ValueError("Jupyter state must stay inside this repository.")
        runtime = self.state / "runtime"
        runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = self.state / "data"
        kernel = data / "kernels" / "python3"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [self.python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3 (course uv)", "language": "python"
        }), encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "JUPYTER_RUNTIME_DIR": str(runtime),
            "JUPYTER_DATA_DIR": str(data),
            "JUPYTER_PATH": str(data) + (os.pathsep + environment["JUPYTER_PATH"] if environment.get("JUPYTER_PATH") else ""),
            "JUPYTER_CONFIG_DIR": str(self.state / "config"),
            "JUPYTERLAB_SETTINGS_DIR": str(self.state / "settings"),
            "JUPYTERLAB_WORKSPACES_DIR": str(self.state / "workspaces"),
            "IPYTHONDIR": str(self.state / "ipython"),
            "JUPYTER_TOKEN": secrets.token_urlsafe(32),
        })
        log_path = self.state / "lab.log"
        with log_path.open("ab") as log:
            log_path.chmod(0o600)
            self.process = subprocess.Popen([
                self.python, "-m", "jupyterlab", "--no-browser",
                "--ServerApp.ip=127.0.0.1", "--ServerApp.port=8888",
                "--ServerApp.port_retries=50", "--ServerApp.base_url=/",
                f"--ServerApp.root_dir={self.root}",
            ], cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline and self.process.poll() is None:
            for filename in runtime.glob(f"jpserver-{self.process.pid}.json"):
                try:
                    server = json.loads(filename.read_text(encoding="utf-8"))
                    if self.compatible(server, notebook):
                        return server
                except (OSError, ValueError):
                    pass
            time.sleep(0.2)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        raise RuntimeError("JupyterLab could not start. Check workspace/.jupyter/lab.log, then run uv sync and try again.")

    def open(self, lecture, notebook=None):
        with self.lock:
            notebook = self.prepare_notebook(lecture, notebook)
            server = next((item for item in self.registered_servers() if self.compatible(item, notebook)), None)
            reused = server is not None
            if server is None:
                server = self.start_server(notebook)
            server_root = Path(server.get("root_dir") or server["notebook_dir"]).resolve()
            relative = quote(notebook.relative_to(server_root).as_posix(), safe="/")
            # Separate Lab workspaces avoid its "already open in another tab" prompt.
            workspace = "course-" + secrets.token_hex(8)
            url = self.server_base(server) + f"lab/workspaces/{workspace}/tree/{relative}"
            if server.get("token"):
                url += "?" + urlencode({"token": server["token"]})
            return {"url": url, "reused": reused, "notebook": notebook.relative_to(self.root).as_posix()}
