"""Create and preview the course's static Reveal.js lecture decks."""

import argparse
import html
import json
import re
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__:
    from .notebooks import NotebookLauncher
else:
    from notebooks import NotebookLauncher


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"


def create_lecture(number, title, slides_root=SLIDES):
    if not 1 <= number <= 99:
        raise ValueError("The lecture number must be between 1 and 99.")
    if not title.strip() or "\n" in title or "\r" in title:
        raise ValueError("Use a nonempty, single-line lecture title.")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        raise ValueError("The title needs an English word for the folder name.")
    destination = slides_root / f"{number:02d}-{slug}"
    if destination.exists():
        raise FileExistsError(f"Already exists: {destination}")
    template = slides_root / "template"
    replacements = {"{{NUMBER}}": f"{number:02d}", "{{TITLE}}": html.escape(title.strip())}
    metadata = json.loads((template / "lecture.json").read_text())
    metadata.update(number=f"{number:02d}", title=title.strip())
    metadata["subtitle"] = "Replace with the lecture's central question."
    notebook = json.loads((template / "practice.ipynb").read_text())
    notebook["cells"][0]["source"] = [f"# Lecture {number:02d}: {title.strip()}\n", "\n", "Companion exercises in slide order.\n"]
    destination.mkdir()
    shutil.copyfile(template / "index.html", destination / "index.html")
    source = (template / "slides.md").read_text()
    for key, value in replacements.items():
        source = source.replace(key, value)
    (destination / "slides.md").write_text(source)
    (destination / "lecture.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (destination / "practice.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    return destination


def vendor_assets():
    package = json.loads((SLIDES / "package.json").read_text())
    dependencies = SLIDES / "node_modules"
    vendor = SLIDES / "vendor"
    versions = {}
    for name in ("reveal.js", "katex", "plotly.js-dist-min"):
        manifest = dependencies / name / "package.json"
        if not manifest.exists():
            raise ValueError("Install authoring tools first: npm ci --prefix slides")
        version = json.loads(manifest.read_text())["version"]
        if version != package["dependencies"][name]:
            raise ValueError(f"Unexpected {name} version. Run npm ci --prefix slides.")
        versions[name] = version
    vendor.mkdir(exist_ok=True)
    # Keep browser assets and licenses in Git, so GitHub Pages needs no build step.
    shutil.copytree(dependencies / "reveal.js" / "dist", vendor / "reveal", dirs_exist_ok=True)
    shutil.copyfile(dependencies / "reveal.js" / "LICENSE", vendor / "reveal" / "LICENSE")
    shutil.copytree(dependencies / "katex" / "dist", vendor / "katex", dirs_exist_ok=True)
    shutil.copyfile(dependencies / "katex" / "LICENSE", vendor / "katex" / "LICENSE")
    (vendor / "plotly").mkdir(exist_ok=True)
    for filename in ("plotly.min.js", "LICENSE"):
        shutil.copyfile(dependencies / "plotly.js-dist-min" / filename, vendor / "plotly" / filename)
    (vendor / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    return versions


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(kwargs.pop("directory", ROOT)), **kwargs)

    def do_POST(self):
        if self.path != "/_course/notebook":
            self.send_error(404)
            return
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed or self.headers.get("Origin") != f"http://{host}" or self.headers.get("X-Course-Notebook") != "1":
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError("Invalid notebook request.")
            request = json.loads(self.rfile.read(length))
            result = self.server.notebooks.open(request.get("lecture"))
            self.send_json(200, result)
        except (ValueError, TypeError, AttributeError) as error:
            self.send_json(400, {"error": str(error)})
        except (OSError, RuntimeError):
            self.send_json(503, {"error": "JupyterLab could not start. Run uv sync and try again. Details are in workspace/.jupyter/lab.log."})

    def send_json(self, status, value):
        content = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_head(self):
        path = Path(self.translate_path(self.path)).resolve()
        try:
            relative = path.relative_to(Path(self.directory).resolve())
        except ValueError:
            self.send_error(404)
            return None
        if any(part.startswith(".") or part in {"node_modules", "workspace"} for part in relative.parts):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("new", help="Create a lecture without overwriting existing work")
    create.add_argument("number", type=int)
    create.add_argument("title")
    preview = sub.add_parser("serve", help="Preview the course and slides on this computer")
    preview.add_argument("--port", type=int, default=8000)
    sub.add_parser("vendor", help="Copy pinned browser dependencies into slides/vendor")
    args = parser.parse_args()
    try:
        if args.command == "new":
            print(create_lecture(args.number, args.title))
        elif args.command == "vendor":
            print(json.dumps(vendor_assets()))
        else:
            server = ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler)
            server.notebooks = NotebookLauncher(ROOT)
            print(f"Preview: http://127.0.0.1:{args.port}/slides/example/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
