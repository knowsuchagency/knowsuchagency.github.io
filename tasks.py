import atexit
import datetime as dt
import functools
import logging
import re
from pathlib import Path

import nbformat
import toml
from invoke import task
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from ruamel.yaml import YAML
from traitlets.config import Config
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


@task(aliases=["up"])
def serve(c, draft=True):
    """Run the hugo server alongside a bg thread that will re-render notebooks into md posts."""

    observer = Observer()

    observer.schedule(NotebookHandler(c), "knowsuchagency_blog/notebooks")

    observer.schedule(ConfigHandler(c), ".")

    observer.start()

    atexit.register(functools.partial(observer.join, timeout=0.1))

    c.run("hugo serve" + " -D" if draft else "")


@task
def render_notebooks(c, reload_config=False):
    notebooks_path = Path("knowsuchagency_blog", "notebooks")

    notebooks = (n for n in notebooks_path.iterdir() if n.suffix == ".ipynb")

    for notebook in notebooks:

        front_matter = {}

        notebooks_config = (
            c.config.notebooks
            if not reload_config
            else YAML().load(Path("invoke.yaml"))["notebooks"]
        )

        if notebook.stem in notebooks_config:
            front_matter.update(notebooks_config[notebook.stem])
        else:
            logging.warning(f"no front-matter yet defined for {notebook}")

        Path("content", "posts", f"{notebook.stem}.md").write_text(
            render(notebook, **front_matter)
        )


@functools.singledispatch
def render(
    notebook: str,
    draft=True,
    date=None,
    title=None,
    description=None,
    slug=None,
    tags=None,
    categories=None,
    external_link=None,
    series=None,
    prevent_summary=True,
) -> str:
    """Return hugo-formatted markdown from a notebook."""

    front_matter = {
        "draft": draft,
        "date": date or dt.datetime.now(),
        "title": title or "",
        "description": description or "",
        "slug": slug or "",
        "tags": tags or [],
        "categories": categories or [],
        "externalLink": external_link or "",
        "series": series or [],
    }

    notebook = nbformat.reads(notebook, as_version=4)

    config = Config()

    config.MarkdownExporter.preprocessors = [CustomPreprocessor]

    markdown_exporter = MarkdownExporter(config=config)

    markdown, _ = markdown_exporter.from_notebook_node(notebook)

    clean_md = CustomPreprocessor.clean(markdown)

    return "\n".join(
        (
            "+++",
            toml.dumps(front_matter).strip(),
            "+++",
            "<!--more-->" if prevent_summary else "",
            clean_md,
        )
    )


@render.register
def _(notebook: Path, **kwargs):
    return render(notebook.read_text(), **kwargs)


class CustomPreprocessor(Preprocessor):
    """Remove blank code cells and unnecessary whitespace."""

    def preprocess(self, nb, resources):
        """
        Remove blank cells
        """
        for index, cell in enumerate(nb.cells):

            if cell.cell_type == "code" and not cell.source:

                nb.cells.pop(index)

            else:

                nb.cells[index], resources = self.preprocess_cell(
                    cell, resources, index
                )

        return nb, resources

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Remove extraneous whitespace from code cells' source code
        """
        if cell.cell_type == "code":
            cell.source = cell.source.strip()

        return cell, resources

    @staticmethod
    def clean(string: str) -> str:
        """Get rid of all the wacky newlines nbconvert adds to markdown output and return result."""

        post_code_newlines_patt = re.compile(r"(```)(\n+)")

        inter_output_newlines_patt = re.compile(r"(\s{4}\S+)(\n+)(\s{4})")

        post_code_filtered = re.sub(post_code_newlines_patt, r"\1\n\n", string)

        inter_output_filtered = re.sub(
            inter_output_newlines_patt, r"\1\n\3", post_code_filtered
        )

        return inter_output_filtered


class NotebookHandler(PatternMatchingEventHandler):
    """Handle notebook changes."""

    patterns = ["*.ipynb"]

    def __init__(self, context):
        self.context = context
        super().__init__()

    def process(self, event):
        if (
            "untitled" not in event.src_path.lower()
            and ".~" not in event.src_path
        ):
            render_notebooks(self.context, reload_config=True)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


class ConfigHandler(PatternMatchingEventHandler):
    """Handle configuration changes for notebooks."""

    pattern = ["invoke.yaml"]

    def __init__(self, context):
        self.context = context
        super().__init__()

    def on_modified(self, event):
        render_notebooks(self.context, reload_config=True)
