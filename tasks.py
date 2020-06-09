import atexit
import csv
import datetime as dt
import functools
import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import *

import nbformat
import toml
from invoke import task
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from ruamel.yaml import YAML
from traitlets.config import Config
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


@dataclass
class Post:
    title: str
    description: str
    slug: str = ""
    externalLink: str = ""
    draft: bool = True
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    series: List[str] = field(default_factory=list)
    date: dt.datetime = field(default_factory=dt.datetime.now)

    def render(self, markdown) -> str:
        return "\n".join(
            ("+++", toml.dumps(asdict(self)).strip(), "+++", "<!--more-->", markdown,)
        )

    @functools.singledispatchmethod
    def render_from_jupyter(self, notebook: str):
        notebook = nbformat.reads(notebook, as_version=4)

        config = Config()

        config.MarkdownExporter.preprocessors = [CustomPreprocessor]

        markdown_exporter = MarkdownExporter(config=config)

        markdown, _ = markdown_exporter.from_notebook_node(notebook)

        clean_md = CustomPreprocessor.clean(markdown)

        return self.render(clean_md)

    @render_from_jupyter.register
    def _(self, notebook: Path):
        return self.render_from_jupyter(notebook.read_text())

    @staticmethod
    def from_notion(filepath: Union[os.PathLike, str]) -> Tuple["Post", str]:
        """Given a zip file with a table that contains blogs, return posts and their markdown content."""
        with zipfile.ZipFile(filepath) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as csv_file:
                        csv_file = io.TextIOWrapper(csv_file, encoding="utf-8-sig")
                        reader = csv.DictReader(csv_file)
                        csv_data = {}
                        for row in reader:
                            title = row.pop("Title")
                            # skip templates
                            if title in ("Business Blog Post", "Technical Blog Post"):
                                continue
                            csv_data[title] = row
                else:

                    with zf.open(name) as fp:
                        fp = io.TextIOWrapper(fp, encoding="utf-8-sig")
                        title = fp.readline().strip("#").strip()
                        # skip templates
                        if title in ("Business Blog Post", "Technical Blog Post"):
                            continue

                        fp.readline()

                        content = []

                        variable_prefixes = (
                            "audience",
                            "description",
                            "draft",
                            "keywords",
                            "tags",
                            "series",
                        )

                        while line := fp.readline():

                            for variable in variable_prefixes:
                                line_start = f"{variable.title()}: "
                                if line.startswith(line_start):
                                    break
                            else:
                                content.append(line)

                        content = os.linesep.join(content).strip()

                        csv_data[title]["markdown"] = content

            def parse_array(string):
                return [s.strip() for s in string.split(",")]

            result = []

            for title, data in csv_data.items():

                post = Post(
                    title=title,
                    description=data["Description"],
                    slug=data["Slug"],
                    draft=data["Draft"].strip().lower().startswith("y"),
                    categories=parse_array(data["Keywords"]),
                    series=parse_array(data["Series"]),
                    tags=parse_array(data["Tags"])
                    # TODO: implement date parsing
                )

                markdown = data["markdown"]

                result.append((post, markdown,))

            return result


@task(aliases=["up"])
def serve(c, draft=True):
    """Run the hugo server alongside a background thread that will re-render notebooks into markdown posts."""

    observer = Observer()

    observer.schedule(NotebookHandler(c), "knowsuchagency_blog/notebooks")

    observer.schedule(ConfigHandler(c), ".")

    observer.start()

    atexit.register(functools.partial(observer.join, timeout=0.1))

    c.run("hugo serve" + " -D" if draft else "")


@task
def render_notebooks(c, reload_config=False, notion_zip=None):
    """Render notebooks into respective markdown posts."""
    notion_zip = notion_zip or c.config.notion.zipfile

    notebooks_path = Path("knowsuchagency_blog", "notebooks")

    notebooks = (n for n in notebooks_path.iterdir() if n.suffix == ".ipynb")

    posts_path = Path("content", "posts")

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

        Path(posts_path, f"{notebook.stem}.md").write_text(
            Post(**front_matter).render_from_jupyter(notebook)
        )

    if notion_zip:
        for post, markdown in Post.from_notion(notion_zip):
            filename = post.title.replace(" ", "-").replace("/", "-or-") + ".md"
            Path(posts_path, filename).write_text(post.render(markdown))


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
        if "untitled" not in event.src_path.lower() and ".~" not in event.src_path:
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
