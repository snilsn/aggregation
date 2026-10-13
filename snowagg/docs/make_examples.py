"""The examples page, docs/examples.md: for each script in examples/ its
explanation (the docstring), its figure and the full script, copied from the
file so the page always shows the code that is there.
tests/test_docs.py checks that the page is up to date.

usage: python docs/make_examples.py [--figures]
    --figures: also run the examples and redraw their figures
               (docs/img/examples/, about 1 minute)
"""
import ast
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")
IMG = os.path.join(HERE, "img", "examples")
OUT = os.path.join(HERE, "examples.md")

HEADER = """# Examples

[← README](../../README.md) · [User guide](guide.md) · **Examples** · [Technical details](../README.md)

Complete scripts in [`snowagg/examples/`](../examples), each a few seconds to a minute. Run one
with `python examples/<name>.py`: it prints its results, and with a file name as argument it
also draws its figure (`python examples/crystals.py crystals.png`). The [user guide](guide.md)
explains the functions they use. This page is written by `docs/make_examples.py`.
"""

# (script, title, figure?, run time)
ENTRIES = [
    ("crystals", "The crystal types", True, "3 s"),
    ("first_snowflake", "A first snowflake", True, "3 s"),
    ("riming", "More and more rime", True, "12 s"),
    ("growth_stages", "Watching a snowflake grow", True, "8 s"),
    ("deposition", "Deposition and sublimation", True, "30 s"),
    ("mass_size", "A mass-size relation", True, "4 s"),
    ("export", "Saving a snowflake", False, "1 s"),
    ("parallel_ensemble", "An ensemble in parallel processes", False, "6 CPU minutes: 30-45 s on 8-16 CPUs"),
]


def anchor(title):
    return "".join(c for c in title.lower().replace(" ", "-") if c.isalnum() or c == "-")


def render():
    parts = [HEADER, "\n".join(f"* [{title}](#{anchor(title)})" for _, title, _, _ in ENTRIES), ""]
    for name, title, figure, time in ENTRIES:
        with open(os.path.join(EXAMPLES, name + ".py")) as f:
            script = f.read().rstrip("\n")
        doc = ast.get_docstring(ast.parse(script))
        text, _, usage = doc.partition("\n\n    python ")
        parts += [f"## {title}", "", text.strip(), ""]
        if figure:
            parts += [f"![{title}](img/examples/{name}.png)", ""]
        parts += [f"Run: `python examples/{name}.py{' ' + name + '.png' if figure else ''}` ({time}).", "",
                  f"<details><summary>The script: <code>examples/{name}.py</code></summary>", "",
                  "```python", script, "```", "", "</details>", ""]
    return "\n".join(parts)


def figures():
    os.makedirs(IMG, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "python"), OMP_NUM_THREADS="2")
    for name, _, figure, _ in ENTRIES:
        if figure:
            print("running", name, flush=True)
            subprocess.run([sys.executable, os.path.join(EXAMPLES, name + ".py"), os.path.join(IMG, name + ".png")],
                           check=True, env=env)
    with tempfile.TemporaryDirectory() as tmp:  # the export example writes files
        subprocess.run([sys.executable, os.path.join(EXAMPLES, "export.py"), tmp], check=True, env=env)


if __name__ == "__main__":
    if "--figures" in sys.argv:
        figures()
    with open(OUT, "w") as f:
        f.write(render())
    print("wrote", OUT)
