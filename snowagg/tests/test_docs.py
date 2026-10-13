"""The documentation: the examples page shows the example scripts as they are."""
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "docs"))


def test_examples_page_is_up_to_date():
    import make_examples

    with open(make_examples.OUT) as f:
        assert f.read() == make_examples.render(), "run python docs/make_examples.py"
