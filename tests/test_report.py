import pytest

from conso_forecast import report


def test_improvement_pct():
    assert report.improvement_pct(750.0, 1000.0) == pytest.approx(25.0)


def test_update_readme_replaces_only_the_results_block():
    readme = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n"
    new_block = "<!-- RESULTS:START -->\nnew $1 \\1\n<!-- RESULTS:END -->"
    updated = report.update_readme(readme, new_block)
    assert updated == "intro\n" + new_block + "\noutro\n"


def test_update_readme_requires_markers():
    with pytest.raises(ValueError):
        report.update_readme("no markers here", "block")
