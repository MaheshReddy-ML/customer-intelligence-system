from pathlib import Path

from src.utils.config import load_config


def test_load_config_reads_yaml_file(tmp_path: Path):
    """Validate project YAML configuration loading."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "project:\n  name: test-project\nclustering:\n  n_clusters: 3\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["project"]["name"] == "test-project"
    assert config["clustering"]["n_clusters"] == 3
