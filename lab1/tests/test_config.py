from rag.config import PROJECT_ROOT, Settings, load_config


def test_default_config_paths_are_absolute():
    config = load_config()
    assert config.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert all(p.is_absolute() for _, p in config.paths)


def test_unknown_config_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pathz: {}\n", encoding="utf-8")
    try:
        load_config(bad)
    except ValueError:
        return
    raise AssertionError("опечатка в ключе конфига должна давать ошибку")


def test_langfuse_disabled_without_keys():
    settings = Settings(_env_file=None, langfuse_public_key=None, langfuse_secret_key=None)
    assert not settings.langfuse_enabled
