from encoders.utils import load_config


# test that config returns a dict
def test_load_config():
    cfg = load_config()

    assert isinstance(cfg, dict)


test_load_config()
