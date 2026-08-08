import os


os.environ["MDNS_ENABLE"] = "false"
os.environ.pop("HA_SMOOTHING_ENABLE", None)

import app


def setup_function():
    app._SMOOTHING_BUFFERS.clear()
    app._REQUEST_IP_SEEN.clear()


def test_smoothing_defaults_to_three_datapoints():
    assert app.HA_SMOOTHING_ENABLE is True
    assert app.HA_SMOOTHING_WINDOW == 3
    assert app._smooth_value("sensor.power", 3.0) == 3.0
    assert app._smooth_value("sensor.power", 6.0) == 4.5
    assert app._smooth_value("sensor.power", 9.0) == 6.0
    assert app._smooth_value("sensor.power", 12.0) == 9.0


def test_request_scaling_splits_over_asking_devices_minus_one(monkeypatch):
    monkeypatch.setattr(app, "REQUEST_SIDE_SCALING_ENABLE", True)
    monkeypatch.setattr(app, "REQUEST_SIDE_SCALING_CLIENTS", 0)
    monkeypatch.setattr(app, "_active_request_ip_count", lambda: 4)

    assert app._apply_request_side_power_scaling((90.0, 60.0, 30.0)) == (30.0, 20.0, 10.0)


def test_request_scaling_divisor_has_minimum_of_one(monkeypatch):
    monkeypatch.setattr(app, "REQUEST_SIDE_SCALING_ENABLE", True)
    monkeypatch.setattr(app, "REQUEST_SIDE_SCALING_CLIENTS", 0)
    monkeypatch.setattr(app, "_active_request_ip_count", lambda: 2)

    assert app._apply_request_side_power_scaling((90.0, 60.0, 30.0)) == (90.0, 60.0, 30.0)
