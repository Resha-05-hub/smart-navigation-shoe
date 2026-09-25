"""Consistency checks for config/config.yaml and hardware placeholders."""

from unittest.mock import patch

import pytest

from src import main as app_main
from src.core.enums import ObstacleZone
from src.pipeline import create_detector
from src.sensors.sensor_manager import SensorManager
from src.utils import helpers
from src.utils.helpers import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_default_mode_is_valid(config):
    assert config["system"]["mode"] in app_main.MODES


def test_gpio_pins_do_not_overlap(config):
    sensor_cfg = config["sensors"]["distance_sensor"]
    sensor_pins = list(sensor_cfg["gpio_trigger_pins"].values()) + list(sensor_cfg["gpio_echo_pins"].values())
    motor_pins = list(config["alerts"]["vibration"]["gpio_pins"].values())
    all_pins = sensor_pins + motor_pins
    assert len(all_pins) == len(set(all_pins)), f"GPIO pin used twice: {sorted(all_pins)}"


def test_ultrasonic_placeholders_get_their_own_position_and_pins(config):
    cfg = {"sensors": {"distance_sensor": dict(config["sensors"]["distance_sensor"], type="ultrasonic")}}
    manager = SensorManager(config=cfg)
    for zone in (ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT):
        sensor = manager.get_sensor(zone)
        key = zone.value.lower()
        assert sensor.position == zone
        assert sensor.trigger_pin == cfg["sensors"]["distance_sensor"]["gpio_trigger_pins"][key]
        assert sensor.echo_pin == cfg["sensors"]["distance_sensor"]["gpio_echo_pins"][key]


def test_detector_uses_configured_zone_boundaries(config):
    config = dict(config, risk_analysis=dict(config["risk_analysis"], zones={"left_boundary": 0.4, "right_boundary": 0.6}))
    assert create_detector(config).zone_boundaries == (0.4, 0.6)


def test_removed_dead_keys_stay_removed(config):
    assert "log_file" not in config["logging"]
    assert "critical_distance_m" not in config["risk_analysis"]
    assert "frame_skip" not in config["camera"]


def test_missing_pyyaml_fails_loudly():
    with patch.object(helpers, "yaml", None):
        with pytest.raises(ImportError, match="PyYAML"):
            load_config()
