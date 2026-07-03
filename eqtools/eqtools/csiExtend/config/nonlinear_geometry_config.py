"""Configuration parser for the new nonlinear geometry SMC entry point."""

from __future__ import annotations

import logging

import yaml

from ..data_corrections import normalize_data_correction_specs
from .base_config import CommonConfigBase
from .config_utils import normalize_units_config, parse_sigmas_config
from .explore_config import AliasManager
from .prior_bounds import (
    LOWER_RANGE,
    LOWER_UPPER,
    normalize_prior_bounds_tree,
    validate_prior_bounds_format,
)

logger = logging.getLogger(__name__)


class NonlinearGeometryConfig(CommonConfigBase):
    """Parse the new nonlinear geometry SMC configuration.

    User-facing bounds default to ``lower_upper`` for readability.  The loaded
    object stores all scipy distribution specs as ``lower_range`` because the
    current SMC sampler consumes scipy ``loc/scale`` arguments.
    """

    default_prior_bounds_format = LOWER_UPPER

    def __init__(self, config_file=None, geodata=None, verbose=False, parallel_rank=None):
        self._sigmas_param_name = "values"
        self.input_prior_bounds_format = self.default_prior_bounds_format
        self.prior_bounds_format = LOWER_RANGE
        self.bounds = {}
        self.initial = {}
        self.fixed_params = {}
        self.nfaults = 1
        self.faultnames = [f"fault_{i}" for i in range(self.nfaults)]
        self.slip_sampling_mode = "mag_rake"
        self.fault_aliasnames = None
        self.data_correction_specs = []

        super().__init__(
            config_file=config_file,
            geodata=geodata,
            verbose=verbose,
            parallel_rank=parallel_rank,
        )

        if config_file:
            self.load_config(config_file, geodata=geodata)

        if self.geodata and "data" in self.geodata:
            data_names = [d.name for d in self.geodata.get("data", [])]
            if "sigmas" in self.geodata:
                self.sigmas = parse_sigmas_config(
                    self.geodata["sigmas"],
                    dataset_names=data_names,
                    param_name="values",
                )

    def load_config(self, config_file, geodata=None):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}

        self._normalize_prior_bounds_config(config)

        nfaults = config.get("nfaults", 1)
        user_aliases = config.get("fault_aliasnames", config.get("fault_names", None))
        self.alias_manager = AliasManager(nfaults, user_aliases)
        self.fault_id_to_alias = self.alias_manager.id_to_alias

        for section in ["bounds", "initial", "fixed_params"]:
            if section in config:
                config[section] = self.alias_manager.translate_config_keys(config[section])

        if "geodata" in config and "faults" in config["geodata"]:
            config["geodata"]["faults"] = self.alias_manager.translate_faults_list(
                config["geodata"]["faults"]
            )

        self.bounds = config.get("bounds", {})
        self.initial = config.get("initial", {})
        self.fixed_params = config.get("fixed_params", {})
        self.nfaults = nfaults
        self.fault_aliasnames = user_aliases
        self.faultnames = [f"fault_{i}" for i in range(self.nfaults)]
        self.slip_sampling_mode = config.get("slip_sampling_mode", "mag_rake")
        self.clipping_options = config.get("clipping_options", {})
        self.units = normalize_units_config(config.get("units"))
        self.geodata = config.get("geodata", {})

        lon_lat_0 = config.get("lon_lat_0", None)
        if lon_lat_0:
            self.lon0, self.lat0 = lon_lat_0
        self.data_sources = config.get("data_sources", {})

        self._update_geodata(geodata)
        self._validate_verticals()

        if "data" in self.geodata:
            data_names = [d.name for d in self.geodata.get("data", [])]
            if "sigmas" in self.geodata:
                self.sigmas = parse_sigmas_config(
                    self.geodata["sigmas"],
                    dataset_names=data_names,
                    param_name="values",
                )

        self.dataFaults = self.geodata.get("faults", None)
        self.data_correction_specs = normalize_data_correction_specs(
            self.geodata,
            verticals=self.geodata.get("verticals"),
        )
        self._select_data_sets()

    def _normalize_prior_bounds_config(self, config):
        input_format = validate_prior_bounds_format(
            config.get("prior_bounds_format", self.default_prior_bounds_format)
        )
        self.input_prior_bounds_format = input_format

        if "bounds" in config:
            config["bounds"] = normalize_prior_bounds_tree(
                config["bounds"],
                input_format,
                context="bounds",
            )

        geodata = config.get("geodata") or {}
        if geodata.get("poly_bounds") is not None:
            geodata["poly_bounds"] = normalize_prior_bounds_tree(
                geodata["poly_bounds"],
                input_format,
                context="geodata.poly_bounds",
            )

        polys = geodata.get("polys")
        if isinstance(polys, dict):
            boundaries = polys.get("boundaries")
            if boundaries is not None:
                polys["boundaries"] = normalize_prior_bounds_tree(
                    boundaries,
                    input_format,
                    context="geodata.polys.boundaries",
                )

        sigmas = geodata.get("sigmas")
        if isinstance(sigmas, dict) and sigmas.get("bounds") is not None:
            sigmas["bounds"] = normalize_prior_bounds_tree(
                sigmas["bounds"],
                input_format,
                context="geodata.sigmas.bounds",
            )

        corrections = geodata.get("data_corrections")
        if isinstance(corrections, dict):
            geodata["data_corrections"] = normalize_prior_bounds_tree(
                corrections,
                input_format,
                context="geodata.data_corrections",
            )

        config["prior_bounds_format"] = LOWER_RANGE
        self.prior_bounds_format = LOWER_RANGE
