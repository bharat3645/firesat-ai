from __future__ import annotations

import pytest

from firesat.config import Region
from firesat.data.synthetic import SyntheticDataGenerator

TEST_SEQUENCE_LENGTH = 6


@pytest.fixture
def tiny_region() -> Region:
    return Region(
        id="test-region",
        name="Test Region",
        description="Small synthetic region used only in unit tests.",
        min_lon=-150.0,
        min_lat=60.0,
        max_lon=-149.0,
        max_lat=61.0,
    )


@pytest.fixture
def tiny_region_b() -> Region:
    return Region(
        id="test-region-b",
        name="Test Region B",
        description="Second small synthetic region for multi-region tests.",
        min_lon=-148.0,
        min_lat=63.0,
        max_lon=-147.0,
        max_lat=64.0,
    )


@pytest.fixture
def tiny_dataset(tiny_region):
    gen = SyntheticDataGenerator(
        region=tiny_region, start_year=2020, end_year=2021, grid_size=4, seed=1
    )
    return gen.generate()


@pytest.fixture
def tiny_datasets(tiny_region, tiny_region_b):
    gen_a = SyntheticDataGenerator(
        region=tiny_region, start_year=2020, end_year=2021, grid_size=4, seed=1
    )
    gen_b = SyntheticDataGenerator(
        region=tiny_region_b, start_year=2020, end_year=2021, grid_size=4, seed=2
    )
    return {tiny_region.id: gen_a.generate(), tiny_region_b.id: gen_b.generate()}
