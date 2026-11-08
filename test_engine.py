import pytest, engine as en

@pytest.fixture
def df(): return en.load()

def test_db_has_at_least_15_materials_and_all_families(df):
    assert len(df) >= 15 and set(df.family) == set(en.FAMILIES)

def test_filter_bounds_inclusive(df):
    out = en.filter_materials(df, {"sigma_y": (295, 295)})
    assert out.material.tolist() == ["Steel AISI 1020 (annealed)"]

def test_filter_yield_above_200(df):
    assert (en.filter_materials(df, {"sigma_y": (200, None)}).sigma_y >= 200).all()

def test_filter_invalid_range_raises(df):
    with pytest.raises(ValueError): en.filter_materials(df, {"E": (10, 5)})

def test_filter_no_match_is_empty(df):
    assert en.filter_materials(df, {"cost": (None, 0.01)}).empty

@pytest.mark.parametrize("bad", [0, -5, "abc", None, float("nan")])
def test_validate_rejects_bad_numbers(bad):
    with pytest.raises(ValueError):
        en.validate({"material": "X", "family": "Metal", "density": bad, "E": 1, "sigma_y": 1, "cost": 1})

def test_validate_rejects_duplicate_and_family():
    row = {"material": "x", "family": "Metal", "density": 1, "E": 1, "sigma_y": 1, "cost": 1}
    with pytest.raises(ValueError): en.validate(row, ["X"])
    with pytest.raises(ValueError): en.validate({**row, "family": "Wood"})

def test_index_value_specific_strength(df):
    r = df.iloc[0]
    assert en.index_value(df, "Light, strong tie  (σy/ρ)")[0] == pytest.approx(r.sigma_y / r.density)

def test_rank_best_is_100_percent(df):
    r = en.rank(df, "Light, stiff beam  (E^½/ρ)")
    assert r["relative_to_best_%"].iloc[0] == pytest.approx(100)

def test_pareto_not_dominated(df):
    f = en.pareto_front(df, "density", "sigma_y")
    assert 0 < len(f) < len(df)

def test_add_material_roundtrip(tmp_path):
    p = tmp_path / "m.json"; p.write_text(en.DB_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    en.add_material({"material": "Test", "family": "Metal", "density": 1000, "E": 10, "sigma_y": 50, "cost": 1}, p)
    assert "Test" in en.load(p).material.values
