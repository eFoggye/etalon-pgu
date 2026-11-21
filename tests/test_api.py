import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from powerplant.api import create_app

TYPICAL = {"AT": 20.3, "AP": 1012.9, "RH": 75.0}


@pytest.fixture(scope="module")
def client(small_bundle):
    with TestClient(create_app(bundle=small_bundle)) as test_client:
        yield test_client


def predict(client: TestClient, **payload) -> dict:
    response = client.post("/api/predict", json={**TYPICAL, **payload})
    assert response.status_code == 200, response.text
    return response.json()


def csv_upload(frame: pd.DataFrame) -> dict:
    return {"file": ("hours.csv", frame.to_csv(index=False).encode(), "text/csv")}


def test_health_reports_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "test"


def test_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>" in response.text


def test_plan_forecast_has_interval_and_no_warnings_for_typical_hour(client):
    body = predict(client)

    plan = body["plan"]
    assert plan["low"] < plan["power"] < plan["high"]
    assert body["reference"] is None
    assert body["warnings"] == []


def test_contributions_add_up_to_forecast(client):
    plan = predict(client)["plan"]

    total = plan["base_mw"] + sum(item["mw"] for item in plan["contributions"])

    assert total == pytest.approx(plan["power"], abs=0.05)


def test_physically_impossible_humidity_is_rejected(client):
    response = client.post("/api/predict", json={**TYPICAL, "RH": 150})

    assert response.status_code == 422


def test_unseen_combination_is_flagged(client):
    body = predict(client, AT=33.0, V=38.0)

    assert any("Сочетание" in warning for warning in body["warnings"])


def test_shortfall_risk_grows_with_required_volume(client):
    low_risk = predict(client, required_mw=430.0)["plan_risk"]["shortfall_probability"]
    high_risk = predict(client, required_mw=470.0)["plan_risk"]["shortfall_probability"]

    assert 0.0 <= low_risk < high_risk <= 1.0


def test_deviation_is_split_into_vacuum_and_other(client):
    body = predict(client, V=52.0, actual_mw=440.0)

    deviation = body["deviation"]
    assert body["reference"] is not None
    assert deviation["total_mw"] == pytest.approx(
        deviation["vacuum_mw"] + deviation["other_mw"], abs=0.02
    )
    assert deviation["rub_per_hour"] >= 0


def test_meta_lists_features_with_training_ranges_and_presets(client):
    body = client.get("/api/meta").json()

    names = [feature["name"] for feature in body["features"]]
    assert names == ["AT", "V", "AP", "RH"]
    assert body["features"][0]["train_min"] < body["features"][0]["train_max"]
    assert "Типичный час" in body["presets"]


def test_batch_flags_sustained_drop_in_example(client, small_bundle):
    response = client.post("/api/batch", files=csv_upload(small_bundle.example_batch))

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["summary"]["rows"] == len(small_bundle.example_batch)
    assert body["summary"]["alarm_hours"] > 0
    assert not any(row["alarm"] for row in body["rows"][:24])


def test_batch_example_is_downloadable(client):
    response = client.get("/api/batch/example")

    assert response.status_code == 200
    frame = pd.read_csv(io.StringIO(response.text))
    assert {"AT", "V", "AP", "RH", "PE"} <= set(frame.columns)


def test_batch_rejects_file_without_required_columns(client):
    frame = pd.DataFrame({"AT": [20.0], "RH": [70.0]})

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 422
    assert "AP" in response.json()["detail"]


def test_batch_rejects_too_many_rows(client):
    frame = pd.DataFrame({**{key: [value] * 10_001 for key, value in TYPICAL.items()}})

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 413


def test_batch_ignores_extra_text_columns(client, small_bundle):
    frame = small_bundle.example_batch.assign(время="2026-11-21 12:00")

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 200, response.text


def test_batch_rejects_header_only_file(client):
    response = client.post("/api/batch", files=csv_upload(pd.DataFrame(columns=["AT", "AP", "RH"])))

    assert response.status_code == 422
    assert "строк" in response.json()["detail"]


def test_batch_accepts_plant_shutdown_values(client, small_bundle):
    frame = small_bundle.example_batch.assign(PE=0.0)

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 200, response.text


def test_batch_shorter_than_a_day_raises_no_alarm(client, small_bundle):
    frame = small_bundle.example_batch.head(10).assign(PE=300.0)

    body = client.post("/api/batch", files=csv_upload(frame)).json()

    assert body["summary"]["alarm_hours"] == 0


def test_batch_reads_excel(client, small_bundle):
    buffer = io.BytesIO()
    small_bundle.example_batch.to_excel(buffer, index=False)
    files = {"file": ("hours.xlsx", buffer.getvalue(),
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    response = client.post("/api/batch", files=files)

    assert response.status_code == 200, response.text


def test_batch_rejects_too_many_columns(client):
    frame = pd.DataFrame({**TYPICAL, **{f"c{i}": 0 for i in range(60)}}, index=[0])

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 422


def test_broken_file_error_does_not_leak_internals(client):
    files = {"file": ("hours.xlsx", b"not an excel file", "application/octet-stream")}

    response = client.post("/api/batch", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Не удалось прочитать файл: нужен CSV или Excel"


def test_zero_price_is_respected(client):
    body = predict(client, V=52.0, actual_mw=430.0, price_rub_per_mwh=0)

    assert body["deviation"]["rub_per_hour"] == 0


def test_explain_can_be_switched_off(client):
    body = predict(client, explain=False)

    assert body["plan"]["contributions"] == []


def test_curve_marks_points_outside_training_data(client):
    curve = predict(client)["curve"]  # типичный час: края по температуре вне данных

    assert any(point["novel"] for point in curve)
    assert not all(point["novel"] for point in curve)


def test_batch_accepts_humidity_sensor_overshoot(client, small_bundle):
    frame = small_bundle.example_batch.assign(RH=100.16)

    response = client.post("/api/batch", files=csv_upload(frame))

    assert response.status_code == 200, response.text


def test_hourly_loss_is_counted_from_reference_not_from_plan(client):
    body = predict(client, V=52.0, actual_mw=440.0, price_rub_per_mwh=1000)

    deviation = body["deviation"]
    expected = max(0.0, -deviation["other_mw"]) * 1000
    assert deviation["rub_per_hour"] == pytest.approx(expected, abs=1)
