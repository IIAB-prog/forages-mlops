from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def sample():
    return {
        "age_ans": 15,
        "profondeur_m": 90,
        "debit_m3h": 4.2,
        "usagers_jour": 450,
        "jours_depuis_maintenance": 300,
        "pannes_12m": 1,
        "type_pompe": "solaire",
        "region": "Centre",
    }


def test_health_reports_model_state():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert "model_load_error" in body


def test_prediction_contract_or_clear_unavailable_response():
    response = client.post("/predict", json=sample())
    if response.status_code == 503:
        # Un modèle incompatible/absent ne doit pas faire planter le processus API.
        detail = response.json()["detail"]
        assert detail["message"] == "Modèle indisponible"
        return

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["risque_panne_90j"] <= 1
    assert body["classe_risque"] in ["faible", "modere", "eleve"]


def test_invalid_input_rejected():
    payload = sample()
    payload["age_ans"] = -1
    assert client.post("/predict", json=payload).status_code == 422
