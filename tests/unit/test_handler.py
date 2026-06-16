from unittest.mock import MagicMock, patch

# Patch module-level AWS clients before the app module is imported
with patch("boto3.client", return_value=MagicMock()), \
     patch("textractor.Textractor", return_value=MagicMock()), \
     patch("anthropic.AsyncAnthropicBedrock", return_value=MagicMock()):
    from twins_textract import app as app_module

from fastapi.testclient import TestClient

client = TestClient(app_module.app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"app": "twins-document-classifier", "status": "running"}


def test_classify_missing_input_uri():
    response = client.post("/api/classify-document", json={})
    assert response.status_code == 400
    assert "input_pdf_s3_uri" in response.json()["detail"]


def test_classify_missing_output_uri():
    response = client.post(
        "/api/classify-document",
        json={"input_pdf_s3_uri": "s3://bucket/test.pdf"},
    )
    assert response.status_code == 400
    assert "output_prefix_s3_uri" in response.json()["detail"]