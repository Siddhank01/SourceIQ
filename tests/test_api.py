import backend_db
from fastapi.testclient import TestClient

from api import app


def test_models_are_explicitly_supported():
    response = TestClient(app).get('/api/models')
    assert response.status_code == 200
    assert 'openai/gpt-oss-120b' in response.json()['models']


def test_answer_rejects_unsupported_model():
    response = TestClient(app).post('/api/answer', data={'question': 'What is this?', 'model': 'not-a-groq-model'})
    assert response.status_code == 400
    assert 'Unsupported Groq model' in response.json()['detail']


def test_answer_rejects_missing_sources_before_generation(monkeypatch):
    monkeypatch.setattr('api.require_key', lambda: 'test-key')
    response = TestClient(app).post('/api/answer', data={'question': 'What is this?', 'model': 'openai/gpt-oss-20b'})
    assert response.status_code == 422
    assert 'source' in response.json()['detail']


def test_session_upsert_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_db, 'DB_PATH', tmp_path / 'test.sqlite3')
    payload = {'id': 'session-1', 'owner': 'tester@example.com', 'title': 'Research', 'description': '1 question', 'saved': True, 'messages': [{'role': 'user', 'text': 'Question'}], 'sources': []}
    client = TestClient(app)
    first = client.post('/api/sessions', json=payload)
    second = client.post('/api/sessions', json={**payload, 'title': 'Updated research'})
    sessions = client.get('/api/sessions', params={'owner': 'tester@example.com'}).json()
    deleted = client.delete('/api/sessions/session-1', params={'owner': 'tester@example.com'})
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(sessions) == 1
    assert sessions[0]['title'] == 'Updated research'
    assert deleted.status_code == 200
    assert client.get('/api/sessions/session-1', params={'owner': 'tester@example.com'}).status_code == 404
