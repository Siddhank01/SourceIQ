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
    monkeypatch.setattr('api.run_self_rag', lambda *args, **kwargs: {
        'final_answer': 'I cannot provide a reliable answer from the available evidence.',
        'status': 'Abstain',
        'verification_status': 'abstained',
        'abstain_reason': 'No safe evidence was retrieved.',
        'reflection_trace': [{'event': 'abstain', 'reason': 'No safe evidence was retrieved.'}],
        'retrieved_documents': [],
        'sources': [],
    })
    response = TestClient(app).post('/api/answer', data={'question': 'What is this?', 'model': 'openai/gpt-oss-20b'})
    payload = response.json()
    assert response.status_code == 200
    assert payload['status'] == 'Abstain'
    assert payload['verification_status'] == 'abstained'
    assert payload['reflection_trace'][0]['event'] == 'abstain'


def test_answer_exposes_workflow_trace(monkeypatch):
    monkeypatch.setattr('api.run_self_rag', lambda *args, **kwargs: {
        'final_answer': 'The answer is supported [chunk-a].',
        'status': 'Final answer',
        'verification_status': 'verified',
        'confidence': 0.91,
        'reflection_trace': [{'event': 'retrieval_decision'}, {'event': 'passage_relevance'}, {'event': 'claim_verification'}],
        'retrieved_documents': [{'id': 'chunk-a', 'content': 'evidence', 'metadata': {}}],
        'sources': [{'id': 'chunk-a'}],
        'passage_relevance_results': [{'passage_id': 'chunk-a', 'relevant': True, 'score': 0.91}],
        'answer_support_verification': {'supported': True},
        'retrieval_attempts': 1,
        'generation_attempts': 1,
    })
    response = TestClient(app).post('/api/answer', data={'question': 'What is this?', 'model': 'openai/gpt-oss-20b'})
    payload = response.json()
    assert response.status_code == 200
    assert payload['citations'] == ['chunk-a']
    assert len(payload['reflection_trace']) == 3
    assert payload['passage_relevance_results'][0]['passage_id'] == 'chunk-a'


def test_session_upsert_and_delete(tmp_path, monkeypatch):
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
