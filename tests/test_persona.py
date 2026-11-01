"""Conversation transport and history checks; no real provider or credentials."""
import json
import threading
from types import SimpleNamespace
import urllib.error
import pytest
from opendafeiyu.dialogue import DeepSeekClient, InteractionController, command_task, validate_task
from test_dialogue import built, runtime, advance


def client(tmp_path, **config):
    path = tmp_path/'private.json'
    path.write_text(json.dumps({'api_key': 'unit-test-placeholder', **config}), encoding='utf8')
    return DeepSeekClient(path)


def answer(task=None):
    return {'model': 'test-model', 'choices': [{'message': {'content': json.dumps(
        task or {'action': 'talk', 'target': 'none', 'reply': '鲸鱼朋友在这里。'})}}]}


def test_history_contains_only_accepted_bounded_turns(tmp_path, monkeypatch):
    c = client(tmp_path, history_turns=2)
    sent = []
    monkeypatch.setattr(c, '_request', lambda messages: sent.append(messages) or answer())
    c('尚未接受的请求')
    for text in ('旧的', '我喜欢海蓝色', '我喜欢草莓'):
        c.remember(text, command_task('wave'))
    c.respond('我喜欢什么？', {'supported_actions': ['talk', 'stop']})
    assert [m['content'] for m in sent[-1] if m['role'] == 'user'] == ['我喜欢海蓝色', '我喜欢草莓', '我喜欢什么？']
    assert 'runtime_context' in sent[-1][0]['content']
    c.clear_history(); c('重新开始')
    assert len(sent[-1]) == 2


def test_history_can_be_disabled_and_is_not_shared_between_clients(tmp_path, monkeypatch):
    first = client(tmp_path); first.remember('私人昵称', command_task('wave'))
    second = client(tmp_path, history_turns=0)
    second.remember('不保存', command_task('wave'))
    sent = []
    monkeypatch.setattr(second, '_request', lambda messages: sent.append(messages) or answer())
    second('你好')
    assert len(sent[0]) == 2


def test_schema_retry_is_bounded_and_does_not_echo_invalid_output(tmp_path, monkeypatch):
    c = client(tmp_path)
    sent = []
    def request(messages):
        sent.append([m.copy() for m in messages])
        return answer({'action': 'talk', 'none': 'untrusted-marker', 'reply': '你好'}) if len(sent)==1 else answer()
    monkeypatch.setattr(c, '_request', request)
    assert c('你好')['attempts'] == 2
    assert 'untrusted-marker' not in json.dumps(sent[-1])
    assert c._history == []
    sent.clear()
    def invalid(messages):
        sent.append(messages)
        return answer({'action': 'raw_joints', 'target': 'none', 'reply': 'x'})
    monkeypatch.setattr(c, '_request', invalid)
    with pytest.raises(RuntimeError, match='one format retry'): c('继续')
    assert len(sent) == 2


def test_http_failure_is_sanitized_and_not_retried(tmp_path, monkeypatch):
    c = client(tmp_path)
    calls = []
    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, 'unit-test-placeholder', {}, None)
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(RuntimeError, match='^API HTTP 401$'): c('你好')
    assert len(calls) == 1 and not c._history


def test_payload_keeps_json_transport_and_configurable_style(tmp_path, monkeypatch):
    c = client(tmp_path, temperature=.4, history_turns=99)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, _): return json.dumps(answer()).encode()
    payloads = []
    def request(req, **kwargs):
        payloads.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', request)
    assert c('说个笑话')['task']['action'] == 'talk'
    payload = payloads[0]
    assert payload['temperature'] == .4 and c.history_turns == 12
    assert payload['response_format'] == {'type': 'json_object'}
    assert 'unit-test-placeholder' not in json.dumps(payload)


@pytest.mark.parametrize('text', ['', '   ', 'x'*501, None])
def test_invalid_input_does_not_call_provider(tmp_path, monkeypatch, text):
    c = client(tmp_path)
    monkeypatch.setattr(c, '_request', lambda _: pytest.fail('Unexpected provider call'))
    with pytest.raises(ValueError): c(text)


def test_empty_reply_is_not_accepted():
    with pytest.raises(ValueError): validate_task({'action': 'talk', 'target': 'none', 'reply': ' '})


@pytest.mark.parametrize('command', ['新对话', '清空对话', 'New chat!'])
def test_new_conversation_discards_inflight_and_queued_history(tmp_path, monkeypatch, command):
    provider = client(tmp_path)
    provider.remember('我喜欢草莓', command_task('你好'))
    started, release = threading.Event(), threading.Event()
    sent = []
    def request(messages):
        sent.append(messages)
        if messages[-1]['content'] == '旧请求':
            started.set(); release.wait(5)
            return answer(command_task('摆尾'))
        return answer()
    monkeypatch.setattr(provider, '_request', request)
    low = SimpleNamespace(phase='ready', contacts=lambda: (True, True), cancel=lambda: None)
    control = InteractionController(low, provider)
    try:
        control.submit('旧请求', 0); control.poll(0)
        assert started.wait(2)
        old_future = control.pending[0][1]
        control.submit('尚未发送的请求', .1)
        control.submit(command, .2)
        assert not provider._history and control.queued is None and control.action is None
        assert control.stop_requested == .2
        release.set(); old_future.result(timeout=2); control.poll(.3)
        assert not provider._history and control.action is None
        assert any(e['event'] == 'stale_response_rejected' for e in control.events)
        control.submit('新的问候', .4); control.poll(.4)
        control.pending[0][1].result(timeout=2); control.poll(.5)
        assert len(sent) == 2
        assert [m['content'] for m in sent[-1] if m['role'] != 'system'] == ['新的问候']
        assert '主人' in sent[-1][0]['content']
        assert provider._history[0]['content'] == '新的问候'
        assert any(e['event'] == 'conversation_cleared' for e in control.events)
    finally:
        release.set(); control.close()


@pytest.mark.parametrize('text', ['不要清空对话', '请解释“新对话”是什么意思'])
def test_mentioning_new_conversation_preserves_history(tmp_path, monkeypatch, text):
    provider = client(tmp_path)
    provider.remember('我喜欢草莓', command_task('你好'))
    sent = []
    monkeypatch.setattr(provider, '_request', lambda messages: sent.append(messages) or answer())
    low = SimpleNamespace(phase='ready', contacts=lambda: (True, True), cancel=lambda: None)
    control = InteractionController(low, provider)
    try:
        control.submit(text, 0); control.poll(0)
        control.pending[0][1].result(timeout=2); control.poll(.1)
        assert [m['content'] for m in sent[0] if m['role'] == 'user'] == ['我喜欢草莓', text]
        assert not any(e['event'] == 'conversation_cleared' for e in control.events)
    finally: control.close()


def test_cancelled_response_never_enters_conversation_memory(built):
    release = threading.Event()
    started = threading.Event()
    class Translator:
        def __init__(self): self.history = []; self.contexts = []
        def respond(self, text, context):
            self.contexts.append(context)
            started.set(); release.wait(10)
            return {'task': command_task('tail'), 'mode': 'test_delayed_persona'}
        def remember(self, text, task): self.history.append((text, task))
    translator = Translator()
    m, d, c = runtime(built, translator)
    try:
        advance(m,d,c,3); c.submit('摆尾',float(d.time)); c.poll(float(d.time))
        assert started.wait(2)
        c.submit('停一下',float(d.time)); advance(m,d,c,1)
        release.set(); c.pending[0][1].result(timeout=2); c.poll(float(d.time))
        assert not translator.history and c.status == 'stopped'
        assert 'jump' not in translator.contexts[0]['supported_actions']
        c.submit('挥手',float(d.time)); c.poll(float(d.time))
        c.pending[0][1].result(timeout=2); c.poll(float(d.time))
        assert len(translator.history) == 1 and translator.history[0][0] == '挥手'
    finally: release.set(); c.close()
