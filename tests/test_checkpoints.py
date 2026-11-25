"""Publication/recovery tests using JSON payloads: no PyTorch or datasets."""
import json
from pathlib import Path
import pytest
from src.checkpoints import CheckpointStore, IdentityMismatch, atomic_json


def writer(state, path):
    Path(path).write_text(json.dumps(state), encoding='utf-8')


def reader(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def store(root, identity=None):
    return CheckpointStore(root, identity or {'code': 'a', 'config': 'x', 'dataset': 'd'}, writer=writer, reader=reader)


def payload(saved, epoch):
    return {'identity': saved.identity, 'completed_epoch': epoch, 'value': epoch * 10}


def test_sequential_verified_publication_and_duplicate_refusal(tmp_path):
    saved = store(tmp_path)
    one = saved.save(payload(saved, 1))
    saved.save(payload(saved, 2))
    assert saved.load()[0]['value'] == 20
    assert (tmp_path / one['file']).exists()
    with pytest.raises(ValueError, match='next completed epoch'):
        saved.save(payload(saved, 2))
    with pytest.raises(ValueError):
        saved.save(payload(saved, 4))


def test_corrupt_latest_recovers_previous_and_can_republish(tmp_path):
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    last = saved.save(payload(saved, 2))
    (tmp_path / last['file']).write_bytes(b'truncated')
    assert saved.load()[0]['completed_epoch'] == 1
    saved.save(payload(saved, 2))
    assert saved.load()[0]['completed_epoch'] == 2


def test_corrupt_pointer_recovers_verified_backup(tmp_path):
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    saved.save(payload(saved, 2))
    saved.pointer.write_text('{')
    assert saved.load()[0]['completed_epoch'] == 1
    saved.save(payload(saved, 2))
    assert saved.load()[0]['completed_epoch'] == 2


def test_temporary_or_unpublished_payload_is_ignored(tmp_path):
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    (tmp_path / 'epoch_999.pt.tmp').write_bytes(b'incomplete')
    (tmp_path / 'orphan.pt').write_text(json.dumps(payload(saved, 2)))
    assert saved.load()[0]['completed_epoch'] == 1


def test_failed_writer_preserves_verified_previous_state(tmp_path):
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    def interrupted(state, path):
        Path(path).write_bytes(b'partial')
        raise OSError('Drive unavailable')
    saved.writer = interrupted
    with pytest.raises(OSError, match='Drive unavailable'):
        saved.save(payload(saved, 2))
    assert saved.load()[0]['completed_epoch'] == 1


def test_pointer_publication_failure_does_not_commit_orphan(tmp_path, monkeypatch):
    import src.checkpoints as module
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    original = module.atomic_json
    def fail(path, value):
        if Path(path) == saved.pointer:
            raise OSError('pointer publication interrupted')
        return original(path, value)
    monkeypatch.setattr(module, 'atomic_json', fail)
    with pytest.raises(OSError):
        saved.save(payload(saved, 2))
    assert saved.load()[0]['completed_epoch'] == 1
    monkeypatch.setattr(module, 'atomic_json', original)
    saved.save(payload(saved, 2))
    assert saved.load()[0]['completed_epoch'] == 2


def test_mismatch_is_rejected_before_any_payload_deserialisation(tmp_path):
    saved = store(tmp_path)
    saved.save(payload(saved, 1))
    other = store(tmp_path, {'code': 'different'})
    other.reader = lambda path: pytest.fail('Must not load incompatible payload')
    with pytest.raises(IdentityMismatch):
        other.load()
    with pytest.raises(IdentityMismatch):
        saved.save({'identity': {}, 'completed_epoch': 2})


def test_all_corrupt_generations_fail_clearly(tmp_path):
    saved = store(tmp_path)
    entry = saved.save(payload(saved, 1))
    (tmp_path / entry['file']).write_bytes(b'bad')
    with pytest.raises(ValueError, match='No verified checkpoint'):
        saved.load()


def test_json_rejects_nonfinite_before_replacing_existing(tmp_path):
    path = tmp_path / 'record.json'
    atomic_json(path, {'value': 1})
    with pytest.raises(ValueError):
        atomic_json(path, {'value': float('nan')})
    assert json.loads(path.read_text()) == {'value': 1}
