"""Check repository-date provenance, lineage, and vocabulary deduplication."""
from pathlib import Path
import csv
import io
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from collect_hf_tokenizer_catalog import vocabulary_fingerprint
from export_hf_tokenizer_catalog import csv_text, earliest_record, enrich_curated, lineage, root_models


def test_creation_date_floor_is_not_a_release_or_first_record():
    model = {'id': 'test/model', 'createdAt': '2022-03-02T23:29:04.000Z'}
    assert earliest_record(model, {}) == (None, None, None)
    history = {'status': 'complete', 'earliest_commit': {'date': '2019-11-05T12:00:00.000Z'},
               'source_url': 'https://huggingface.co/test/model/commit/abc'}
    assert earliest_record(model, history) == (
        '2019-11-05T12:00:00.000Z', 'earliest_hf_commit', history['source_url'])


def test_repository_creation_is_labeled_as_a_proxy_not_release():
    result = earliest_record({'id': 'test/model', 'createdAt': '2024-01-01T00:00:00.000Z'}, {})
    assert result[1] == 'hf_repository_creation'
    assert result[2] == 'https://huggingface.co/api/models/test/model'


def test_exact_release_takes_priority_and_proxy_preserves_partial_release():
    rows = [{'model_id': 'test/exact', 'release_date': '2024-02-03', 'release_date_precision': 'day',
             'release_date_source_url': 'https://example.org/release'},
            {'model_id': 'test/partial', 'release_date': '2020', 'release_date_precision': 'year',
             'release_date_source_url': 'https://example.org/paper'}]
    models = {row['model_id']: {'id': row['model_id'], 'createdAt': '2024-01-01T00:00:00.000Z'} for row in rows}
    exact, proxy = enrich_curated(rows, models, {})
    assert exact['timeline_date'] == '2024-02-03'
    assert exact['timeline_date_basis'] == 'documented_release'
    assert proxy['timeline_date'] == '2024-01-01'
    assert proxy['timeline_date_basis'] == 'hf_repository_creation'
    assert proxy['release_date'] == '2020'
    assert proxy['release_date_precision'] == 'year'


def test_bulk_csv_preserves_arrays_booleans_missing_and_zero():
    rows = [{'parents': ['org/a', 'org/b'], 'cycle': False, 'missing': None, 'zero': 0}]
    parsed = list(csv.DictReader(io.StringIO(csv_text(rows))))
    assert parsed == [{'parents': '["org/a","org/b"]', 'cycle': 'false', 'missing': '', 'zero': '0'}]


def test_lineage_preserves_multiple_parents_and_relationships():
    model = {'cardData': {'base_model': ['org/a'], 'base_model_relation': 'merge'},
             'tags': ['base_model:org/a', 'base_model:quantized:org/b'], 'library_name': 'gguf'}
    parents, relationships, _ = lineage(model)
    assert parents == ['org/a', 'org/b']
    assert set(relationships) == {'merge', 'quantized', 'quantized_or_gguf_conversion'}
    assert lineage({'id': 'org/unknown'})[1] == ['not_declared']


def test_declared_ancestry_stops_at_unknown_parent_and_detects_cycles():
    graph = {'org/a': ['org/b'], 'org/b': ['org/c']}
    assert root_models('org/a', graph) == (['org/c'], False)
    graph['org/c'] = ['org/a']
    assert root_models('org/a', graph) == ([], True)


def test_vocabulary_hash_ignores_serialization_and_merges_but_keeps_ids():
    first = {'model': {'vocab': {'a': 0, 'b': 2}, 'merges': []},
             'added_tokens': [{'id': 3, 'content': '[END]'}]}
    equivalent = {'model': {'vocab': {'b': 2, 'a': 0}, 'merges': ['a b']},
                  'added_tokens': [{'content': '[END]', 'id': 3, 'special': True}]}
    assert vocabulary_fingerprint(first) == vocabulary_fingerprint(equivalent)
    equivalent['model']['vocab']['b'] = 1
    assert vocabulary_fingerprint(first) != vocabulary_fingerprint(equivalent)


def test_vocabulary_hash_rejects_conflicting_token_ids():
    document = {'model': {'vocab': {'a': 0}}, 'added_tokens': [{'id': 0, 'content': 'b'}]}
    with pytest.raises(ValueError, match='Conflicting'):
        vocabulary_fingerprint(document)
    with pytest.raises(ValueError, match='Conflicting'):
        vocabulary_fingerprint({'model': {'vocab': {'a': 0, 'b': 0}}})
