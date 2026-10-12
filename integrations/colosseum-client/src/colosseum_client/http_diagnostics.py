"""HTTP diagnostics with field-only validation errors and no credentials/URLs."""
from http import HTTPStatus
import json
from pathlib import Path
import re
import tempfile


def _validation_errors(response):
    """Keep known field paths/types; never persist echoed input or server messages."""
    if response.status_code != 422:
        return []
    try:
        body = response.json()
    except (ValueError, TypeError):
        return []
    detail = body.get('detail') if isinstance(body, dict) else None
    if not isinstance(detail, list):
        return []
    fields = {'body', 'query', 'path', 'llm_api_urls', 'llm_providers', 'robot_id', 'track',
              'instruction', 'scene', 'evaluator', 'institution', 'test', 'recording',
              'inference_mode', 'max_steps', 'cameras', 'task_id'}
    kinds = {'literal_error', 'extra_forbidden', 'missing', 'string_type', 'int_type',
             'int_parsing', 'bool_type', 'bool_parsing', 'list_type', 'string_too_long',
             'greater_than', 'greater_than_equal', 'less_than', 'less_than_equal'}
    errors = []
    for item in detail[:20]:
        if not isinstance(item, dict) or not isinstance(item.get('loc'), (list, tuple)):
            continue
        kind = item.get('type')
        if not isinstance(kind, str) or kind not in kinds:
            continue
        location = [part if (isinstance(part, str) and part in fields)
                    or (type(part) is int and 0 <= part <= 10000) else '<field>'
                    for part in item['loc'][:8]]
        errors.append({'location': location, 'type': kind})
    return errors


def log_http_error(response, *, operation, log_dir=None):
    record = {'operation': operation, 'status': response.status_code}
    try:
        record['error'] = HTTPStatus(response.status_code).phrase
    except ValueError:
        record['error'] = 'HTTP request failed'
    validation = _validation_errors(response)
    if validation:
        record['validation_errors'] = validation
    # Do not serialize arbitrary headers, e.g. Location, Set-Cookie or Authorization.
    for header in ('x-request-id', 'x-amz-request-id'):
        value = response.headers.get(header, '')
        if re.fullmatch(r'[A-Za-z0-9._:+/=-]{1,200}', value):
            record[header] = value
    text = json.dumps(record, sort_keys=True)
    print(text, flush=True)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # Unique, private files also avoid overwriting errors within one second.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    prefix='http-error-', suffix='.json', dir=log_dir, delete=False) as output:
                output.write(text + '\n')
        except OSError:
            print('Could not save HTTP diagnostic; see the summary above.', flush=True)
    if validation:
        fields = ', '.join('.'.join(str(p) for p in item['location']) for item in validation)
        return record['error'] + ' (invalid fields: ' + fields + ')'
    return record['error']
