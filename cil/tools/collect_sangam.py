"""Run the frozen Sangam training or validation sessions and inspect every capture."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

CIL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CIL))
from tools.inspect_collection import inspect


def accepted_sessions(raw, manifest, route, offset, noisy):
    for folder in sorted(raw.glob(f'*_{route}_*'), reverse=True):
        path = folder / 'session.json'
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        if (meta['route_sha256'] == manifest[route]['route_sha256'] and meta['status'] == 'route_finished'
                and meta['initial_offset_m'] == offset and bool(meta['noise_enabled']) == noisy):
            if meta.get('collection_accepted') or inspect(folder, decode_all=False)['data_checks_passed']:
                yield folder, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--routes', nargs='+')
    noise_mode = parser.add_mutually_exclusive_group()
    noise_mode.add_argument('--recovery', action='store_true')
    noise_mode.add_argument('--noise', action='store_true')
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    manifest = json.loads((CIL / config['paths']['route_plan']).read_text())['routes']
    raw = CIL / config['paths']['raw_data']
    names = args.routes or [name for name, info in manifest.items() if info['split'] != 'test']
    noisy = args.recovery or args.noise
    if noisy and any(manifest[name]['split'] != 'train' for name in names):
        parser.error('recovery demonstrations are training only')
    if any(manifest[name]['split'] == 'test' for name in names):
        parser.error('test recordings cannot enter the collection')
    accepted, rejected = [], []
    for index, name in enumerate(names):
        route = manifest[name]
        offset = (-.35 if index % 2 else .35) if args.recovery else 0.
        item = next(accepted_sessions(raw, manifest, name, offset, noisy), None)
        if item is None:
            for speed in (18, 15):
                limit = math.ceil(route['length_m'] / ((speed - 1) / 3.6) * 1.55 + 16)
                before = set(raw.glob(f'*_{name}_*'))
                command = [sys.executable, 'collect.py', '--config', str(config_path), '--route', name,
                           '--auto-reset', '--speed', str(speed), '--seconds', str(limit),
                           '--initial-offset', str(offset)]
                if args.recovery:
                    command += ['--turn-recovery', '--seed', str(31 + index)]
                elif args.noise:
                    command += ['--noise', '--seed', str(67 + index)]
                print('CAPTURE', name, route['split'], speed, offset, flush=True)
                result = subprocess.run(command, cwd=CIL, check=False)
                candidates = sorted(set(raw.glob(f'*_{name}_*')) - before)
                if result.returncode == 0 and len(candidates) == 1:
                    folder = candidates[0]
                    report = inspect(folder, decode_all=False)
                    print('QUALITY', name, report['data_checks_passed'], report['errors'], flush=True)
                    if report['data_checks_passed']:
                        item = folder, json.loads((folder / 'session.json').read_text())
                        break
                rejected.append({'name': name, 'speed': speed, 'returncode': result.returncode,
                                 'sessions': [str(p.relative_to(CIL)) for p in candidates]})
        if item is not None:
            folder, meta = item
            accepted.append({'session': str(folder.relative_to(CIL)), 'split': meta['split']})
        data = {'status': 'recording', 'accepted': accepted, 'rejected_attempts': rejected}
        args.ledger.write_text(json.dumps(data, indent=2) + '\n')
    data['status'] = 'complete'
    args.ledger.write_text(json.dumps(data, indent=2) + '\n')
    print('FINISHED', len(accepted), '/', len(names), 'accepted', flush=True)


if __name__ == '__main__':
    main()
