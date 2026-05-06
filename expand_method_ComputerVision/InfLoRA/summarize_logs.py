#!/usr/bin/env python3
import argparse
import ast
import glob
import re
import statistics
import sys
from pathlib import Path


CURVE_PATTERNS = {
    'top1': re.compile(r'CNN top1 curve:\s*(\[.*\])'),
    'top1_with_task': re.compile(r'CNN top1 with task curve:\s*(\[.*\])'),
    'task_curve': re.compile(r'CNN top1 task curve:\s*(\[.*\])'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize InfLoRA/SRT_InfLoRA log files into a compact comparison table.'
    )
    parser.add_argument(
        '--log',
        action='append',
        default=[],
        help='Specific log file to parse. Can be passed multiple times.',
    )
    parser.add_argument(
        '--glob',
        action='append',
        default=[],
        help='Glob pattern for log files, e.g. "logs/cifar100/**/*.log". Can be passed multiple times.',
    )
    parser.add_argument(
        '--reference',
        action='append',
        default=[],
        help='Reference score to compare against final top1, formatted as NAME=VALUE.',
    )
    return parser.parse_args()


def collect_log_paths(args):
    paths = {Path(path).resolve() for path in args.log}
    for pattern in args.glob:
        for path in glob.glob(pattern, recursive=True):
            if path.endswith('.log'):
                paths.add(Path(path).resolve())
    return sorted(paths)


def parse_curve(text, pattern):
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        curve = ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f'Could not parse curve {matches[-1]!r}: {exc}') from exc
    if not isinstance(curve, list):
        raise ValueError(f'Parsed curve is not a list: {curve!r}')
    return [float(value) for value in curve]


def normalize_task_curve(curve):
    if curve is None:
        return None
    if curve and max(abs(value) for value in curve) <= 1.000001:
        return [100.0 * value for value in curve]
    return curve


def parse_log(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    top1 = parse_curve(text, CURVE_PATTERNS['top1'])
    top1_with_task = parse_curve(text, CURVE_PATTERNS['top1_with_task'])
    task_curve = normalize_task_curve(parse_curve(text, CURVE_PATTERNS['task_curve']))

    if top1 is None or top1_with_task is None or task_curve is None:
        raise ValueError('Missing one or more summary curves in log file.')

    return {
        'path': path,
        'seed': path.stem,
        'n_tasks': len(top1),
        'final_top1': top1[-1],
        'final_top1_with_task': top1_with_task[-1],
        'final_task_acc': task_curve[-1],
        'avg_top1_over_tasks': sum(top1) / len(top1),
        'avg_top1_with_task_over_tasks': sum(top1_with_task) / len(top1_with_task),
        'avg_task_acc_over_tasks': sum(task_curve) / len(task_curve),
    }


def parse_references(reference_args):
    references = []
    for item in reference_args:
        if '=' not in item:
            raise ValueError(f'Invalid reference {item!r}. Use NAME=VALUE.')
        name, value = item.split('=', 1)
        references.append((name.strip(), float(value.strip())))
    return references


def format_stat(values):
    if not values:
        return 'n/a'
    if len(values) == 1:
        return f'{values[0]:.2f}'
    return f'{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}'


def print_table(rows):
    headers = [
        'seed', 'tasks', 'final_top1', 'final_with_task', 'final_task_acc', 'avg_top1', 'log',
    ]
    table_rows = [
        [
            row['seed'],
            str(row['n_tasks']),
            f"{row['final_top1']:.2f}",
            f"{row['final_top1_with_task']:.2f}",
            f"{row['final_task_acc']:.2f}",
            f"{row['avg_top1_over_tasks']:.2f}",
            str(row['path']),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(parts):
        return '  '.join(value.ljust(widths[index]) for index, value in enumerate(parts))

    print(render(headers))
    print(render(['-' * width for width in widths]))
    for row in table_rows:
        print(render(row))


def print_aggregate(rows, references):
    final_top1 = [row['final_top1'] for row in rows]
    final_top1_with_task = [row['final_top1_with_task'] for row in rows]
    final_task_acc = [row['final_task_acc'] for row in rows]
    avg_top1 = [row['avg_top1_over_tasks'] for row in rows]

    print('\nAggregate')
    print(f'  final_top1:       {format_stat(final_top1)}')
    print(f'  final_with_task:  {format_stat(final_top1_with_task)}')
    print(f'  final_task_acc:   {format_stat(final_task_acc)}')
    print(f'  avg_top1:         {format_stat(avg_top1)}')

    if references:
        ref_anchor = statistics.mean(final_top1)
        print('\nReferences vs final_top1')
        for name, value in references:
            gap = ref_anchor - value
            print(f'  {name}: reference={value:.2f}, gap={gap:+.2f}')


def main():
    args = parse_args()
    paths = collect_log_paths(args)
    if not paths:
        print('No log files found. Pass --log or --glob.', file=sys.stderr)
        sys.exit(1)

    try:
        references = parse_references(args.reference)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = []
    for path in paths:
        try:
            rows.append(parse_log(path))
        except ValueError as exc:
            skipped.append((path, str(exc)))

    if not rows:
        print('No complete log files could be parsed.', file=sys.stderr)
        for path, reason in skipped:
            print(f'  skipped {path}: {reason}', file=sys.stderr)
        sys.exit(1)

    if skipped:
        for path, reason in skipped:
            print(f'[warn] skipped {path}: {reason}', file=sys.stderr)

    print_table(rows)
    print_aggregate(rows, references)


if __name__ == '__main__':
    main()
