#!/usr/bin/env python3
"""Reproduce the released task-order analysis from compact inputs."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'analysis' / 'scripts'


def run(name, *arguments):
    subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, arguments)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true',
                        help='Also generate the numerical PDF/SVG figures (requires Arial)')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate the supplied inputs/results without regenerating them')
    parser.add_argument('--preview-dir', type=Path,
                        help='Optional PNG preview folder, used with --figures')
    parser.add_argument('--report', type=Path,
                        help='Optional JSON validation report path')
    args = parser.parse_args()
    if args.check_only and args.figures:
        parser.error('--check-only and --figures cannot be combined')
    if args.preview_dir and not args.figures:
        parser.error('--preview-dir requires --figures')
    try:
        # Fail before rewriting outputs if any frozen input has changed.
        run('validate.py', '--inputs-only')
        if not args.check_only:
            run('recompute.py')
        validation_args = ['--report', args.report.resolve()] if args.report else []
        run('validate.py', *validation_args)
        if args.figures:
            figure_args = ['--preview-dir', args.preview_dir.resolve()] if args.preview_dir else []
            run('figures.py', *figure_args)
            run('task_distributions.py', *figure_args)
    except subprocess.CalledProcessError as error:
        return error.returncode
    print('Complete. Numerical results: analysis/results/.' +
          (' Figures: figures/.' if args.figures else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
