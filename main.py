"""InSITE Engine command line.

    python main.py run                 # deterministic pipeline only (also the default)
    python main.py demo                # write synthetic demo data into input_path
    python main.py think [--offline]   # one autonomous cognitive cycle of the Mind
    python main.py ask "question"      # point the Mind at a specific question
    python main.py live                # keep thinking: cycle on a schedule / on new data
"""
import argparse
import json
import logging
import sys

import yaml

DEFAULT_CONFIG = 'InSITE_Engine/config/config.yaml'


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_run(cfg, args):
    from InSITE_Engine.pipeline import run_pipeline

    result = run_pipeline(cfg)
    written = result.save(cfg['output_path'])
    summary = result.summary()
    print(json.dumps({k: summary[k] for k in ('event_count', 'events_by_kind', 'events_by_severity')}, indent=2))
    print(f'{len(result.datasets)} datasets processed; {len(written)} files written to {cfg["output_path"]}')
    print('InSITE Engine completed successfully.')
    return 0


def cmd_demo(cfg, args):
    from InSITE_Engine.demo.generate import generate_demo_data

    truth = generate_demo_data(cfg['input_path'], days=args.days, seed=args.seed)
    print(f'Demo data written to {cfg["input_path"]} ({len(truth.get("scenarios", truth))} scenarios embedded).')
    return 0


def _mind(cfg, args):
    from InSITE_Engine.mind import Mind

    if getattr(args, 'effort', None):
        cfg['mind']['effort'] = args.effort
    return Mind(cfg)


def _print_outcome(outcome):
    print()
    print(f'Cycle {outcome.cycle}: {outcome.status}, cost ${outcome.cost_usd:.2f}')
    if outcome.summary:
        print()
        print(outcome.summary)
    if outcome.report_path:
        print()
        print(f'Report: {outcome.report_path}')


def cmd_think(cfg, args):
    from InSITE_Engine.mind import Mind, MindUnavailable

    if args.offline:
        _print_outcome(Mind(cfg, llm=False).think(offline=True))
        return 0
    try:
        mind = _mind(cfg, args)
    except MindUnavailable as e:
        print(f'The Mind is unavailable ({e}); running a perception-only cycle instead.', file=sys.stderr)
        _print_outcome(Mind(cfg, llm=False).think(offline=True))
        return 0
    _print_outcome(mind.think())
    return 0


def cmd_ask(cfg, args):
    from InSITE_Engine.mind import MindUnavailable

    try:
        mind = _mind(cfg, args)
    except MindUnavailable as e:
        print(f'The Mind is unavailable: {e}', file=sys.stderr)
        return 2
    _print_outcome(mind.ask(args.question))
    return 0


def cmd_live(cfg, args):
    from InSITE_Engine.mind import MindUnavailable

    try:
        mind = _mind(cfg, args)
    except MindUnavailable as e:
        print(f'The Mind is unavailable: {e}', file=sys.stderr)
        return 2
    mind.live(interval_seconds=args.interval, max_cycles=args.max_cycles, budget_usd_total=args.budget_usd)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description='InSITE Engine')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('run', help='run the deterministic pipeline and write its outputs')

    demo = sub.add_parser('demo', help='generate synthetic demo data')
    demo.add_argument('--days', type=int, default=7)
    demo.add_argument('--seed', type=int, default=7)

    effort = argparse.ArgumentParser(add_help=False)
    effort.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])

    think = sub.add_parser('think', parents=[effort], help='run one cognitive cycle')
    think.add_argument('--offline', action='store_true', help='perception only, no model calls')

    ask = sub.add_parser('ask', parents=[effort], help='ask the Mind a question about the site')
    ask.add_argument('question')

    live = sub.add_parser('live', parents=[effort], help='run cycles autonomously')
    live.add_argument('--interval', type=float, help='seconds between cycles (default from config)')
    live.add_argument('--max-cycles', type=int)
    live.add_argument('--budget-usd', type=float, help='stop once total spend reaches this')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.get('log_level', 'INFO'), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    commands = {'run': cmd_run, 'demo': cmd_demo, 'think': cmd_think, 'ask': cmd_ask, 'live': cmd_live}
    return commands[args.command or 'run'](cfg, args)


if __name__ == '__main__':
    sys.exit(main())
