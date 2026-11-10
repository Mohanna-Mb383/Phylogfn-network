"""
make_paper_method_cfgs.py -- write the configurations of the da Silva et al. (2025) method comparison.

    python tools/make_paper_method_cfgs.py

Writes src/configs/network_cfgs/paper/:
  sim6_{tb,db,wdb_logcount,wdb_count,wdb_depth,wdb_logcount_la}.yaml   (N = 6 gate problem, 60 epochs x 50 steps)
  n4_{tb,db,wdb_logcount,wdb_count}.yaml                                (N = 4 reference problem, 40 epochs x 50 steps)
  DS{1..8}_{db,wdb_logcount}.yaml                                       (identical to cfg_net_DS1_300ep.yaml except the
                                                                         objective; DS7/DS8 with MINI_BATCH_SPLITS: 4)
TB on DS1-DS8 is the existing baseline run (cfg_net_DS1_300ep.yaml); it is not re-run.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.configs.defaults import get_cfg_defaults

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'src/configs/network_cfgs/paper')
os.makedirs(OUT, exist_ok=True)
VARIANTS = {
    'tb': dict(NET_LOSS='TB', WDB_GAMMA='uniform', LA=False),
    'db': dict(NET_LOSS='DB', WDB_GAMMA='uniform', LA=False),
    'wdb_logcount': dict(NET_LOSS='WDB', WDB_GAMMA='logcount', LA=False),
    'wdb_count': dict(NET_LOSS='WDB', WDB_GAMMA='count', LA=False),
    'wdb_depth': dict(NET_LOSS='WDB', WDB_GAMMA='depth', LA=False),
    'wdb_logcount_la': dict(NET_LOSS='WDB', WDB_GAMMA='logcount', LA=True),
}


def write(name, base, epochs=None, steps=None, splits=None, r_max=None, depth=None, anneal=None, variant='tb'):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(ROOT, base))
    v = VARIANTS[variant]
    cfg.GFN.NET_LOSS = v['NET_LOSS']; cfg.GFN.WDB_GAMMA = v['WDB_GAMMA']; cfg.GFN.MODEL.LOOKAHEAD.ENABLED = v['LA']
    if epochs is not None:
        cfg.GFN.TRAINING_DATA_LOADER.EPOCHS_NUM = epochs
    if steps is not None:
        cfg.GFN.TRAINING_DATA_LOADER.STEPS_PER_EPOCH = steps
    if splits is not None:
        cfg.GFN.TRAINING_DATA_LOADER.MINI_BATCH_SPLITS = splits
    if r_max is not None:
        cfg.ENV.NETWORK.R_MAX = r_max
    if depth is not None:
        cfg.GFN.MODEL.TRANSFORMER.DEPTH = depth
    if anneal is not None:
        cfg.GFN.TRAINING_DATA_LOADER.TEMPERATURE_ANNEALING.TEMPERATURE_ANNEALING = anneal
    path = os.path.join(OUT, name + '.yaml')
    cfg.dump(stream=open(path, 'w'))
    return path


paths = []
for variant in VARIANTS:
    paths.append(write('sim6_' + variant, 'src/configs/network_cfgs/cfg_net_small_cpu.yaml', epochs=60, steps=50, depth=2, anneal=False, variant=variant))
for variant in ('tb', 'db', 'wdb_logcount', 'wdb_count'):
    paths.append(write('n4_' + variant, 'src/configs/network_cfgs/cfg_net_small_cpu.yaml', epochs=40, steps=50, depth=2, anneal=False, r_max=2, variant=variant))
for d in range(1, 9):
    for variant in ('db', 'wdb_logcount'):
        paths.append(write('DS%d_%s' % (d, variant), 'src/configs/network_cfgs/cfg_net_DS1_300ep.yaml', splits=(4 if d >= 7 else None), variant=variant))
print('\n'.join(os.path.relpath(p, ROOT) for p in paths))
