#!/usr/bin/env python3
"""Collect DS1 KNN run outputs on vm03 without touching the training directory.

Usage:
 python tools/collect_vm03.py --run-root ~/knn_fix_runs/ds1_knn_free_gpu_20261123_093107 \
   --source-root ~/phylogfn_knn_vectorized_ds1_20261123_013714 \
   --destination results/network/DS1/knn_k5_vectorized_20261123

Only run this with trusted local pickle files; pickle.load executes arbitrary Python.
"""
import argparse, csv, hashlib, json, os, pickle, re, shutil, sys
from pathlib import Path

PAT = re.compile(r'Epoch\s+(\d+):\s+MLL\s+([-+\d.eE]+)\s+PearsonR\s+([-+\d.eE]+)\s+mean R\s+([-+\d.eE]+)\s+R histogram\s+\[(\d+),\s*(\d+),\s*(\d+)\]')
FINAL = re.compile(r'Final MLL:\s+mean\s+([-+\d.eE]+)\s+std\s+([-+\d.eE]+)\s+\((\d+) estimates of (\d+) trajectories\)')


def write_csv(dst, rows, fieldnames):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-root',required=True,type=Path)
    ap.add_argument('--source-root',required=True,type=Path)
    ap.add_argument('--destination',required=True,type=Path)
    ap.add_argument('--include-source-snapshot',action='store_true')
    ap.add_argument('--hash-checkpoints',action='store_true',help='Slow: SHA256 each full checkpoint, off by default')
    ap.add_argument('--include-raw-pickle',action='store_true',help='Include original trusted pickle files after reviewing public-release suitability')
    a=ap.parse_args()
    root=a.run_root.expanduser().resolve(); src=a.source_root.expanduser().resolve(); dst=a.destination.expanduser().resolve()
    if not root.is_dir(): ap.error(f'Run root missing: {root}')
    if not src.is_dir(): ap.error(f'Source root missing: {src}')
    if root==dst or root in dst.parents or dst in root.parents:ap.error('Destination must be separate from training directory')
    if dst==src or src in dst.parents or dst in src.parents:ap.error('Destination must be separate from source tree')
    dst.mkdir(parents=True,exist_ok=True); (dst/'raw').mkdir(exist_ok=True);(dst/'data').mkdir(exist_ok=True)
    logs=sorted(root.rglob('train.log'))
    # Prefer the outer tee log, which was set to <run-root>/train.log.
    outer=root/'train.log'
    if outer.exists():log=outer
    elif logs:log=max(logs,key=lambda p:p.stat().st_mtime)
    else:log=None
    home=Path.home().resolve()
    def public_path(p):
        p=Path(p)
        try:return '$HOME/'+str(p.relative_to(home))
        except ValueError:return str(p)
    summary={'run_root':public_path(root),'source_root':public_path(src),'status':'incomplete export until all referenced files are present','files':{}}
    if log:
        s=log.read_text(encoding='utf-8',errors='replace')
        shutil.copy2(log,dst/'raw/train.log')
        epochs={}
        for m in PAT.finditer(s):
            epoch,mll,pearson,mean_r,r0,r1,r2=m.groups()
            epochs[int(epoch)]={'epoch':int(epoch),'mll':float(mll),'pearson_r':float(pearson),'mean_reticulations':float(mean_r),'r0':int(r0),'r1':int(r1),'r2':int(r2)}
        if epochs:
            records=[epochs[k] for k in sorted(epochs)]
            write_csv(dst/'data/epochs_from_full_train_log.csv',records,list(records[0]))
            summary['logged_epoch_count']=len(records)
            summary['latest_logged_epoch']=max(epochs)
        finals=FINAL.findall(s)
        if finals:
            mean,sd,repeats,n=finals[-1]
            summary['logged_final_mll']={'mean':float(mean),'std':float(sd),'repeats':int(repeats),'trajectories_per_repeat':int(n)}
        summary['files']['train_log']=public_path(log)
    else:print('WARNING: train.log not found',file=sys.stderr)

    hist_paths=list(root.rglob('history.pkl'))
    if hist_paths:
        p=max(hist_paths,key=lambda x:x.stat().st_mtime)
        # Copy raw pickle for archival provenance; read it ONLY because it is our trusted run file.
        if a.include_raw_pickle and p.stat().st_size <= 10*1024*1024: shutil.copy2(p,dst/'raw/history.pkl')
        elif a.include_raw_pickle: print('WARNING: history.pkl exceeds 10MB; not copied',file=sys.stderr)
        with p.open('rb') as f: history=pickle.load(f)
        if not isinstance(history,list):raise RuntimeError(f'Expected list of epoch dicts, got {type(history).__name__}')
        acc=sum(int(r.get('n_updates_accepted',0)) for r in history)
        rej=sum(int(r.get('n_updates_rejected',0)) for r in history)
        summary.update(history_epochs=len(history),accepted_updates=acc,rejected_updates=rej)
        epoch_rows=[]
        for idx,r in enumerate(history):
            epoch_rows.append({'epoch':r.get('epoch',idx),'accepted':r.get('n_updates_accepted'),
                               'rejected':r.get('n_updates_rejected'),'temperature':r.get('temperature'),
                               'loss_mean':r.get('loss_mean')})
        write_csv(dst/'data/update_history.csv',epoch_rows,['epoch','accepted','rejected','temperature','loss_mean'])
        summary['files']['history']=public_path(p)
    else:print('WARNING: history.pkl not found',file=sys.stderr)

    final_paths=list(root.rglob('final_mlls.p'))
    if final_paths:
        fp=max(final_paths,key=lambda x:x.stat().st_mtime)
        if a.include_raw_pickle and fp.stat().st_size <= 10*1024*1024:shutil.copy2(fp,dst/'raw/final_mlls.p')
        with fp.open('rb') as f: estimates=pickle.load(f)
        if isinstance(estimates,(tuple,list)):
            (dst/'data/final_mll_estimates.json').write_text(json.dumps({'estimates':[float(x) for x in estimates]},indent=2)+'\n')
        summary['files']['final_mlls']=public_path(fp)

    cfg_paths=list(root.rglob('config.yaml'))
    if not cfg_paths:
        historical=Path.home()/'phylogfn_proposal_20260929/phylogfn/runs/20260929_192648_ds1_knn_only_300ep_final/config.yaml'
        if historical.exists():cfg_paths=[historical]
    if cfg_paths:
        cfg=cfg_paths[0];shutil.copy2(cfg,dst/'raw/config.yaml');summary['files']['config_yaml']=public_path(cfg)
    else:print('WARNING: config.yaml not found',file=sys.stderr)

    checkpoints=[]
    for p in root.rglob('checkpoint_*.pt'):
        row={'name':p.name,'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size,'modified_time':p.stat().st_mtime}
        if a.hash_checkpoints:row['sha256']=sha256(p)
        checkpoints.append(row)
    checkpoints.sort(key=lambda x:x['name'])
    (dst/'data/checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
    summary['checkpoint_count']=len(checkpoints)

    rels=[
        'train_net.py','src/env/phylo_network_env.py','src/env/__init__.py','src/env/fast_features.py',
        'src/utils/knn.py','src/utils/update_monitor.py','src/gfn/tb_gfn_net.py',
        'src/gfn/rollout_worker_net.py','src/model/network_model/one_step_network_model.py',
        'src/model/edges_model/continuous/network_heads.py',
        'src/model/edges_model/continuous/continuous_mixture.py']
    provenance={}
    for rel in rels:
        p=src/rel
        if p.is_file():
            provenance[rel]={'sha256':sha256(p),'size_bytes':p.stat().st_size}
            if a.include_source_snapshot:
                dest=dst/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (dst/'data/source_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summary['source_snapshot_included']=bool(a.include_source_snapshot)
    if summary.get('history_epochs')==300 and summary.get('accepted_updates')==60000 and summary.get('rejected_updates')==0:
        summary['status']='complete run confirmed from local history'
    (dst/'data/vm03_export_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))
    print('Exported to:',dst)
    print('NOTE: No checkpoints or DNA dataset copied; inspect all exported files before publishing.')

if __name__=='__main__':main()
