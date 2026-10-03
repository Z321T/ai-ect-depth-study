"""Run all frozen E2 seeds, independent proofs and fixed BN ablation in order."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def execution_steps(registry, registry_path, device='auto'):
    folder=Path(registry_path).parent
    steps=[]
    for row in sorted(registry['runs'],key=lambda r:r['seed']):
        seed=row['seed'];run=row['output_dir']
        steps.append(dict(kind='train',seed=seed,output=run,command=[
            'scripts/train_paper_dc_ac.py','--config',row['config_path'],'--registry',registry_path,
            '--output',run,'--device',device]))
        proof=(folder/'verification'/f"{row['run_id']}_cuda.json").as_posix()
        steps.append(dict(kind='verify',seed=seed,output=proof,command=[
            'scripts/verify_paper_dc_ac.py','--run',run,'--device',device,'--output',proof]))
    for row in sorted(registry['runs'],key=lambda r:r['seed']):
        seed=row['seed'];run=(folder/'bn'/row['run_id']).as_posix()
        steps.append(dict(kind='bn',seed=seed,output=run,command=[
            'scripts/recalibrate_paper_dc_ac_bn.py','--origin',row['output_dir'],
            '--output',run,'--device',device]))
        proof=(folder/'verification'/f"{row['run_id']}_bn_cuda.json").as_posix()
        steps.append(dict(kind='verify_bn',seed=seed,output=proof,command=[
            'scripts/verify_paper_dc_ac.py','--run',run,'--device',device,'--output',proof]))
    return steps


def ensure_fresh_execution(root,folder,steps):
    targets=[folder/name for name in ('execution_status.json','controller.running','logs','verification','analysis_results_v1')]
    targets += [root/step['output'] for step in steps]
    for path in targets:
        if path.exists():raise FileExistsError(f'Controller never overwrites or resumes partial training: {path}')


def write_json(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',newline='\n')
    temporary.replace(path)


def summarize(root, folder, registry):
    """Use completed, independently checked reports; never run inference here."""
    import numpy as np
    from src.ect.integrity import file_sha256
    from scripts.verify_paper_run import read_json
    output=folder/'analysis_results_v1';output.mkdir(exist_ok=False)
    inputs={}
    def report(path):
        path=Path(path)
        value=read_json(root/path/'report.json')
        if value['status']!='complete' or value['heldout_test_evaluated'] is not False:
            raise ValueError('Complete train/validation report required')
        inputs[(path/'report.json').as_posix()]=file_sha256(root/path/'report.json')
        return value
    rows=[];groups=[];classes=[];curves=[];histories=[];cells=[]
    for seed in (0,1,2):
        run=f'resnext_s{seed}'
        paths=[('original','original_bn',Path('results/experiments/paper_baseline_v1')/run),
               ('original','calibrated_bn',Path('results/experiments/paper_bn_recalibration_v1')/run),
               ('dc_ac','original_bn',folder.relative_to(root)/run),
               ('dc_ac','calibrated_bn',folder.relative_to(root)/'bn'/run)]
        baseline=None
        for input_kind,bn_kind,path in paths:
            r=report(path)
            if input_kind=='original' and bn_kind=='original_bn': baseline=r
            if input_kind=='dc_ac':
                proof=folder/'verification'/f'{run}{"_bn" if bn_kind=="calibrated_bn" else ""}_cuda.json'
                p=read_json(proof)
                if p['status']!='verified' or p['report_sha256']!=file_sha256(root/path/'report.json'):
                    raise ValueError('Full E2 proof required for summary')
                inputs[proof.relative_to(root).as_posix()]=file_sha256(proof)
            cell=dict(seed=seed,input=input_kind,bn=bn_kind,run_dir=path.as_posix(),selected_epoch=r['selected_epoch'])
            cells.append(cell|dict(metrics=r['metrics']))
            for split in ('train','validation'):
                metric=r['metrics'][split]
                rows.append(cell|dict(split=split,accuracy=metric['accuracy'],macro_f1=metric['macro_f1'],
                    delta_accuracy_vs_original=metric['accuracy']-baseline['metrics'][split]['accuracy']))
                for g,m in r['group_metrics'][split].items():
                    groups.append(cell|dict(split=split,group_id=g,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
                for c,m in metric['per_class'].items():
                    classes.append(cell|dict(split=split,class_index=c,**m))
            if bn_kind=='original_bn':
                e=r['epochs'];v=[x for x in e if x['validation'] is not None]
                avg=lambda entries,key:float(np.mean([x[key] for x in entries]))
                history=cell|dict(first100_train_loss=avg(e[:100],'train_loss'),last100_train_loss=avg(e[-100:],'train_loss'),
                    first100_online_accuracy=avg(e[:100],'online_train_accuracy'),last100_online_accuracy=avg(e[-100:],'online_train_accuracy'),
                    first10_validation_accuracy=float(np.mean([x['validation']['accuracy'] for x in v[:10]])),
                    last10_validation_accuracy=float(np.mean([x['validation']['accuracy'] for x in v[-10:]])),
                    epoch1000_train_loss=e[-1]['train_loss'],epoch1000_online_accuracy=e[-1]['online_train_accuracy'],
                    epoch1000_validation_accuracy=e[-1]['validation']['accuracy'])
                histories.append(history)
                for x in v:
                    curves.append(cell|dict(epoch=x['epoch'],accuracy=x['validation']['accuracy'],macro_f1=x['validation']['macro_f1']))
    def table(name,values):
        with (output/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(values)
    for name,values in [('metrics.csv',rows),('groups.csv',groups),('per_class.csv',classes),
                        ('histories.csv',histories),('validation_curves.csv',curves)]: table(name,values)
    aggregate=[]
    for kind in ('original','dc_ac'):
        for bn in ('original_bn','calibrated_bn'):
            subset=[r for r in rows if r['input']==kind and r['bn']==bn and r['split']=='validation']
            a=np.array([r['accuracy'] for r in subset]);f=np.array([r['macro_f1'] for r in subset])
            aggregate.append(dict(input=kind,bn=bn,accuracy_mean=float(a.mean()),accuracy_sample_sd=float(a.std(ddof=1)),
                accuracy_worst=float(a.min()),macro_f1_mean=float(f.mean()),
                paired_delta_mean=float(np.mean([r['delta_accuracy_vs_original'] for r in subset]))))
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4.5))
    for kind,color in [('original','#555555'),('dc_ac','#0072B2')]:
        for bn,style in [('original_bn','-'),('calibrated_bn','--')]:
            values=[r for r in rows if r['input']==kind and r['bn']==bn and r['split']=='validation']
            axes[0].plot([r['seed'] for r in values],[100*r['accuracy'] for r in values],style+'o',label=f'{kind} / {bn}',color=color)
        for seed in (0,1,2):
            values=[r for r in curves if r['input']==kind and r['seed']==seed]
            axes[1].plot([r['epoch'] for r in values],[100*r['accuracy'] for r in values],alpha=.7,label=f'{kind} s{seed}')
    axes[0].set(xlabel='Seed',ylabel='Validation accuracy (%)',xticks=[0,1,2]);axes[1].set(xlabel='Epoch',ylabel='Validation accuracy (%)')
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=8)
    figure.tight_layout();figure.savefig(output/'comparison.png',dpi=160);figure.savefig(output/'comparison.svg');plt.close(figure)
    svg=output/'comparison.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    outputs={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}
    script=Path(__file__).resolve()
    summary=dict(schema_version=1,status='complete',protocol=registry['protocol'],heldout_test_evaluated=False,
        primary='paired clean validation Accuracy',secondary='Macro-F1',seed_count=3,uncertainty='sample SD, not confidence interval',
        cells=cells,aggregate=aggregate,histories=histories,inputs_sha256=inputs,outputs_sha256=outputs,
        script=dict(path=script.relative_to(root).as_posix(),sha256=file_sha256(script),text=script.read_text()))
    write_json(output/'summary.json',summary)
    # Keep exact CSVs locally for loaders/proofs and publish compact reversible copies.
    for cell in cells:
        if cell['input']!='dc_ac':continue
        run=root/cell['run_dir']; compressed={}
        for split in ('train','validation'):
            source=run/f'{split}_predictions.csv';target=source.with_suffix('.csv.gz')
            with source.open('rb') as inp,target.open('xb') as out:
                with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as encoded:
                    for chunk in iter(lambda:inp.read(1024*1024),b''):encoded.write(chunk)
            compressed[source.name]=dict(original_sha256=file_sha256(source),compressed_sha256=file_sha256(target),compressed_path=target.name)
        write_json(run/'compression.json',dict(schema_version=1,restore_refuses_overwrite=True,files=compressed))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registry',required=True,type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args();root=args.project_root.resolve()
    from src.ect.paper_dc_ac_training import source_hashes
    from src.ect.paper_dc_ac_registration import inside,read,validate_registered_run
    path=inside(root,args.registry);registry=read(path);folder=path.parent
    for row in registry['runs']:
        validate_registered_run(root,path,read(root/row['config_path']),root/registry['cache_dir'],root/row['output_dir'])
    from src.ect.devices import resolve_device
    device=str(resolve_device(args.device))
    steps=execution_steps(registry,path.relative_to(root).as_posix(),device)
    ensure_fresh_execution(root,folder,steps)
    return run_controller(root,path,registry,device,steps)


def run_controller(root,path,registry,device,steps):
    from src.ect.paper_dc_ac_training import source_hashes
    folder=path.parent
    lock=folder/'controller.running'
    with lock.open('x') as handle:handle.write(str(os.getpid())+'\n')
    started=time.perf_counter()
    status=dict(schema_version=1,status='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        registry_path=path.relative_to(root).as_posix(),device=device,steps=[],source_sha256=None)
    try:
        status['source_sha256']=source_hashes()
        (folder/'logs').mkdir(exist_ok=False);(folder/'verification').mkdir(exist_ok=False)
        write_json(folder/'execution_status.json',status)
        for step in steps:
            entry={k:v for k,v in step.items() if k!='command'}|dict(status='running')
            status['steps'].append(entry);write_json(folder/'execution_status.json',status)
            log=folder/'logs'/f"{step['kind']}_s{step['seed']}.log"
            command=[sys.executable,*step['command'],'--project-root',str(root)]
            entry['command']=command;entry['log_path']=log.relative_to(root).as_posix()
            with log.open('x') as handle:
                result=subprocess.run(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT)
            entry.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode)
            write_json(folder/'execution_status.json',status)
            if result.returncode:raise RuntimeError(f"{step['kind']} seed{step['seed']} failed; see {log}")
        summary=summarize(root,folder,registry)
        status.update(status='complete',elapsed_seconds=time.perf_counter()-started,summary_path=(folder/'analysis_results_v1/summary.json').relative_to(root).as_posix())
        write_json(folder/'execution_status.json',status)
        print(json.dumps(summary['aggregate'],indent=2),flush=True)
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        write_json(folder/'execution_status.json',status)
        write_json(folder/'execution_failure.json',status)
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
