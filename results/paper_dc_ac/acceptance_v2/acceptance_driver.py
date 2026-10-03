"""Reproduce E2 smoke training, independent CPU/CUDA/BN proofs and regression.

Run in a context with CUDA access. Refuses an existing run/proof/log output.
The independent review file is supplied separately after reviewing frozen code.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.ect.paper_dc_ac_training import source_hashes
from src.ect.integrity import file_sha256
import torch

FOLDER=Path(__file__).resolve().parent

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',newline='\n')

def command(arguments,logname):
    path=FOLDER/logname
    with path.open('x') as handle:
        result=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Acceptance failed: {path} (exit {result.returncode})')
    return path


def main():
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA required; CPU skips cannot pass acceptance')
    frozen=source_hashes()
    started=time.perf_counter();runs=[];bn_runs=[]
    for device in ('cpu','cuda'):
        run=FOLDER/device
        command(['scripts/train_paper_dc_ac.py','--seed','0','--smoke','--device',device,
                 '--output',str(run)],f'{device}_train.log')
        proof=FOLDER/f'{device}_verification.json'
        command(['scripts/verify_paper_dc_ac.py','--run',str(run),'--device',device,'--output',str(proof)],f'{device}_verify.log')
        runs.append(dict(device=device,run_dir=run.relative_to(ROOT).as_posix(),verification_path=proof.relative_to(ROOT).as_posix()))
        bn=FOLDER/f'{device}_bn';bn_proof=FOLDER/f'{device}_bn_verification.json'
        command(['scripts/recalibrate_paper_dc_ac_bn.py','--origin',str(run),'--output',str(bn),'--device',device],f'{device}_bn.log')
        command(['scripts/verify_paper_dc_ac.py','--run',str(bn),'--device',device,'--output',str(bn_proof)],f'{device}_bn_verify.log')
        bn_runs.append(dict(device=device,run_dir=bn.relative_to(ROOT).as_posix(),verification_path=bn_proof.relative_to(ROOT).as_posix()))
    cross=FOLDER/'cuda_cpu_verification.json';bn_cross=FOLDER/'cuda_bn_cpu_verification.json'
    command(['scripts/verify_paper_dc_ac.py','--run',runs[1]['run_dir'],'--device','cpu','--allow-device-change','--output',str(cross)],'cuda_cpu_verify.log')
    command(['scripts/verify_paper_dc_ac.py','--run',bn_runs[1]['run_dir'],'--device','cpu','--allow-device-change','--output',str(bn_cross)],'cuda_bn_cpu_verify.log')
    before=time.perf_counter()
    log=command(['-m','unittest','discover','-s','tests','-v'],'regression.log')
    import re
    content=log.read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',content)
    if not match or not content.rstrip().endswith('OK'):raise RuntimeError('Regression must pass without skips')
    regression=dict(schema_version=1,status='passed',exit_code=0,tests=int(match.group(1)),skipped=0,
        cuda_available=True,device_name=torch.cuda.get_device_name(0),elapsed_seconds=time.perf_counter()-before,
        source_sha256=frozen,files_sha256={log.relative_to(ROOT).as_posix():file_sha256(log)},
        command=[sys.executable,'-m','unittest','discover','-s','tests','-v'])
    write(FOLDER/'regression.json',regression)
    if source_hashes()!=frozen:raise RuntimeError('Source changed during acceptance; refuse freezing')
    write(FOLDER/'steps_complete.json',dict(schema_version=1,status='passed',source_sha256=frozen,runs=runs,bn_runs=bn_runs,
        cross_device_verification_path=cross.relative_to(ROOT).as_posix(),bn_cross_device_verification_path=bn_cross.relative_to(ROOT).as_posix(),
        elapsed_seconds=time.perf_counter()-started,driver_sha256=file_sha256(__file__)))
    print(json.dumps({'status':'passed','tests':regression['tests'],'elapsed_seconds':time.perf_counter()-started}),flush=True)


def finalize():
    from src.ect.paper_dc_ac_registration import check_acceptance,CHECKS
    steps=json.loads((FOLDER/'steps_complete.json').read_text())
    if source_hashes()!=steps['source_sha256']:raise RuntimeError('Source differs from accepted steps')
    review=FOLDER/'independent_review.json'
    if not review.exists():raise RuntimeError('Independent reviewed source evidence required')
    files={}
    def add(path,digest=None):
        path=Path(path).resolve()
        key=path.relative_to(ROOT).as_posix();actual=file_sha256(path)
        if digest is not None and actual!=digest:raise RuntimeError(f'Evidence drift: {path}')
        files[key]=actual
    for path in FOLDER.rglob('*'):
        if path.is_file() and path.name!='acceptance.json':add(path)
    for path in FOLDER.glob('*verification.json'):
        proof=json.loads(path.read_text())
        for label,digest in proof['checked_files_sha256'].items():add(label,digest)
    for label,digest in steps['source_sha256'].items():add(ROOT/label,digest)
    acceptance=dict(schema_version=1,status='accepted',protocol='paper_dc_ac_finite_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=steps['source_sha256'],checks={name:True for name in sorted(CHECKS)},test_classification_enabled=False,
        runs=steps['runs'],bn_runs=steps['bn_runs'],cross_device_verification_path=steps['cross_device_verification_path'],
        bn_cross_device_verification_path=steps['bn_cross_device_verification_path'],
        regression_path=(FOLDER/'regression.json').relative_to(ROOT).as_posix(),review_path=review.relative_to(ROOT).as_posix(),
        files_sha256=files,driver=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=file_sha256(__file__),text=Path(__file__).read_text()))
    path=FOLDER/'acceptance.json'
    if path.exists():raise FileExistsError(path)
    write(path,acceptance)
    try:check_acceptance(ROOT,path)
    except BaseException:
        path.rename(FOLDER/'rejected_acceptance.json')
        raise
    print(json.dumps({'status':'accepted','files':len(files),'source_files':len(steps['source_sha256'])}),flush=True)


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--finalize':finalize()
    else:main()
