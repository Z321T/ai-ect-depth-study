"""Finite-budget, validation-only reconstruction of the paper training method."""
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import sys
import types

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .baseline import stratified_rows
from .devices import resolve_device, seed_everything
from .integrity import file_sha256
from .paper_crops import training_crops, ten_crop_probabilities
from .paper_models import PaperResNeXt1D
from .prepare import PreparedDataset
from .preprocessing import standardize

CODE_FILES = ('paper_training.py', 'paper_registration.py', 'paper_models.py', 'paper_crops.py', 'devices.py',
              'baseline.py', 'prepare.py', 'dataset.py', 'preprocessing.py', 'integrity.py')
ARTIFACTS = {'model.pt', 'train_rows.json', 'validation_rows.json',
             'train_predictions.csv', 'validation_predictions.csv'}
_SOURCE_FOLDER = Path(__file__).resolve().parent
_SOURCE_PATHS = {f'src/ect/{n}': _SOURCE_FOLDER/n for n in CODE_FILES}
_SOURCE_PATHS['scripts/train_paper_baseline.py'] = _SOURCE_FOLDER.parents[1]/'scripts/train_paper_baseline.py'
# Capture identity during this module's import, before callers can edit source
# and then attribute still-loaded functions to the edited disk version.
_IMPORTED_SOURCE_BYTES = {name:path.read_bytes() for name,path in _SOURCE_PATHS.items()}
FULL_PURPOSE = 'exploratory_train_validation_method_reconstruction'
SMOKE_PURPOSE = 'paper_runner_smoke_validation_only'
# This is an invariant contract, not a second adjustable experiment template.
FIXED = dict(schema_version=1, protocol='paper_baseline_finite_v1',
    model='PaperResNeXt1D', blocks_per_stage=3, training_seeds=[0, 1, 2],
    patience=None, optimizer='Adam', learning_rate=4e-5, betas=[.9,.999],
    optimizer_eps=1e-8, weight_decay=0., learning_rate_milestones_epochs=[5000,7500],
    learning_rate_values=[4e-6,4e-7], crop_length=224, train_crop_namespace='training',
    validation_crop_namespace='validation', validation_crops=10, validation_crop_seed=10,
    selection=['validation_accuracy','earlier_epoch'], noise_augmentation=False,
    test_classification_enabled=False, deterministic=True, tf32=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', newline='\n')


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_hashes(canonical=False):
    verify_imported_sources()
    return {name:hashlib.sha256(value.replace(b'\r\n',b'\n') if canonical else value).hexdigest()
            for name,value in _IMPORTED_SOURCE_BYTES.items()}


def verify_imported_sources():
    for name,path in _SOURCE_PATHS.items():
        if path.read_bytes()!=_IMPORTED_SOURCE_BYTES[name]:
            raise ValueError(f'Source changed after module import: {name}; start a fresh process')
        if name.startswith('src/ect/'):
            module = sys.modules.get(f'{__package__}.{path.stem}')
            if module is not None:
                _verify_loaded_module(module,path,_IMPORTED_SOURCE_BYTES[name])


def _verify_loaded_module(module,path,source):
    # A dependency can have been imported earlier than this runner. Comparing
    # its loaded function/method code against freshly compiled source closes
    # that gap without executing source or changing any old frozen module.
    compiled = compile(source,str(path),'exec',dont_inherit=True)
    expected = {}
    def collect(code):
        expected[code.co_qualname] = code
        for item in code.co_consts:
            if isinstance(item,types.CodeType):
                collect(item)
    collect(compiled)
    defaults = {}
    def definitions(node,prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
                name=f'{prefix}.{child.name}' if prefix else child.name
                try:
                    positional=tuple(ast.literal_eval(v) for v in child.args.defaults) or None
                    keyword={arg.arg:ast.literal_eval(v) for arg,v in zip(child.args.kwonlyargs,child.args.kw_defaults) if v is not None} or None
                except (ValueError,TypeError) as error:
                    raise ValueError(f'Literal source defaults required for identity checking: {name}') from error
                defaults[name]=(positional,keyword)
                definitions(child,name+'.<locals>')
            elif isinstance(child,ast.ClassDef):
                definitions(child,f'{prefix}.{child.name}' if prefix else child.name)
            else:
                definitions(child,prefix)
    definitions(ast.parse(source))
    def check(function):
        if isinstance(function,types.FunctionType) and function.__module__==module.__name__:
            if function.__code__!=expected.get(function.__code__.co_qualname):
                raise ValueError(f'Imported executable code differs from source: {path.name}')
            actual=(function.__defaults__,function.__kwdefaults__)
            if repr(actual)!=repr(defaults.get(function.__code__.co_qualname)):
                raise ValueError(f'Imported function defaults differ from source: {path.name}')
    for obj in vars(module).values():
        check(obj)
        if isinstance(obj,type) and obj.__module__==module.__name__:
            for method in vars(obj).values():
                if isinstance(method,(staticmethod,classmethod)):
                    method=method.__func__
                if isinstance(method,property):
                    for fn in (method.fget,method.fset,method.fdel):
                        check(fn)
                else:
                    check(method)


def validate_config(c):
    extra = {'purpose', 'device', 'threads', 'batch_size', 'max_epochs', 'validation_every_epochs',
             'train_per_class', 'validation_per_class', 'cache_dir', 'notes', 'seed'}
    if not isinstance(c, dict) or set(c) != set(FIXED) | extra:
        raise ValueError('Complete paper_baseline_finite_v1 configuration required')
    for k, v in FIXED.items():
        if c[k] != v or type(c[k]) is not type(v):
            raise ValueError(f'Frozen paper method setting differs: {k}')
    if c['purpose'] not in (FULL_PURPOSE, SMOKE_PURPOSE) or c['device'] not in ('auto','cpu','cuda'):
        raise ValueError('Known train/validation purpose and auto/cpu/cuda device required')
    for key in ('threads', 'batch_size', 'max_epochs', 'validation_every_epochs'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'Positive integer {key} required')
    if type(c['seed']) is not int or c['seed'] not in (0,1,2):
        raise ValueError('Registered seed 0/1/2 required')
    if c['max_epochs'] % c['validation_every_epochs']:
        raise ValueError('Epoch cap must include the final scheduled validation')
    if not isinstance(c['cache_dir'], str) or not c['cache_dir'] or not isinstance(c['notes'], str):
        raise ValueError('Cache path and notes required')
    for k in ('train_per_class','validation_per_class'):
        if c[k] is not None and (type(c[k]) is not int or c[k] < 1):
            raise ValueError('Positive per-class count or null required')
    if c['purpose'] == FULL_PURPOSE:
        expected = dict(threads=1, batch_size=128, max_epochs=1000, validation_every_epochs=10,
                        train_per_class=None, validation_per_class=None, device='auto')
        if any(c[k] != v or type(c[k]) is not type(v) for k,v in expected.items()):
            raise ValueError('Frozen full-data budget/selection settings required')
    json.dumps(c, allow_nan=False)


def learning_rate_for_epoch(config, epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Epoch is one-based positive integer')
    lr = config['learning_rate']
    for milestone, value in zip(config['learning_rate_milestones_epochs'], config['learning_rate_values']):
        if epoch >= milestone:
            lr = value
    return lr


def select_epoch(epochs):
    eligible = [e for e in epochs if e.get('validation') is not None]
    if not eligible:
        raise ValueError('Scheduled validation records required')
    return max(eligible, key=lambda e: (e['validation']['accuracy'], -e['epoch']))


def metrics(labels, predictions, classes):
    detail = classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0)
    return dict(accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, labels=classes, average='macro', zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=classes).tolist(),
        per_class={str(c): detail[str(c)] for c in classes})


def predict_paper(bundle, raw, identities, batch_size=128):
    """Return fixed seed-10 validation-crop means, also for train diagnostics."""
    return ten_crop_probabilities(bundle['model'], raw, identities, bundle['normalization'],
                                  seed=bundle['report']['config']['validation_crop_seed'], batch_size=batch_size)


def bn_summary(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            v = module.running_var.detach().cpu().numpy()
            result[name] = dict(variance_min=float(v.min()), variance_median=float(np.median(v)),
                variance_max=float(v.max()), mean_abs_max=float(module.running_mean.detach().abs().max().cpu()),
                batches=int(module.num_batches_tracked.detach().cpu()))
    return result


def _inside(root, path):
    p = Path(path)
    p = p.resolve() if p.is_absolute() else (root/p).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Run/cache path must stay within project root')
    return p


def _failure(output, report, error, started):
    failed = report | dict(status='failed', error_type=type(error).__name__, error=str(error),
                           elapsed_seconds=time.perf_counter()-started)
    # Mark failure first: an interrupted cleanup cannot leave a loadable success.
    write_json(output/'failure.json', failed)
    (output/'report.json').unlink(missing_ok=True)


def train_paper_baseline(project_root, config, output_dir, cache_dir=None, progress=None,
                         registry_path=None, device_override=None):
    """Train all scheduled epochs; select Accuracy/earlier epoch, never use test."""
    validate_config(config)
    config = json.loads(json.dumps(config, allow_nan=False))
    if device_override is not None and device_override not in ('auto','cpu','cuda'):
        raise ValueError('Device override must be auto/cpu/cuda')
    root = Path(project_root).resolve()
    output = _inside(root, output_dir)
    cache_dir = config['cache_dir'] if cache_dir is None else cache_dir
    cache = _inside(root, cache_dir)
    registration = None
    if config['purpose'] == FULL_PURPOSE:
        if registry_path is None:
            raise ValueError('Frozen execution registration required before full training')
        from .paper_registration import validate_registered_run
        registration = validate_registered_run(root, registry_path, config, cache, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(schema_version=1, status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=config['protocol'], purpose=config['purpose'], config=config,
        config_sha256=config_digest(config), heldout_test_evaluated=False, epochs=[],
        execution_registration=registration, class_mapping_status='unverified',
        selection_rule='validation accuracy then earlier one-based epoch',
        crop_epoch_convention='training RNG epoch = one-based recorded epoch minus one',
        diagnostic_crop_protocol='train and validation use the same seed-10 validation namespace, no optimizer update')
    try:
        (output/'running.marker').write_text('Incomplete run: refuse checkpoint loading.\n',newline='\n')
        report['code_sha256'] = source_hashes()
        report['code_sha256_lf'] = source_hashes(canonical=True)
        seed_everything(config['seed'], threads=config['threads'])
        device = resolve_device(device_override or config['device'])
        report['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
            sklearn=sklearn.__version__, torch=str(torch.__version__), torch_cuda_build=torch.version.cuda,
            device=str(device), device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
            threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32))
        with PreparedDataset(root, cache) as store, threadpool_limits(limits=config['threads']):
            report.update(cache_path=str(cache.relative_to(root)), cache_metadata_sha256=file_sha256(cache/'metadata.json'),
                normalization_sha256=file_sha256(cache/'normalization.json'), normalization=store.stats,
                manifest_sha256=store.metadata['manifest_sha256'],
                raw_input_sha256={k:v['sha256'] for k,v in store.metadata['inputs'].items()})
            manifests = (root/store.metadata['summary_path']).parent
            records, selected, xs, ys, ids = {}, {}, {}, {}, {}
            for split in ('train','validation'):
                rows = [json.loads(line) for line in (manifests/f'{split}.jsonl').read_text().splitlines()]
                raw, labels = store.arrays[split]
                if not np.array_equal(labels, [r['class_index'] for r in rows]):
                    raise ValueError('Cached labels differ from manifest order')
                selected[split] = stratified_rows(labels, config[f'{split}_per_class'], 20261002)
                records[split] = [rows[i] for i in selected[split]]
                xs[split], ys[split] = raw[selected[split]], labels[selected[split]]
                ids[split] = [r['wave_sha256'] for r in records[split]]
            classes = np.unique(ys['train'])
            if len(classes)<2 or not np.array_equal(classes, np.arange(len(classes))) or not np.array_equal(classes,np.unique(ys['validation'])):
                raise ValueError('Matching contiguous classes required')
            report.update(classes=classes.tolist(), sample_counts={s:len(y) for s,y in ys.items()},
                          available_sample_counts={s:len(store.arrays[s][1]) for s in ys})
            model = PaperResNeXt1D(num_classes=len(classes), blocks_per_stage=config['blocks_per_stage']).to(device)
            initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
            report['parameter_count'] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['betas']),
                eps=config['optimizer_eps'], weight_decay=config['weight_decay'])
            report['optimizer'] = dict(name='Adam', betas=config['betas'], eps=config['optimizer_eps'], weight_decay=config['weight_decay'])
            bundle = dict(model=model, normalization=store.stats, report=report)
            best_state, best_probabilities, best_key = None, None, None
            for epoch in range(1, config['max_epochs']+1):
                epoch_started = time.perf_counter()
                lr = learning_rate_for_epoch(config, epoch)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                model.train()
                order = np.random.default_rng(np.random.SeedSequence([config['seed'],epoch-1,2026])).permutation(len(ys['train']))
                loss_sum, correct = 0., 0
                for start in range(0,len(order),config['batch_size']):
                    indices = order[start:start+config['batch_size']]
                    cropped = training_crops(xs['train'][indices], [ids['train'][i] for i in indices], config['seed'], epoch-1)
                    normalized = standardize(cropped,store.stats)
                    inputs = torch.from_numpy(np.ascontiguousarray(normalized.transpose(0,2,1))).to(device)
                    labels = torch.from_numpy(np.asarray(ys['train'][indices],dtype=np.int64)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits,labels)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu())*len(indices)
                    correct += int((logits.detach().argmax(1)==labels).sum().cpu())
                entry = dict(epoch=epoch, learning_rate=lr, train_loss=loss_sum/len(order),
                    online_train_accuracy=correct/len(order), bn=bn_summary(model), validation=None)
                if epoch % config['validation_every_epochs']==0:
                    probabilities = predict_paper(bundle,xs['validation'],ids['validation'],config['batch_size'])
                    val = metrics(ys['validation'],probabilities.argmax(1),classes)
                    entry['validation'] = {k:val[k] for k in ('accuracy','macro_f1')}
                    key = (val['accuracy'],-epoch)
                    if best_key is None or key>best_key:
                        best_key, best_probabilities = key, probabilities.copy()
                        best_state = {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
                entry['elapsed_seconds'] = time.perf_counter()-epoch_started
                report['epochs'].append(entry)
                write_json(output/'progress.json', dict(status='running',seed=config['seed'],last_epoch=entry,
                    selected_epoch=None if best_key is None else -best_key[1], elapsed_seconds=time.perf_counter()-started))
                if progress is not None:
                    progress(entry)
            chosen = select_epoch(report['epochs'])
            model.load_state_dict(best_state,strict=True)
            model.eval()
            report['selected_epoch'] = chosen['epoch']
            report['parameter_update_l2'] = float(sum(((p.detach().cpu().double()-initial[n].double())**2).sum().item() for n,p in model.named_parameters())**.5)
            final_probs = {s:predict_paper(bundle,xs[s],ids[s],config['batch_size']) for s in ('train','validation')}
            if not np.array_equal(final_probs['validation'].argmax(1),best_probabilities.argmax(1)):
                raise ValueError('Selected model prediction differs from selected validation')
            report['metrics'] = {s:metrics(ys[s],p.argmax(1),classes) for s,p in final_probs.items()}
            report['group_metrics'] = {}
            for s in final_probs:
                groups = np.asarray([r['group_id'] for r in records[s]])
                report['group_metrics'][s] = {str(g):metrics(ys[s][groups==g],final_probs[s][groups==g].argmax(1),classes) for g in np.unique(groups)}
            if report['metrics']['validation']['accuracy'] != chosen['validation']['accuracy']:
                raise ValueError('Selected metric differs from history')
            with tempfile.TemporaryDirectory(prefix='.paper-',dir=output) as tmp:
                stage = Path(tmp)
                checkpoint = dict(schema_version=1,protocol=config['protocol'],config=config,
                    normalization=store.stats,selected_epoch=chosen['epoch'],classes=classes.tolist(),state_dict=best_state)
                torch.save(checkpoint,stage/'model.pt')
                restored = PaperResNeXt1D(num_classes=len(classes),blocks_per_stage=config['blocks_per_stage'])
                saved = torch.load(stage/'model.pt',map_location='cpu',weights_only=True)
                restored.load_state_dict(saved['state_dict'],strict=True)
                if any(not torch.equal(v,saved['state_dict'][k]) for k,v in restored.state_dict().items()):
                    raise ValueError('CPU state checkpoint roundtrip failed')
                restored.to(device).eval()
                restored_bundle = dict(model=restored,normalization=store.stats,report=report)
                for split in final_probs:
                    reloaded = predict_paper(restored_bundle,xs[split],ids[split],config['batch_size'])
                    if not np.array_equal(reloaded.argmax(1),final_probs[split].argmax(1)):
                        raise ValueError('Same-device checkpoint prediction differs')
                    write_json(stage/f'{split}_rows.json',selected[split].tolist())
                    with (stage/f'{split}_predictions.csv').open('w',newline='') as handle:
                        writer = csv.writer(handle,lineterminator='\n')
                        writer.writerow(['row_index','wave_sha256','group_id','class_index','predicted_class_index'])
                        for index,row,pred in zip(selected[split],records[split],final_probs[split].argmax(1)):
                            writer.writerow([int(index),row['wave_sha256'],row['group_id'],row['class_index'],int(pred)])
                report['model_roundtrip_verified'] = True
                report['files_sha256'] = {n:file_sha256(stage/n) for n in sorted(ARTIFACTS)}
                report.update(status='complete',elapsed_seconds=time.perf_counter()-started)
                # Verify executable source did not drift during a long run.
                if source_hashes()!=report['code_sha256']:
                    raise ValueError('Training source changed during execution')
                if registration is not None:
                    current_registration = validate_registered_run(root,registry_path,config,cache,output)
                    if current_registration != registration:
                        raise ValueError('Execution registration changed during training')
                for name in sorted(ARTIFACTS):
                    (stage/name).rename(output/name)
            write_json(output/'progress.json',dict(status='complete',seed=config['seed'],epochs=config['max_epochs'],selected_epoch=report['selected_epoch']))
        # Dataset, thread-limit and temporary-directory cleanup all precede
        # the success commit. An unwritable failure marker cannot hide running.
        write_json(output/'.report.json.pending',report)
        (output/'.report.json.pending').replace(output/'report.json')
        (output/'running.marker').unlink()
        return report
    except BaseException as error:
        _failure(output,report,error,started)
        raise


def load_paper_model(run_dir, device='cpu'):
    """Verify complete artifacts, selection history, executable sources and metadata."""
    folder = Path(run_dir)
    if (folder/'failure.json').exists():
        raise ValueError('Failed run cannot be loaded')
    if (folder/'running.marker').exists():
        raise ValueError('Incomplete running run cannot be loaded')
    report = json.loads((folder/'report.json').read_text())
    if report.get('status')!='complete' or type(report.get('schema_version')) is not int or report['schema_version']!=1:
        raise ValueError('Complete schema-1 paper run required')
    validate_config(report['config'])
    if report.get('config_sha256')!=config_digest(report['config']):
        raise ValueError('Configuration checksum mismatch')
    epochs = report.get('epochs',[])
    if len(epochs)!=report['config']['max_epochs'] or [e.get('epoch') for e in epochs]!=list(range(1,len(epochs)+1)):
        raise ValueError('All scheduled epochs required')
    for e in epochs:
        due = e['epoch'] % report['config']['validation_every_epochs']==0
        if (e.get('validation') is not None)!=due or e.get('learning_rate')!=learning_rate_for_epoch(report['config'],e['epoch']):
            raise ValueError('Validation/schedule history contradiction')
        if not np.isfinite(e['train_loss']) or not 0<=e['online_train_accuracy']<=1:
            raise ValueError('Invalid training metrics')
        if due and any(not np.isfinite(v) or not 0<=v<=1 for v in e['validation'].values()):
            raise ValueError('Invalid validation metrics')
    chosen = select_epoch(epochs)
    if (report.get('selected_epoch')!=chosen['epoch'] or
        any(report['metrics']['validation'].get(k)!=v for k,v in chosen['validation'].items())):
        raise ValueError('Contradictory selected epoch/metrics')
    if set(report.get('files_sha256',{}))!=ARTIFACTS:
        raise ValueError('Complete artifact checksum table required')
    for name,digest in report['files_sha256'].items():
        if file_sha256(folder/name)!=digest:
            raise ValueError(f'Run checksum mismatch: {name}')
    if report.get('code_sha256_lf')!=source_hashes(canonical=True):
        raise ValueError('Executed source differs from checkpoint')
    saved = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    if (saved.get('schema_version')!=1 or saved.get('protocol')!=report['protocol'] or
        saved.get('config')!=report['config'] or saved.get('normalization')!=report['normalization'] or
        saved.get('selected_epoch')!=report['selected_epoch'] or saved.get('classes')!=report['classes'] or
        report['classes']!=list(range(len(report['classes']))) or len(report['classes'])<2):
        raise ValueError('Checkpoint metadata differs from report')
    if any(t.device.type!='cpu' or not torch.isfinite(t).all() for t in saved['state_dict'].values()):
        raise ValueError('Finite portable CPU checkpoint required')
    seed_everything(report['config']['seed'],threads=report['config']['threads'])
    model = PaperResNeXt1D(num_classes=len(report['classes']),blocks_per_stage=report['config']['blocks_per_stage'])
    model.load_state_dict(saved['state_dict'],strict=True)
    model.to(resolve_device(device)).eval()
    return dict(model=model,normalization=saved['normalization'],report=report)
