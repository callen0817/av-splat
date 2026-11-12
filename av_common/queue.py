"""Sequential resumable file jobs with per-source change detection."""
import hashlib
import json
from pathlib import Path
import os
from av_common.data import atomic_json,fingerprint,check_cancel


def run_queue(manifest_path,state_path,processor,cancel=lambda:False,progress=lambda message:None):
    manifest_path=Path(manifest_path).resolve(); state_path=Path(state_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or not isinstance(manifest.get('jobs'),list): raise ValueError('Expected batch schema_version 1 and jobs array.')
    jobs=[]; identifiers=set(); inputs=set(); outputs=set()
    for item in manifest['jobs']:
        identifier=str(item['id'])
        if not identifier or identifier in identifiers: raise ValueError('Batch job IDs must be unique.')
        identifiers.add(identifier)
        source=(manifest_path.parent/item['input']).resolve(); output=(manifest_path.parent/item['output']).resolve()
        if output in outputs: raise ValueError('Batch jobs must not share an output filename.')
        inputs.add(source); outputs.add(output); jobs.append((identifier,source,output,item.get('config',{})))
    if manifest_path in outputs or inputs.intersection(outputs) or state_path in inputs or state_path in outputs or state_path==manifest_path:
        raise ValueError('Batch outputs/journal must not overwrite input data or the manifest.')
    for source in inputs:
        for output in outputs:
            if source.exists() and output.exists() and os.path.samefile(source,output): raise ValueError('A batch output is a hard link to input data.')
    journal=json.loads(state_path.read_text()) if state_path.exists() else {'schema_version':1,'jobs':{}}
    if journal.get('schema_version')!=1 or not isinstance(journal.get('jobs'),dict): raise ValueError('Invalid batch journal.')
    outcomes=[]
    for identifier,source,output,config in jobs:
        check_cancel(cancel)
        old=journal['jobs'].get(identifier,{})
        try:
            if not source.is_file(): raise FileNotFoundError(f'Batch input file is unavailable: {source}')
            stamp=fingerprint(source)
        except OSError as error:
            record={'status':'failed','source':{'path':str(source)},'error':str(error),'attempts':old.get('attempts',0)+1}
            journal['jobs'][identifier]=record; atomic_json(state_path,journal)
            outcomes.append({'id':identifier,'status':'failed','error':str(error)})
            continue
        identity=hashlib.sha256(json.dumps({'source':stamp,'output':str(output),'config':config},sort_keys=True).encode()).hexdigest()
        old=journal['jobs'].get(identifier,{})
        if old.get('identity')==identity and old.get('status')=='done' and output.exists() and old.get('output')==fingerprint(output):
            progress(f'Batch {identifier}: unchanged completed job skipped'); outcomes.append({'id':identifier,'status':'skipped'}); continue
        record={'identity':identity,'status':'running','source':stamp,'attempts':old.get('attempts',0)+1}
        journal['jobs'][identifier]=record; atomic_json(state_path,journal)
        try:
            output.parent.mkdir(parents=True,exist_ok=True); progress(f'Batch {identifier}: processing {source.name}')
            processor(source,output,config,cancel,progress)
            if fingerprint(source)!=stamp: raise ValueError('Source changed while batch job ran.')
            record.update(status='done',output=fingerprint(output)); outcomes.append({'id':identifier,'status':'done'})
        except InterruptedError:
            record.update(status='cancelled'); atomic_json(state_path,journal); raise
        except Exception as error:
            record.update(status='failed',error=str(error)); outcomes.append({'id':identifier,'status':'failed','error':str(error)})
        atomic_json(state_path,journal)
    return outcomes
