import pytest

from ifv_training.io import write_json, sha256_file
from ifv_training.psd_repair import _sha
from ifv_training.psd_repair_storage import save_bound
from scripts.server.recover_psd_cached_review import load_review_inputs


def prepared(tmp_path):
    case = tmp_path / 'case'
    write_json(tmp_path / 'trace.json', {'observed': 'unchanged'})
    config = {'trace': {'path': str(tmp_path / 'trace.json'), 'sha256': sha256_file(tmp_path / 'trace.json')}}
    save_bound(case / 'run-inputs.json', identity=config, payload={'status': 'initialized'})
    save_bound(case / 'slate-state.json', identity={'inputs': _sha(config)}, payload={
        'status': 'repairing', 'rounds': [],
        'pending_proposal': {'round_index': 0, 'hints': {'2': {'text': 'Check the inference.'}}}})
    episode = {'termination': 'success'}
    directory = case / 'slate-rounds/00'
    save_bound(directory / 'continuation.json', identity={'inputs': _sha(config), 'hints': {'2': 'Check the inference.'}},
               payload={'teacher_complete': True, 'teacher_episode_trace': episode})
    write_json(directory / 'episode.json', episode)
    return case, directory


def test_review_only_recovery_reuses_exact_completed_episode(tmp_path):
    case, directory = prepared(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.rglob('*.json')}
    config, found, continuation = load_review_inputs(case)
    assert found == directory and continuation['teacher_complete']
    assert all(sha256_file(p) == digest for p, digest in before.items())


@pytest.mark.parametrize('mutation', ['source', 'episode', 'continuation'])
def test_review_only_recovery_refuses_changed_source_or_capture(tmp_path, mutation):
    case, directory = prepared(tmp_path)
    target = {'source': tmp_path / 'trace.json', 'episode': directory / 'episode.json',
              'continuation': directory / 'continuation.json'}[mutation]
    write_json(target, {'changed': True})
    with pytest.raises(ValueError):
        load_review_inputs(case)


def test_resume_preserves_protocol_budgets_and_original_paths(tmp_path):
    from scripts.server.resume_psd_reviewed_case import restore_args
    config = {'output_dir': str(tmp_path / 'out'), 'repair_attempts': 6, 'proposal_rounds': 12,
              'trace': {'path': str(tmp_path / 'trace'), 'sha256': 'bound'},
              'continuation_policy_version': 'version-bound-by-driver'}
    args = restore_args(config)
    assert args.resume and not args.skip_auto_judge
    assert args.repair_attempts == 6 and args.proposal_rounds == 12
    assert args.trace == tmp_path / 'trace' and args.output_dir == tmp_path / 'out'
    assert 'continuation_policy_version' not in vars(args)


def pending_with_completed_round(tmp_path):
    from ifv_training.io import load_json
    case,directory=prepared(tmp_path)
    config=load_json(case/'run-inputs.json')['identity']
    config.update(repair_attempts=6,proposal_rounds=12,output_dir=str(case))
    save_bound(case/'run-inputs.json',identity=config,payload={'status':'initialized'})
    files={str(p):sha256_file(p) for p in directory.iterdir() if p.is_file()}
    state={'status':'repairing','rounds':[{'files':files}],'proposals':[{'status':'accepted'},{'status':'accepted'}],
        'pending_proposal':{'round_index':1,'hints':{'2':{'text':'Check the inference.'}}}}
    save_bound(case/'slate-state.json',identity={'inputs':_sha(config)},payload=state)
    return case,directory,config,state


def test_pending_engineering_recovery_preserves_completed_round(tmp_path):
    from scripts.server.resume_psd_pending_cases import load_pending,restore_args
    case,directory,config,state=pending_with_completed_round(tmp_path)
    found,saved,files=load_pending(case)
    assert found==config and saved==state
    assert str(directory/'episode.json') in files
    args=restore_args(found)
    assert args.repair_attempts==6 and args.proposal_rounds==12 and args.resume


@pytest.mark.parametrize('what',['source','completed','cursor','budget'])
def test_pending_engineering_recovery_rejects_changed_contract(tmp_path,what):
    from scripts.server.resume_psd_pending_cases import load_pending
    case,directory,config,state=pending_with_completed_round(tmp_path)
    if what=='source':write_json(tmp_path/'trace.json',{'changed':True})
    elif what=='completed':write_json(directory/'episode.json',{'changed':True})
    else:
        if what=='cursor':state['pending_proposal']['round_index']=0
        else:
            config['repair_attempts']=7
            save_bound(case/'run-inputs.json',identity=config,payload={'status':'initialized'})
        save_bound(case/'slate-state.json',identity={'inputs':_sha(config)},payload=state)
    with pytest.raises(AssertionError):load_pending(case)
