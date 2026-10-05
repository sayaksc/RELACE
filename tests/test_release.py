import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('environment,model,mini_batch,max_steps', [
    ('alfworld', 'Qwen/Qwen2.5-1.5B-Instruct', '64', '50'),
    ('webshop', 'Qwen/Qwen2.5-1.5B-Instruct', '64', '15'),
    ('alfworld', 'Qwen/Qwen2.5-7B-Instruct', '64', '50'),
    ('webshop', 'Qwen/Qwen2.5-7B-Instruct', '128', '15'),
])
def test_launch_profile(environment, model, mini_batch, max_steps):
    env = dict(os.environ, DRY_RUN='1', MODEL_PATH=model, LOGGER='console', RELACE_GAMMA='0.6')
    for name in ['PPO_MINI_BATCH_SIZE', 'PROJECT_DIR']:
        env.pop(name, None)
    command = subprocess.check_output(['bash', str(ROOT / f'examples/relace_trainer/run_{environment}.sh'),
                                       'vllm', 'trainer.total_training_steps=7'], env=env, text=True)
    args = shlex.split(command)
    assert 'algorithm.adv_estimator=relace' in args
    assert 'algorithm.relace.gamma=0.6' in args
    assert f'algorithm.relace.hindsight_template={environment}' in args
    assert f'actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch}' in args
    assert f'env.max_steps={max_steps}' in args
    assert args[-1] == 'trainer.total_training_steps=7'
    assert Path(args[1]).is_file()


def test_data_preparation_is_local_and_has_expected_schema():
    path = ROOT / 'examples/data_preprocess/prepare.py'
    spec = importlib.util.spec_from_file_location('prepare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep even temporary test output inside this repository.
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = module.prepare(directory, 3, 2)
        for split, count in [('train', 3), ('test', 2)]:
            rows = pd.read_parquet(output / f'{split}.parquet')
            assert len(rows) == count
            assert set(rows.columns) == {'data_source', 'prompt', 'ability', 'extra_info'}
            assert rows.iloc[0]['prompt'][0]['role'] == 'user'
            assert rows.iloc[0]['prompt'][0]['content'] == ''
