"""Offline numerical and integration checks; no pretrained downloads."""
from contextlib import contextmanager, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from lace import LACE, LACEConfig
from lace.hierarchy import TemporalHierarchy
from lace.losses import accuracy_loss, structural_coherence_loss



class FrozenTestBackbone(nn.Module):
    """Deterministic frozen encoder for offline adaptation tests."""
    latent_dim = 4

    def __init__(self, context_len, pred_len, model_name):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32) + 1, requires_grad=False)

    def forward(self, x):
        return x.mean(dim=1) * self.weight[None, :]


@contextmanager
def stub_backbones(prefix='lace'):
    modules = {}
    for name, cls in [('backbone', 'FrozenMoiraiBackbone'),
                      ('backbone_moirai2', 'FrozenMoirai2Backbone'),
                      ('backbone_chronos2', 'FrozenChronos2Backbone'),
                      ('backbone_timesfm2', 'FrozenTimesFM2Backbone')]:
        m = ModuleType(f'{prefix}.{name}')
        setattr(m, cls, FrozenTestBackbone)
        modules[m.__name__] = m
    sentinel = object()
    previous = {name:sys.modules.get(name,sentinel) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name,old in previous.items():
            if old is sentinel:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = old


def test_model(backbone='Salesforce/moirai-1.0-R-small'):
    with stub_backbones():
        return LACE(context_len=24, pred_len=24, factors=[1, 3, 12, 24],
                    adapter_dim=5, backbone_name=backbone)


class ReleaseTests(unittest.TestCase):
    def test_all_four_backbone_routes(self):
        for name in ['Salesforce/moirai-1.0-R-small', 'Salesforce/moirai-2.0-R-small',
                     'amazon/chronos-2', 'google/timesfm-2.0-500m-pytorch']:
            with self.subTest(backbone=name):
                model = test_model(name).eval()
                out = model(torch.ones(2, 24, 1))
                self.assertEqual({f:tuple(x.shape) for f,x in out.items()},
                                 {1:(2,24), 3:(2,8), 12:(2,2), 24:(2,1)})

    def test_checkpoint_round_trip_and_parameter_keys(self):
        model = test_model().eval()
        x = torch.randn(2, 24, 1)
        before = model(x)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, weights_only=True)
        self.assertTrue(any(k.startswith('adapters.') for k in state))
        self.assertTrue(any(k.startswith('heads.') for k in state))
        restored = test_model().eval()
        restored.load_state_dict(state, strict=True)
        for f,y in restored(x).items():
            torch.testing.assert_close(y, before[f], rtol=0, atol=0)

    def test_retained_supervised_baselines_forward(self):
        args=SimpleNamespace(task_name='long_term_forecast',seq_len=24,pred_len=12,
                             enc_in=1,dec_in=1,c_out=1,individual=False,moving_avg=25,
                             lstm_hidden=8,lstm_layers=1,embed='timeF',freq='h',dropout=0.0)
        x=torch.randn(2,24,1)
        marks=torch.zeros(2,24,4)
        for name in ['DLinear','LSTM']:
            with self.subTest(model=name):
                model=importlib.import_module('models.'+name).Model(args).eval()
                out=model(x,marks,None,None)
                self.assertEqual(tuple(out.shape),(2,12,1))
                self.assertTrue(torch.isfinite(out).all())

    def test_structural_loss_reaches_adapters_and_heads(self):
        torch.manual_seed(7)
        model = test_model()
        loss, _ = structural_coherence_loss(model.effective_mappings(), model.effective_biases(), model.hierarchy)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for branch in [model.adapters, model.heads]:
            for p in branch.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.weight.grad)

    def test_effective_mapping_equals_branch_forecast(self):
        model = test_model().eval()
        z = torch.randn(3, model.latent_dim)
        for f in model.factors:
            expected = z @ model.effective_mappings()[f].T + model.effective_biases()[f]
            torch.testing.assert_close(model.forecast_scale(z, f), expected)

    def test_sum_and_mean_aggregation_match_explicit_operators(self):
        for reduction in ['sum', 'mean']:
            h = TemporalHierarchy([1, 3, 12, 24], 24, reduction)
            x = torch.arange(48, dtype=torch.float32).reshape(2,24)
            for f in [3, 12, 24]:
                y = h.aggregate_prediction(x, fine_factor=1, coarse_factor=f)
                expected = x.reshape(2,24//f,f)
                expected = expected.sum(-1) if reduction=='sum' else expected.mean(-1)
                torch.testing.assert_close(y, expected)

    def test_invalid_hierarchies_are_rejected(self):
        for factors in [[1,1,24], [1,5,24], [0,3,24]]:
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                TemporalHierarchy(factors,24)

    def test_factor_aware_inverse_scaling_preserves_aggregation(self):
        from exp.exp_multiscale_lace_forecast import Exp_Multiscale_LACE_Forecast
        exp = object.__new__(Exp_Multiscale_LACE_Forecast)
        exp.hierarchy = TemporalHierarchy([1,3,12,24],24,'sum')
        x = np.arange(48,dtype=float).reshape(2,24)
        fine = exp._inverse_scale(x, 1, mean=7, std=2)
        for f in [3,12,24]:
            coarse = exp._inverse_scale(x.reshape(2,24//f,f).sum(-1), f, mean=7, std=2)
            np.testing.assert_allclose(coarse, fine.reshape(2,24//f,f).sum(-1))

    def test_wls_is_coherent_and_preserves_coherent_input(self):
        from src.evaluator.reconcile_controls_lace import summing_matrix, reconcile_wls_struct
        S = summing_matrix(24,[1,3,12,24])
        x = np.random.default_rng(2).normal(size=(4,24))
        coherent = x @ S.T
        np.testing.assert_allclose(reconcile_wls_struct(coherent,S),coherent,rtol=1e-6,atol=1e-7)
        rec = reconcile_wls_struct(np.random.default_rng(3).normal(size=coherent.shape),S)
        np.testing.assert_allclose(rec,rec[:,:24]@S.T,rtol=1e-6,atol=1e-7)

    def test_model_registry_is_independent_of_working_directory(self):
        from exp.exp_basic import Exp_Basic
        import os
        old = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as t:
                os.chdir(t)
                registry = Exp_Basic._scan_models_directory(None)
                self.assertEqual(registry['LACE_moirai1'],'models.LACE_moirai1')
        finally:
            os.chdir(old)

    def test_dataset_registry_and_paper_summary_edges(self):
        from src.evaluator.evaluate_lace import get_dataset_config
        for name,fs in [('c_traffic',[1,6,36,144]),('k_traffic',[1,3,12,288]),
                        ('ETTm1_HUFL',[1,4,24,96]),('road_sensor_62',[1,3,12,288])]:
            cfg=get_dataset_config(name)
            self.assertEqual(list(cfg['factor_to_res']),fs)
            self.assertEqual(cfg['model_dir'],'LACE_moirai1')

    def test_training_checkpoint_and_physical_unit_outputs(self):
        import os
        import pandas as pd
        from exp import exp_multiscale_lace_forecast as experiment
        old_cwd = Path.cwd()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as t:
                work = Path(t)
                dates = pd.date_range('2020-01-01', periods=600, freq='h')
                physical = 20 + 3*np.sin(np.arange(600)/13)
                pd.DataFrame({'date':dates,'value':physical}).to_csv(work/'series.csv',index=False)
                os.chdir(work)
                for model in ['LACE_moirai1','LACE_moirai2','LACE_chronos2','LACE_timesfm2']:
                    for mode in ['none','structural']:
                        with self.subTest(model=model,mode=mode):
                            args = SimpleNamespace(
                                model=model,task_name='multiscale_lace_forecast',
                                use_gpu=False,gpu_type='cuda',gpu=0,use_multi_gpu=False,
                                seq_len=24,label_len=12,pred_len=24,
                                lace_factors=[1,3,12,24],lace_adapter_dim=5,
                                lace_representation_norm='batch',lace_reduction='sum',
                                lace_coherence_mode=mode,lace_lambda=0 if mode=='none' else 1,
                                data='fixed_split',root_path=t,data_path='series.csv',
                                data_name='offline_smoke',data_resolution='1h',
                                features='S',target='value',freq='h',embed='timeF',
                                seasonal_patterns='Monthly',test_split_timestamp=str(dates[450]),
                                batch_size=32,num_workers=0,augmentation_ratio=0,
                                train_epochs=1,patience=2,learning_rate=1e-5,
                                checkpoints=str(work/'custom-checkpoints'),inverse=True,
                                use_wandb=False,
                            )
                            with (stub_backbones(),
                                  patch.object(experiment,'EVALUATION_RESULTS_PATH',str(work/'evaluation')),
                                  patch.object(experiment,'visual'),redirect_stdout(io.StringIO())):
                                exp = experiment.Exp_Multiscale_LACE_Forecast(args)
                                setting=f'{model}_{mode}'
                                exp.train(setting)
                                self.assertTrue((work/'custom-checkpoints'/f'{setting}.pth').is_file())
                                exp.test(setting,test=1)
                            folder=work/'evaluation'/model/'offline_smoke'/'1h'/mode
                            finest=np.load(folder/'true_factor_1.npy')
                            np.testing.assert_allclose(finest[0,:],physical[450:474],rtol=1e-6,atol=1e-5)
                            for factor in [1,3,12,24]:
                                p=np.load(folder/f'pred_factor_{factor}.npy')
                                truth=np.load(folder/f'true_factor_{factor}.npy')
                                self.assertEqual(p.shape,truth.shape)
                                self.assertTrue(np.isfinite(p).all())
                                expected=finest.reshape(len(finest),24//factor,factor).sum(axis=2)
                                np.testing.assert_allclose(truth,expected,rtol=1e-6,atol=1e-5)
        finally:
            os.chdir(old_cwd)
            torch.set_num_threads(old_threads)

    def test_preparation_uses_original_aggregation_and_protects_outputs(self):
        import pandas as pd
        from tools.prepare_datasets import prepare_network, prepare_electricity, write_outputs
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            d=Path(t)
            dates=pd.date_range('2020-01-01',periods=288,freq='5min')
            pd.DataFrame({'date':dates,'total_traffic_kbps':np.arange(288,dtype=float)}).to_csv(d/'network.csv',index=False)
            prepared=prepare_network(d/'network.csv','k_traffic')
            self.assertEqual(prepared['k1d_ideal.csv'].iloc[0,0],np.arange(288).sum())
            write_outputs(prepared,d/'out')
            with self.assertRaises(FileExistsError):
                write_outputs(prepared,d/'out')
            dates=pd.date_range('2020-01-01',periods=97,freq='15min')
            pd.DataFrame({'date':dates,'HUFL':np.full(97,4.)}).to_csv(d/'ETTm1.csv',index=False)
            prepared=prepare_electricity(d/'ETTm1.csv')
            self.assertEqual(len(prepared['ETTm1_HUFL_1d.csv']),1)
            self.assertEqual(prepared['ETTm1_HUFL_1d.csv'].iloc[0,0],96.)

    def test_dry_run_matches_recorded_protocol(self):
        recorded=json.loads((ROOT/'configs/paper_protocol.json').read_text())['experiments']
        actual=[]
        for model in ['LACE_moirai1','LACE_moirai2','LACE_chronos2','LACE_timesfm2']:
            env=dict(os.environ,MODEL_NAME=model,ONLY_DATASET='',ONLY_MODE='')
            result=subprocess.run(['bash',str(ROOT/'scripts/run_lace_overnight.sh'),'--dry-run'],
                                  cwd=tempfile.gettempdir(),env=env,text=True,capture_output=True,check=True)
            for line in result.stdout.splitlines():
                if not line.startswith('COMMAND: '):
                    continue
                argv=shlex.split(line[len('COMMAND: '):])
                config={}
                i=3
                while i<len(argv):
                    key=argv[i]
                    i+=1
                    values=[]
                    while i<len(argv) and not argv[i].startswith('--'):
                        values.append(argv[i])
                        i+=1
                    config[key.removeprefix('--')]=values if len(values)>1 else values[0] if values else True
                actual.append(config)
        self.assertEqual(len(actual),48)
        self.assertEqual(actual,recorded)


if __name__ == '__main__':
    unittest.main()
