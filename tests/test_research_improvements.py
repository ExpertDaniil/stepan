from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
import io
import json
import tempfile
import time
import unittest

from agent.budget import Budget
from agent.catalog import Catalog
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.ml import ShiftedLogTargetModel, build_pipeline, enforce_threads, validate_estimator
from agent.observability.logging import RunLogger
from scripts.analyze_run import inspect_run
from tests.helpers import Harness
from tests.test_gateway_logging import FakeTransport
from tests import test_gateway_logging as gateway_cases


class AdaptiveGatewayTests(unittest.TestCase):
    setUp = gateway_cases.GatewayTests.setUp
    tearDown = gateway_cases.GatewayTests.tearDown
    context = gateway_cases.GatewayTests.context
    args = gateway_cases.GatewayTests.args
    final_payload = gateway_cases.GatewayTests.final_payload
    def test_truncation_grows_budget_and_preserves_actual_usage(self):
        transport = FakeTransport([
            {"choices":[{"finish_reason":"length","message":{"role":"assistant","reasoning_content":"available provider field"}}],"usage":{"completion_tokens":1024}},
            {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(self.final_payload())}}],"usage":{"completion_tokens":100}},
        ])
        budget = Budget(self.h.config,time.monotonic(),self.h.log)
        gateway = Gateway(self.h.config,budget,self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        result = gateway.complete(self.args())
        self.assertEqual([r['max_tokens'] for r in transport.requests],[1024,2048])
        self.assertEqual(result['usage']['output_tokens'],1124)
        self.assertEqual(budget.token_snapshot()['output_charged'],1124)
        self.assertEqual(budget.token_snapshot()['output_reserved'],0)

    def test_incomplete_tool_arguments_at_cap_retry_without_executing_tool(self):
        call={"id":"broken","type":"function","function":{"name":"write_code","arguments":'{"researcher_id":"researcher-1"'}}
        transport=FakeTransport([
            {"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[call]}}],"usage":{"completion_tokens":1024}},
            {"choices":[{"message":{"content":json.dumps(self.final_payload())}}]},
        ])
        gateway=Gateway(self.h.config,Budget(self.h.config,time.monotonic(),self.h.log),self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        result=gateway.complete(self.args())
        self.assertEqual(result['tool_calls'],[])
        self.assertEqual(transport.requests[1]['max_tokens'],2048)
        self.assertIn('llm.output_truncated',(self.h.run/'events.jsonl').read_text())

    def test_truncation_at_ceiling_is_not_repeated(self):
        config=replace(self.h.config,llm_max_output_tokens_limit=1024)
        transport=FakeTransport([{"choices":[{"finish_reason":"length","message":{}}],"usage":{"completion_tokens":1024}}])
        gateway=Gateway(config,Budget(config,time.monotonic(),self.h.log),self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        with self.assertRaises(AgentError) as caught:
            gateway.complete(self.args())
        self.assertEqual(caught.exception.code,'LLM_OUTPUT_TRUNCATED')
        self.assertEqual(len(transport.requests),1)

    def test_per_request_timeout_is_distinct_from_operation_budget(self):
        timeouts=[]
        class Transport(FakeTransport):
            def complete(inner,request,timeout,context):
                timeouts.append(timeout)
                return super().complete(request,timeout,context)
        transport=Transport([{"choices":[{"message":{"content":json.dumps(self.final_payload())}}]}])
        config=replace(self.h.config,llm_timeout_seconds=1,llm_operation_timeout_seconds=60)
        gateway=Gateway(config,Budget(config,time.monotonic(),self.h.log),self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        gateway.complete(self.args())
        self.assertGreater(timeouts[0],0.9)
        self.assertLessEqual(timeouts[0],1)

    def test_missing_usage_is_charged_conservatively_without_fabricating_usage(self):
        transport=FakeTransport([{"choices":[{"message":{"content":json.dumps(self.final_payload())}}]}])
        budget=Budget(self.h.config,time.monotonic(),self.h.log)
        gateway=Gateway(self.h.config,budget,self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        result=gateway.complete(self.args())
        self.assertIsNone(result['usage']['output_tokens'])
        self.assertEqual(budget.token_snapshot()['output_reported'],0)
        self.assertEqual(budget.token_snapshot()['output_charged'],1024)

    def test_token_reservations_prevent_parallel_overcommit(self):
        budget=Budget(replace(self.h.config,llm_max_total_output_tokens=2048),time.monotonic(),self.h.log)
        first,second=budget.begin_llm_call(1024),budget.begin_llm_call(1024)
        with self.assertRaises(AgentError):
            budget.begin_llm_call(1024)
        self.assertEqual(budget.calls,2)
        budget.finish_llm_call(first,{'completion_tokens':100})
        with self.assertRaises(AgentError):
            budget.begin_llm_call(1024)
        budget.finish_llm_call(second,{'completion_tokens':100})
        self.assertEqual(budget.begin_llm_call(1024),3)

    def test_slow_api_emits_progress_without_extra_requests(self):
        class SlowTransport(FakeTransport):
            def complete(inner,request,timeout,context):
                time.sleep(1.05)
                return super().complete(request,timeout,context)
        transport=SlowTransport([{"choices":[{"message":{"content":json.dumps(self.final_payload())}}]}])
        config=replace(self.h.config,llm_progress_seconds=1)
        gateway=Gateway(config,Budget(config,time.monotonic(),self.h.log),self.h.artifacts,self.h.bus,self.h.log,output_schemas(self.h.bus),transport)
        gateway.complete(self.args())
        self.assertEqual(len(transport.requests),1)
        self.assertIn('llm.waiting',(self.h.run/'events.jsonl').read_text())


class RecipeAndLoggingTests(unittest.TestCase):
    def test_configure_preserves_credentials_unknown_settings_and_supports_dry_run(self):
        from scripts.configure import configure
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'.env'
            original='OPENAI_API_KEY="unit-private-key"\nOPENAI_BASE_URL=https://provided.invalid/v1\nLLM_MODE=live\nWORK_DIR=custom\nLLM_MAX_OUTPUT_TOKENS=8192\n'
            path.write_text(original)
            configure(path)
            self.assertEqual(path.read_text(),original)
            configure(path,apply=True)
            text=path.read_text()
            self.assertIn('OPENAI_API_KEY="unit-private-key"',text)
            self.assertIn('WORK_DIR=custom',text)
            self.assertIn('LLM_MODE=live',text)
            self.assertIn('LLM_MAX_OUTPUT_TOKENS=16384',text)
            self.assertIn('LLM_OPERATION_TIMEOUT_SECONDS=1200',text)
            self.assertEqual(text.count('LLM_MAX_OUTPUT_TOKENS='),1)

    def test_json_console_preserves_all_structured_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            output=io.StringIO()
            log=RunLogger(Path(temp),'test-json',console_format='json')
            try:
                with redirect_stderr(output):
                    log.event('llm.call_completed',call_no=7,finish_reason='stop',output_tokens=42)
            finally:
                log.close()
            value=json.loads(output.getvalue())
            self.assertEqual(value['call_no'],7)
            self.assertEqual(value['output_tokens'],42)

    def test_histogram_recipe_has_valid_parameters_and_runs(self):
        import numpy as np
        import pandas as pd
        from threadpoolctl import threadpool_limits
        h=Harness('regression')
        try:
            catalog=Catalog('regression',h.config)
            params=catalog.normalize('hist-gradient-regressor-v1',{'max_iter':20})
            self.assertNotIn('n_jobs',params)
            self.assertIs(params['early_stopping'],False)
            model=validate_estimator(build_pipeline('hist-gradient-regressor-v1',h.spec,params))
            X=pd.DataFrame({'x':np.arange(40),'category':['a','b']*20})
            with threadpool_limits(limits=1):
                model.fit(X,np.arange(40))
                self.assertTrue(np.isfinite(model.predict(X)).all())
        finally:
            h.close()

    def test_target_transform_uses_fit_labels_and_returns_original_units(self):
        import numpy as np
        from sklearn.dummy import DummyRegressor
        model=ShiftedLogTargetModel(DummyRegressor())
        y=np.array([-3.,0.,10.,100.])
        model.fit([[0],[1],[2],[3]],y)
        expected=np.expm1(np.log1p(y+3).mean())-3
        self.assertAlmostEqual(model.predict([[9]])[0],expected)
        self.assertEqual(model.shift_,3)

    def test_invalid_nested_estimator_is_rejected_before_fit(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        with self.assertRaises(ValueError):
            validate_estimator(Pipeline([('model',HistGradientBoostingRegressor(early_stopping='off'))]))

    def test_extra_trees_nested_thread_limit_is_enforced(self):
        h=Harness('regression')
        try:
            catalog=Catalog('regression',h.config)
            params=catalog.normalize('extra-trees-regressor-v1',{'target_transform':'shifted_log1p'})
            model=enforce_threads(build_pipeline('extra-trees-regressor-v1',h.spec,params),2)
            self.assertEqual(model.get_params()['estimator__model__n_jobs'],2)
            validate_estimator(model)
        finally:
            h.close()

    def test_catalog_assignments_use_incumbent_and_different_families(self):
        h=Harness()
        try:
            catalog=Catalog(h.spec['task_kind'],h.config)
            rows=[{'recipe_id':'linear-classifier-v1','is_incumbent':True}]
            focus=catalog.assignment_recipes('exploit_incumbent',rows)
            explore=catalog.assignment_recipes('explore_alternative',rows)
            self.assertEqual(focus,['linear-classifier-v1'])
            self.assertTrue(all(catalog.recipes[k]['family']!='linear' for k in explore))
        finally:
            h.close()

    def test_console_shows_state_parameters_tokens_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            output=io.StringIO()
            log=RunLogger(Path(temp),'test-run',secrets=['private-secret'])
            try:
                with redirect_stderr(output):
                    log.event('job.finished',level='ERROR',state='failed',hyperparameters={'depth':6},message='private-secret',finish_reason='length',output_tokens=8192)
            finally:
                log.close()
            text=output.getvalue()
            for value in ('state=failed','hyperparameters=', 'output_tokens=8192','finish_reason=length','[REDACTED]'):
                self.assertIn(value,text)
            self.assertNotIn('private-secret',text)

    def test_archive_analysis_detects_censored_output_without_reading_csv(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'run.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('run/events.jsonl',json.dumps({'event':'run.started','run_id':'test'})+'\n')
                z.writestr('run/llm/0001.request.json',json.dumps({'max_tokens':8192}))
                z.writestr('run/llm/0001.response.json',json.dumps({'choices':[{'finish_reason':'length','message':{}}],'usage':{'completion_tokens':8192}}))
                z.writestr('run/train.csv',b'not a readable dataset')
            result=inspect_run(archive)[0]
            self.assertEqual(result['truncated_responses'],1)
            self.assertEqual(result['suggested_initial_output_tokens'],16384)
            self.assertEqual(result['suggested_output_ceiling'],32768)
