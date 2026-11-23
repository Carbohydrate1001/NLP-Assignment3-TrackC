"""Checks for scoring, voting, and retrieval errors that would invalidate results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from harness import parse_answer, gold_answer, vote, needs_more, Retriever, validate_config
import run_trackc as runner


class ProtocolTests(unittest.TestCase):
    def test_numeric_scoring(self):
        self.assertEqual(parse_answer("work 8; #### 1,200.00"), "1200")
        self.assertEqual(parse_answer("#### -0.50"), "-0.5")
        self.assertEqual(parse_answer("3 then 42"), "42")
        self.assertIsNone(parse_answer("I cannot solve this."))
        self.assertNotEqual(parse_answer("#### 3.141"), gold_answer("#### 3.14"))
        self.assertEqual(parse_answer("#### 8\nCorrection: #### 9"), "9")

    def test_vote_and_adaptive_stop(self):
        self.assertEqual(vote(["#### 3", "#### 4", "#### 4"]), "4")
        self.assertEqual(vote(["#### 3", "#### 4", "#### 5"]), "3")
        self.assertIsNone(vote(["no", "unknown"]))
        self.assertFalse(needs_more(["#### 3", "#### 3", "unknown"]))
        self.assertTrue(needs_more(["#### 3", "#### 4", "unknown"]))

    def test_retrieval_uses_question_only(self):
        pool = [{"question": "apples in a basket", "answer": "secret A"},
                {"question": "speed distance train", "answer": "secret B"}]
        self.assertEqual(Retriever(pool).select("train speed", 1), [1])
        pool[0]["answer"] = "train speed distance"
        self.assertEqual(Retriever(pool).select("train speed", 1), [1])

    def test_invalid_api_config(self):
        for ids in ([20], [1, 1], [True], ["0"], [0, 1, 2, 3]):
            with self.assertRaises(ValueError):
                validate_config({"instruction": "solve", "demo_ids": ids}, 20)

    def test_hidden_stage_requires_frozen_complete_experiment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                runner.freeze(root)
            with patch.object(runner, "dataset_split") as hidden:
                with self.assertRaises(FileNotFoundError):
                    runner.final(root)
                hidden.assert_not_called()

    def test_sealing_blocks_tuning_and_existing_test_is_read_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save(root / "frozen.json", {})
            with self.assertRaises(RuntimeError):
                runner.prepare(root)
            runner.save(root / "final_test.json", {"metrics": {"accuracy": .42}})
            with patch.object(runner, "dataset_split") as hidden, patch.object(runner, "Evaluator") as model:
                runner.final(root)
                hidden.assert_not_called()
                model.assert_not_called()

    def test_final_selection_includes_ablations_and_cost_tie_break(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            runner.save(root/'manifest.json',{'seed':42})
            runner.save(root/'visible.json',{'dev':[],'demos':[]})
            for label in ('R0','R1','R2','R3','R4','ablation_fixed3','ablation_fixed5','human_demos_only'):
                score=.6 if label in ('R4','human_demos_only') else .4
                tokens=500 if label=='human_demos_only' else 1000
                runner.save(root/f'{label}.json',{'config':{'instruction':label,'demo_ids':[]},
                    'metrics':{'accuracy':score,'generated_tokens':tokens}})
            runner.freeze(root)
            self.assertEqual(runner.read(root/'frozen.json')['selected_round'],'human_demos_only')

    def test_legacy_cache_requires_identical_inputs_and_evaluator(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            source={name:(runner.HERE/name).read_text(encoding='utf-8')
                    for name in ('harness.py','run_trackc.py')}
            runner.save(root/'provenance/baseline_source.json',source)
            class CachedOnly:
                manifest={'max_new_tokens':256}
                def evaluate(self,*args,**kwargs):
                    raise AssertionError('Cache must not invoke inference')
            rows=[{'question':'sample question','answer':'#### 1'}]
            cfg={'instruction':'solve','demo_ids':[]}
            signature=runner.digest({'rows':rows,'config':cfg,'seed':42,'limit':256,
                                      'manifest':CachedOnly.manifest,'code_hash':runner.digest(source)})
            runner.save(root/'R0.json',{'signature':signature,'metrics':{'accuracy':1.0}})
            runner.cache_eval(root,'R0',CachedOnly(),rows,cfg)
            with self.assertRaises(RuntimeError):
                runner.cache_eval(root,'R0',CachedOnly(),rows,{**cfg,'instruction':'different'})


if __name__ == "__main__":
    unittest.main()
