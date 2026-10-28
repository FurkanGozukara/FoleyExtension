import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('_genjutsu_apply_test')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
apply = importlib.import_module(package.__name__ + '.genjutsu_apply')


class GenjutsuApplyTests(unittest.TestCase):
    def test_original_method_leaves_model_and_conditioning_unchanged(self):
        model, positive = object(), object()
        with mock.patch.object(apply, 'LoraLoaderModelOnly') as loader:
            result = apply.SECoursesH3GenjutsuApply.execute(model, positive, {}, object())
        self.assertIs(result.args[0], model)
        self.assertIs(result.args[1], positive)
        loader.assert_not_called()

    def test_character_and_motion_load_the_correct_weights_in_order(self):
        for background, expected in ((apply.BACKGROUNDS[0], [apply.CHARACTER_LORA]),
                                     (apply.BACKGROUNDS[1], [apply.TRAINING_ADAPTER, apply.MOTION_LORA])):
            with self.subTest(background=background), mock.patch.object(apply, 'LoraLoaderModelOnly') as loader:
                loader.return_value.load_lora_model_only.side_effect = lambda model, filename, strength: (filename,)
                result = apply.SECoursesH3GenjutsuApply.execute('base', 'positive', {}, 'vae',
                            method=apply.METHODS[1], background=background)
                self.assertEqual([call.args[1] for call in loader.return_value.load_lora_model_only.call_args_list], expected)
                self.assertEqual(result.args[0], expected[-1])
                self.assertEqual(result.args[1], 'positive')

    def test_aligned_method_uses_native_guide_and_dense_backend(self):
        guide, latent = object(), object()
        with (mock.patch.object(apply, 'LoraLoaderModelOnly') as loader,
              mock.patch.object(apply.MiniMaxH3AddGuide, 'execute', return_value=apply.io.NodeOutput('aligned')) as add,
              mock.patch.object(apply.ModelAttentionBackend, 'execute', return_value=apply.io.NodeOutput('dense')) as dense):
            loader.return_value.load_lora_model_only.return_value = ('patched',)
            result = apply.SECoursesH3GenjutsuApply.execute('base', 'positive', latent, 'vae',
                        method=apply.METHODS[2], guide=guide)
        self.assertEqual(result.args, ('dense', 'aligned'))
        self.assertIs(add.call_args.kwargs['image'], guide)
        self.assertIs(add.call_args.kwargs['latent'], latent)
        dense.assert_called_once_with('patched', 'comfy kitchen attention')

    def test_schema_exposes_all_bundle_dependencies(self):
        files = [apply.CHARACTER_LORA, apply.MOTION_LORA, apply.TRAINING_ADAPTER]
        with mock.patch.object(apply.folder_paths, 'get_filename_list', return_value=files):
            schema = apply.SECoursesH3GenjutsuApply.define_schema()
        self.assertEqual(schema.node_id, 'SECoursesH3GenjutsuApply')


if __name__ == '__main__':
    unittest.main()
