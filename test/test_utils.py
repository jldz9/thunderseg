import sys
from pathlib import Path
sys.path.append(Path('~/thunderseg/src').expanduser().as_posix())

import unittest
from thunderseg.utils.tool import get_config, Config

# -------------
# Test tool.py
# -------------
# 1. test tool.get_config function

class test_tool_get_config(unittest.TestCase):
    def test_no_input(self):
        self.assertTrue(get_config())
        return get_config()
    def test_input_not_file(self):
        self.assertRaises(FileNotFoundError, get_config, **{"config_path":'not_exist.yaml'})
    def test_input_wrong_file(self):
        self.assertRaises(ValueError,get_config, **{"config_path":Path(__file__).parents[1].joinpath('environment.yml')})

# 2. test tool.Config class
class test_tool_Config(unittest.TestCase):
    def test_basic_initialization(self):
        cfg = Config(a=1, b='test')
        self.assertTrue(cfg.a, 1)
        self.assertTrue(cfg.b, 'test')
    def test_nested_initialization(self):
        cfg = Config(model={"name": "resnet", "layers": 50})
        self.assertIsInstance(cfg.model, Config)
        self.assertEqual(cfg.model.name, "resnet")
        self.assertEqual(cfg.model.layers, 50)
    def test_list_of_dicts(self):
        cfg = Config(layers=[{"type": "conv"}, {"type": "relu"}])
        self.assertIsInstance(cfg.layers[0], Config)
        self.assertEqual(cfg.layers[1].type, "relu")
    def test_update_config(self):
        cfg = Config(a=1, b={"c": 2})
        cfg.update({"a": 10, "b": {"c": 20, "d": 30}})
        self.assertEqual(cfg.a, 10)
        self.assertEqual(cfg.b.c, 20)
        self.assertEqual(cfg.b.d, 30)
    def test_pop_nested_attribute(self):
        cfg = Config(a=1, b={"c": 2, "d": 3})
        cfg.pop("b.c")
        self.assertFalse(hasattr(cfg.b, "c"))
        self.assertEqual(cfg.b.d, 3)
    def test_to_dict(self):
        cfg = Config(x=1, y={"z": [1, 2, {"a": "b"}]})
        result = cfg.to_dict()
        self.assertEqual(result["y"]["z"][2]["a"], "b")
        self.assertIsInstance(result, dict)
    def test_recursion_error(self):
        recursive_dict = {}
        recursive_dict["item"] = recursive_dict
        with self.assertRaises(RecursionError):
            Config(**recursive_dict)

# 3. 
if __name__ == '__main__':
    unittest.main()
