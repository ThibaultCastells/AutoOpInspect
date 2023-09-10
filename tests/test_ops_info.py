import unittest
from AutoOpInspect import OpsInfoProvider
import torchvision
import torch 

class TestOpsInfoProvider(unittest.TestCase):
    
    def setUp(self):
        print("="*40)
        print("Starting setUp\n")

        self.models_to_test = []

        try:
            from diffusers import StableDiffusionPipeline
            model_id = "runwayml/stable-diffusion-v1-5"
            stable_diffusion_model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
            self.models_to_test.append({
                "model": stable_diffusion_model,
                "input": ['a cat', None, None, 1, 7.5, None, 1, 0.0, None, None, None, None, 'pil', False, None, 1, None, 0.0],
                "target_input": [torch.randn(2, 4, 64, 64)],
                "target_output": [(torch.randn(2, 4, 64, 64),)],
                "target": stable_diffusion_model.unet
            })
        except ImportError as e:
            print(e)

        vgg_model = torchvision.models.vgg11()
        self.models_to_test.append({
            "model": vgg_model,
            "input": [torch.randn(1, 3, 224, 224)],
            "target_input": [torch.randn(1, 3, 224, 224)],
            "target_output": [torch.randn(1, 1000)],
            "target": vgg_model
        })
        
        print("setUp completed")
    
    def test_equality(self):
        print("="*40)
        print("Starting test_equality")

        for model_info in self.models_to_test:
            ops_info_provider = OpsInfoProvider(model_info["model"], model_info["input"], target=model_info["target"])
            op_equal = OpsInfoProvider(model_info["model"], model_info["input"], target=model_info["target"])
            self.assertEqual(op_equal, ops_info_provider)
        
        print("test_equality completed")
    
    def test_get_dummy(self):
        print("="*40)
        print("Starting test_get_dummy")

        for model_info in self.models_to_test:
            ops_info_provider = OpsInfoProvider(model_info["model"], model_info["input"], target=model_info["target"])
            dummy_input, dummy_output = ops_info_provider.get_dummy(model_info["target"], mode='both')
            
            self.assertEqual(dummy_input[0].shape, model_info["target_input"][0].shape)
            if isinstance(model_info["target_output"][0], tuple):
                self.assertEqual(dummy_output[0][0].shape, model_info["target_output"][0][0].shape)
            else:
                self.assertEqual(dummy_output[0].shape, model_info["target_output"][0].shape)
        
        print("test_get_dummy completed")

if __name__ == "__main__":
    unittest.main()
