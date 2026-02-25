import torch.nn as nn

class FeatureRecorder:
    def __init__(self, layers: dict[str, nn.Module]):
        """
        layers: dict mapping names to modules you want to hook
        e.g. {"past_e4": model.past_enc.layer4, "ctx_e4": model.ctx_enc.layer4}
        """
        self.layers = layers
        self.activations = {}
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def __enter__(self):
        for name, layer in self.layers.items():
            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()