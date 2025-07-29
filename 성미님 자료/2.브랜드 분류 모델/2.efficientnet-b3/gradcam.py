import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # 타겟 레이어 설정 (예: 'blocks.6.2.conv_pwl')
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_full_backward_hook(self._save_gradient)
                break
        else:
            raise ValueError(f"Layer {target_layer_name} not found in model.")

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8  # normalize 0~1
        return cam
