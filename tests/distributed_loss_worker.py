"""Compare DDP global-pair gradients against the single-process objective."""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel

from dinov3.finetune.temporal.loss import masks, sigmoid_loss


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3, bias=False)
        self.log_temperature = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, images):
        return self.linear(images), self.log_temperature.exp(), self.bias


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        torch.manual_seed(123)
        model = TinyEncoder().double()
        reference = TinyEncoder().double()
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(8, 5, dtype=torch.float64)
        positive, negative = masks(torch.arange(8) // 2, torch.zeros(8), torch.arange(8), torch.arange(8))
        features, temperature, bias = reference(inputs)
        expected_loss = sigmoid_loss(features, temperature, bias, positive, negative)
        expected_loss.backward()
        runner = DistributedDataParallel(model)
        features, temperature, bias = runner(inputs[rank * 4 : (rank + 1) * 4])
        loss = sigmoid_loss(torch.cat(all_gather(features)), temperature, bias, positive, negative)
        loss.backward()
        torch.testing.assert_close(loss, expected_loss)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
