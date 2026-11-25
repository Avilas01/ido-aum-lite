"""Hosted tiny accuracy fixtures only: no CIFAR or pretrained model download."""
import pytest
pytestmark = pytest.mark.torch
torch = pytest.importorskip('torch')
from src.evaluate import accuracy_on_loader


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.flags = []

    def forward(self, x):
        self.flags.append((self.training, torch.is_grad_enabled()))
        return self.scale*x


def fixture_loader(**kwargs):
    logits = torch.zeros(5,10)
    logits[torch.arange(5),torch.tensor([0,1,1,3,4])] = 5
    data = torch.utils.data.TensorDataset(logits,torch.tensor([0,1,2,3,4]))
    return torch.utils.data.DataLoader(data,batch_size=2,**kwargs)


def test_hand_accuracy_partial_final_batch_and_no_grad_modes():
    model = Echo().train()
    result = accuracy_on_loader(model,fixture_loader(),torch.device('cpu'))
    assert result == {'correct':4,'total':5,'accuracy_pct':80.}
    assert model.training and model.scale.grad is None
    assert model.flags == [(False,False)]*3


def test_batchnorm_buffers_and_mixed_modes_are_preserved():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(10),torch.nn.Dropout(.5))
    model.train()
    model[1].eval()
    before = {key:value.clone() for key,value in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    accuracy_on_loader(model,fixture_loader(),torch.device('cpu'))
    assert [m.training for m in model.modules()] == modes
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


@pytest.mark.parametrize('kwargs',[{'shuffle':True},{'drop_last':True}])
def test_noncomplete_or_random_loader_rejected(kwargs):
    with pytest.raises(ValueError,match='every official test example'):
        accuracy_on_loader(Echo(),fixture_loader(**kwargs),torch.device('cpu'))


def test_nonfinite_logits_restore_modes_and_fail():
    class Broken(Echo):
        def forward(self,x):
            return x*float('nan')
    model = Broken().train()
    with pytest.raises(ValueError,match='finite ten-class'):
        accuracy_on_loader(model,fixture_loader(),torch.device('cpu'))
    assert model.training
