import pytest
import test_models as TM
import torch
from common_utils import cpu_and_gpu, set_rng_seed
from torchvision.prototype import models


@pytest.mark.parametrize("model_fn", TM.list_model_fns(models.depth.stereo))
@pytest.mark.parametrize("model_mode", ("standard", "scripted"))
@pytest.mark.parametrize("dev", cpu_and_gpu())
def test_raft_stereo(model_fn, model_mode, dev):
    # A simple test to make sure the model can do forward pass and jit scriptable
    set_rng_seed(0)

    # Use corr_pyramid and corr_block with smaller num_levels and radius to prevent nan output
    # get the idea from test_models.test_raft
    corr_pyramid = models.depth.stereo.raft_stereo.CorrPyramid1d(num_levels=2)
    corr_block = models.depth.stereo.raft_stereo.CorrBlock1d(num_levels=2, radius=2)
    model = model_fn(corr_pyramid=corr_pyramid, corr_block=corr_block).eval().to(dev)

    if model_mode == "scripted":
        model = torch.jit.script(model)

    img1 = torch.rand(1, 3, 64, 64).to(dev)
    img2 = torch.rand(1, 3, 64, 64).to(dev)
    num_iters = 3

    preds = model(img1, img2, num_iters=num_iters)
    depth_pred = preds[-1]

    assert len(preds) == num_iters, "Number of predictions should be the same as model.num_iters"

    assert depth_pred.shape == torch.Size(
        [1, 1, 64, 64]
    ), f"The output shape of depth_pred should be [1, 1, 64, 64] but instead it is {preds[0].shape}"

    # Test against expected file output
    TM._assert_expected(depth_pred.cpu(), name=model_fn.__name__, atol=1e-2, rtol=1e-2)
