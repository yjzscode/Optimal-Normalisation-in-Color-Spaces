import os

import pytest
import test_models as TM
import torch
from torchvision import models
from torchvision.models._api import get_model_weights, Weights, WeightsEnum
from torchvision.models._utils import handle_legacy_interface


run_if_test_with_extended = pytest.mark.skipif(
    os.getenv("PYTORCH_TEST_WITH_EXTENDED", "0") != "1",
    reason="Extended tests are disabled by default. Set PYTORCH_TEST_WITH_EXTENDED=1 to run them.",
)


@pytest.mark.parametrize(
    "name, model_class",
    [
        ("resnet50", models.ResNet),
        ("retinanet_resnet50_fpn_v2", models.detection.RetinaNet),
        ("raft_large", models.optical_flow.RAFT),
        ("quantized_resnet50", models.quantization.QuantizableResNet),
        ("lraspp_mobilenet_v3_large", models.segmentation.LRASPP),
        ("mvit_v1_b", models.video.MViT),
    ],
)
def test_get_model(name, model_class):
    assert isinstance(models.get_model(name), model_class)


@pytest.mark.parametrize(
    "name, weight",
    [
        ("resnet50", models.ResNet50_Weights),
        ("retinanet_resnet50_fpn_v2", models.detection.RetinaNet_ResNet50_FPN_V2_Weights),
        ("raft_large", models.optical_flow.Raft_Large_Weights),
        ("quantized_resnet50", models.quantization.ResNet50_QuantizedWeights),
        ("lraspp_mobilenet_v3_large", models.segmentation.LRASPP_MobileNet_V3_Large_Weights),
        ("mvit_v1_b", models.video.MViT_V1_B_Weights),
    ],
)
def test_get_model_weights(name, weight):
    assert models.get_model_weights(name) == weight


@pytest.mark.parametrize(
    "module", [models, models.detection, models.quantization, models.segmentation, models.video, models.optical_flow]
)
def test_list_models(module):
    def get_models_from_module(module):
        return [
            v.__name__
            for k, v in module.__dict__.items()
            if callable(v) and k[0].islower() and k[0] != "_" and k not in models._api.__all__
        ]

    a = set(get_models_from_module(module))
    b = set(x.replace("quantized_", "") for x in models.list_models(module))

    assert len(b) > 0
    assert a == b


@pytest.mark.parametrize(
    "name, weight",
    [
        ("ResNet50_Weights.IMAGENET1K_V1", models.ResNet50_Weights.IMAGENET1K_V1),
        ("ResNet50_Weights.DEFAULT", models.ResNet50_Weights.IMAGENET1K_V2),
        (
            "ResNet50_QuantizedWeights.DEFAULT",
            models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2,
        ),
        (
            "ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1",
            models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1,
        ),
    ],
)
def test_get_weight(name, weight):
    assert models.get_weight(name) == weight


@pytest.mark.parametrize(
    "model_fn",
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
def test_naming_conventions(model_fn):
    weights_enum = get_model_weights(model_fn)
    assert weights_enum is not None
    assert len(weights_enum) == 0 or hasattr(weights_enum, "DEFAULT")


@pytest.mark.parametrize(
    "model_fn",
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
@run_if_test_with_extended
def test_schema_meta_validation(model_fn):
    # list of all possible supported high-level fields for weights meta-data
    permitted_fields = {
        "backend",
        "categories",
        "keypoint_names",
        "license",
        "_metrics",
        "min_size",
        "min_temporal_size",
        "num_params",
        "recipe",
        "unquantized",
        "_docs",
    }
    # mandatory fields for each computer vision task
    classification_fields = {"categories", ("_metrics", "ImageNet-1K", "acc@1"), ("_metrics", "ImageNet-1K", "acc@5")}
    defaults = {
        "all": {"_metrics", "min_size", "num_params", "recipe", "_docs"},
        "models": classification_fields,
        "detection": {"categories", ("_metrics", "COCO-val2017", "box_map")},
        "quantization": classification_fields | {"backend", "unquantized"},
        "segmentation": {
            "categories",
            ("_metrics", "COCO-val2017-VOC-labels", "miou"),
            ("_metrics", "COCO-val2017-VOC-labels", "pixel_acc"),
        },
        "video": {"categories", ("_metrics", "Kinetics-400", "acc@1"), ("_metrics", "Kinetics-400", "acc@5")},
        "optical_flow": set(),
    }
    model_name = model_fn.__name__
    module_name = model_fn.__module__.split(".")[-2]
    expected_fields = defaults["all"] | defaults[module_name]

    weights_enum = get_model_weights(model_fn)
    if len(weights_enum) == 0:
        pytest.skip(f"Model '{model_name}' doesn't have any pre-trained weights.")

    problematic_weights = {}
    incorrect_params = []
    bad_names = []
    for w in weights_enum:
        actual_fields = set(w.meta.keys())
        actual_fields |= set(
            ("_metrics", dataset, metric_key)
            for dataset in w.meta.get("_metrics", {}).keys()
            for metric_key in w.meta.get("_metrics", {}).get(dataset, {}).keys()
        )
        missing_fields = expected_fields - actual_fields
        unsupported_fields = set(w.meta.keys()) - permitted_fields
        if missing_fields or unsupported_fields:
            problematic_weights[w] = {"missing": missing_fields, "unsupported": unsupported_fields}
        if w == weights_enum.DEFAULT:
            if module_name == "quantization":
                # parameters() count doesn't work well with quantization, so we check against the non-quantized
                unquantized_w = w.meta.get("unquantized")
                if unquantized_w is not None and w.meta.get("num_params") != unquantized_w.meta.get("num_params"):
                    incorrect_params.append(w)
            else:
                if w.meta.get("num_params") != sum(p.numel() for p in model_fn(weights=w).parameters()):
                    incorrect_params.append(w)
        else:
            if w.meta.get("num_params") != weights_enum.DEFAULT.meta.get("num_params"):
                if w.meta.get("num_params") != sum(p.numel() for p in model_fn(weights=w).parameters()):
                    incorrect_params.append(w)
        if not w.name.isupper():
            bad_names.append(w)

    assert not problematic_weights
    assert not incorrect_params
    assert not bad_names


@pytest.mark.parametrize(
    "model_fn",
    TM.list_model_fns(models)
    + TM.list_model_fns(models.detection)
    + TM.list_model_fns(models.quantization)
    + TM.list_model_fns(models.segmentation)
    + TM.list_model_fns(models.video)
    + TM.list_model_fns(models.optical_flow),
)
@run_if_test_with_extended
def test_transforms_jit(model_fn):
    model_name = model_fn.__name__
    weights_enum = get_model_weights(model_fn)
    if len(weights_enum) == 0:
        pytest.skip(f"Model '{model_name}' doesn't have any pre-trained weights.")

    defaults = {
        "models": {
            "input_shape": (1, 3, 224, 224),
        },
        "detection": {
            "input_shape": (3, 300, 300),
        },
        "quantization": {
            "input_shape": (1, 3, 224, 224),
        },
        "segmentation": {
            "input_shape": (1, 3, 520, 520),
        },
        "video": {
            "input_shape": (1, 3, 4, 112, 112),
        },
        "optical_flow": {
            "input_shape": (1, 3, 128, 128),
        },
    }
    module_name = model_fn.__module__.split(".")[-2]

    kwargs = {**defaults[module_name], **TM._model_params.get(model_name, {})}
    input_shape = kwargs.pop("input_shape")
    x = torch.rand(input_shape)
    if module_name == "optical_flow":
        args = (x, x)
    else:
        if module_name == "video":
            x = x.permute(0, 2, 1, 3, 4)
        args = (x,)

    problematic_weights = []
    for w in weights_enum:
        transforms = w.transforms()
        try:
            TM._check_jit_scriptable(transforms, args)
        except Exception:
            problematic_weights.append(w)

    assert not problematic_weights


# With this filter, every unexpected warning will be turned into an error
@pytest.mark.filterwarnings("error")
class TestHandleLegacyInterface:
    class ModelWeights(WeightsEnum):
        Sentinel = Weights(url="https://pytorch.org", transforms=lambda x: x, meta=dict())

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(), id="empty"),
            pytest.param(dict(weights=None), id="None"),
            pytest.param(dict(weights=ModelWeights.Sentinel), id="Weights"),
        ],
    )
    def test_no_warn(self, kwargs):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        builder(**kwargs)

    @pytest.mark.parametrize("pretrained", (True, False))
    def test_pretrained_pos(self, pretrained):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        with pytest.warns(UserWarning, match="positional"):
            builder(pretrained)

    @pytest.mark.parametrize("pretrained", (True, False))
    def test_pretrained_kw(self, pretrained):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        with pytest.warns(UserWarning, match="deprecated"):
            builder(pretrained)

    @pytest.mark.parametrize("pretrained", (True, False))
    @pytest.mark.parametrize("positional", (True, False))
    def test_equivalent_behavior_weights(self, pretrained, positional):
        @handle_legacy_interface(weights=("pretrained", self.ModelWeights.Sentinel))
        def builder(*, weights=None):
            pass

        args, kwargs = ((pretrained,), dict()) if positional else ((), dict(pretrained=pretrained))
        with pytest.warns(UserWarning, match=f"weights={self.ModelWeights.Sentinel if pretrained else None}"):
            builder(*args, **kwargs)

    def test_multi_params(self):
        weights_params = ("weights", "weights_other")
        pretrained_params = [param.replace("weights", "pretrained") for param in weights_params]

        @handle_legacy_interface(
            **{
                weights_param: (pretrained_param, self.ModelWeights.Sentinel)
                for weights_param, pretrained_param in zip(weights_params, pretrained_params)
            }
        )
        def builder(*, weights=None, weights_other=None):
            pass

        for pretrained_param in pretrained_params:
            with pytest.warns(UserWarning, match="deprecated"):
                builder(**{pretrained_param: True})

    def test_default_callable(self):
        @handle_legacy_interface(
            weights=(
                "pretrained",
                lambda kwargs: self.ModelWeights.Sentinel if kwargs["flag"] else None,
            )
        )
        def builder(*, weights=None, flag):
            pass

        with pytest.warns(UserWarning, match="deprecated"):
            builder(pretrained=True, flag=True)

        with pytest.raises(ValueError, match="weights"):
            builder(pretrained=True, flag=False)
