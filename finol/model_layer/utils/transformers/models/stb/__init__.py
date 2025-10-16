from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_stb": ["STB_PRETRAINED_CONFIG_ARCHIVE_MAP", "STBConfig", "STBOnnxConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_stb"] = ["STBTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_stb_fast"] = ["STBTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_stb"] = [
        "STB_PRETRAINED_MODEL_ARCHIVE_LIST",
        "STBForCausalLM",
        "STBForMaskedLM",
        "STBForMultipleChoice",
        "STBForPreTraining",
        "STBForQuestionAnswering",
        "STBForSequenceClassification",
        "STBForTokenClassification",
        "STBLayer",
        "STBModel",
        "STBPreTrainedModel",
        "load_tf_weights_in_stb",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_stb"] = [
        "FlaxSTBForCausalLM",
        "FlaxSTBForMaskedLM",
        "FlaxSTBForMultipleChoice",
        "FlaxSTBForPreTraining",
        "FlaxSTBForQuestionAnswering",
        "FlaxSTBForSequenceClassification",
        "FlaxSTBForTokenClassification",
        "FlaxSTBModel",
        "FlaxSTBPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_stb import STB_PRETRAINED_CONFIG_ARCHIVE_MAP, STBConfig, STBOnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_stb import STBTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_stb_fast import STBTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_stb import (
            STB_PRETRAINED_MODEL_ARCHIVE_LIST,
            STBForCausalLM,
            STBForMaskedLM,
            STBForMultipleChoice,
            STBForPreTraining,
            STBForQuestionAnswering,
            STBForSequenceClassification,
            STBForTokenClassification,
            STBLayer,
            STBModel,
            STBPreTrainedModel,
            load_tf_weights_in_stb,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_stb import (
            FlaxSTBForCausalLM,
            FlaxSTBForMaskedLM,
            FlaxSTBForMultipleChoice,
            FlaxSTBForPreTraining,
            FlaxSTBForQuestionAnswering,
            FlaxSTBForSequenceClassification,
            FlaxSTBForTokenClassification,
            FlaxSTBModel,
            FlaxSTBPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
