from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLES_PER_SECOND = 16000
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * SAMPLE_WIDTH
# 2 BYTES = 16 BITS = 1 SAMPLE
# 1 SECOND OF AUDIO = 32000 BYTES = 16000 SAMPLES


type Device = Literal["cpu", "cuda", "auto"]

# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md#quantize-on-model-conversion
type Quantization = Literal[
    "int8", "int8_float16", "int8_bfloat16", "int8_float32", "int16", "float16", "bfloat16", "float32", "default"
]


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    inference_device: Device = "cuda"  # Force GPU usage for better performance
    device_index: int | list[int] = 0
    compute_type: Quantization = "float16"  # Optimized for H100 (consider int8_float16 for lower VRAM)
    cpu_threads: int = 1  # Use 1 when fully offloading to GPU to avoid context switching
    num_workers: int = 1  # Keep at 1 for GPU inference - multiple workers reduce throughput
    use_batched_mode: bool = True
    batch_size: int = Field(
        default=128,
        ge=1,
        description="Optimized for H100 80GB. Use 64 for A100, 32 for RTX 4090, 16 for RTX 3090.",
    )
    batch_window_ms: int = Field(
        default=50,  # Reduced for lower latency
        ge=1,
        description="Window (ms) to wait and aggregate requests into a batch. 50-100ms typical trade-off.",
    )

    # If true: preload the model on startup (avoid cold starts). Strongly recommended for low latency.
    preload_model: bool = True  # Keep enabled for production

    # Maximum number of requests that can wait in the server queue
    max_queue_size: int = Field(
        default=512, ge=1, description="Max queued requests waiting for batcher. Increase if expecting spikes."
    )

    # Soft limit for concurrent in-flight requests (application-level)
    max_concurrent_requests: int = Field(
        default=200, ge=1, description="Application-level cap to avoid flooding CPU/GPU. Adjust to real tests."
    )

    # When a model is unused for this many seconds unload it. -1 = never unload.
    # For H100 & low-latency prefer never unload.
    model_ttl: int = -1


class OrtOptions(BaseModel):
    exclude_providers: list[str] = ["TensorrtExecutionProvider"]
    """
    List of ORT providers to exclude from the inference session.
    """
    provider_priority: dict[str, int] = {"CUDAExecutionProvider": 100}
    """
    Dictionary of ORT providers and their priority. The higher the value, the higher the priority. Default priority for a provider if not specified is 0.
    """
    provider_opts: dict[str, dict[str, Any]] = {}
    """
    Dictionary of ORT provider options. The keys are provider names, and the values are dictionaries of options.
    Example: {"CUDAExecutionProvider": {"cudnn_conv_algo_search": "DEFAULT"}}
    """


# TODO: document `alias` behaviour within the docstring
class Config(BaseSettings):
    """Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__INFERENCE_DEVICE`(note the double underscore) to `whisper.inference_device`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    stt_model_ttl: int = Field(default=-1, ge=-1)
    """
    Time in seconds until a speech to text (stt) model is unloaded after last usage.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """

    tts_model_ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until a text to speech (tts) model is unloaded after last usage.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """

    api_key: SecretStr | None = None
    """
    If set, the API key will be required for all requests.
    """
    log_level: str = "debug"
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None
    """
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values
    Usage:
        `export ALLOW_ORIGINS='["http://localhost:3000", "http://localhost:3001"]'`
        `export ALLOW_ORIGINS='["*"]'`
    """

    enable_ui: bool = False
    """
    Whether to enable the Gradio UI. You may want to disable this if you want to minimize the dependencies and slightly improve the startup time.
    """

    whisper: WhisperConfig = WhisperConfig()

    # TODO: remove the underscore prefix from the field name
    _unstable_vad_filter: bool = False
    """
    Default value for VAD (Voice Activity Detection) filter in speech recognition endpoints.
    When enabled, the model will filter out non-speech segments. Useful for removing hallucinations in speech recognition caused by background silences.


    NOTE: having `_unstable_vad_filter: True` technically deviates from the OpenAI API specification, so you may want to set it to `False`.

    NOTE: This is an unstable feature and may change in the future.
    """

    loopback_host_url: str | None = None
    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """

    # TODO: document the below configuration options
    chat_completion_base_url: str = "http://localhost:11434/v1"
    chat_completion_api_key: SecretStr = SecretStr("cant-be-empty")

    unstable_ort_opts: OrtOptions = OrtOptions()
