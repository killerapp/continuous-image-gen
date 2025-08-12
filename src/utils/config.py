"""
Configuration management for the image generation system.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import json
from dataclasses import dataclass, asdict, field

@dataclass
class LoraConfig:
    """Lora-specific configuration."""
    lora_dir: Path = Path("C:/ComfyUI/ComfyUI/models/loras")
    enabled_loras: List[str] = field(default_factory=list)
    # Probability of applying any Lora (0.0 to 1.0)
    application_probability: float = 0.7

@dataclass
class ModelConfig:
    """Model-specific configuration."""
    ollama_model: str = "phi4:latest"
    ollama_temperature: float = 0.7
    flux_model_path: str = "dev"
    max_sequence_length: int = 512
    lora: LoraConfig = field(default_factory=LoraConfig)

    @property
    def flux_model(self) -> str:
        return self.flux_model_path

@dataclass
class ImageConfig:
    """Image generation configuration."""
    height: int = 768
    width: int = 1360
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    true_cfg_scale: float = 1.0

@dataclass
class PluginConfig:
    """Plugin-related configuration."""
    enabled: List[str] = field(default_factory=lambda: [
        "time_of_day",
        "nearest_holiday",
        "holiday_fact",
        "art_style",
        "lora",
    ])
    descriptions: Dict[str, str] = field(default_factory=dict)
    plugin_order: Dict[str, int] = field(default_factory=lambda: {
        "time_of_day": 1,
        "nearest_holiday": 2,
        "holiday_fact": 3,
        "art_style": 4,
        "lora": 5,
    })

@dataclass
class SystemConfig:
    """System-related configuration."""
    output_dir: Path = Path("output")
    log_dir: Path = Path("logs")
    cache_dir: Path = Path(".cache")
    image_output_dir: Path = Path("output")
    cpu_only: bool = False
    mps_use_fp16: bool = False


@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    level: str = "INFO"


@dataclass
class TemporalContextConfig:
    """Configuration for temporal context generation."""
    enabled: bool = False
    use_day_of_week: bool = False
    use_time_of_day: bool = False
    use_holidays: bool = False

class Config:
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        image: Optional[ImageConfig] = None,
        system: Optional[SystemConfig] = None,
        logging: Optional[LoggingConfig] = None,
        temporal_context: Optional[TemporalContextConfig] = None,
        plugins: Optional[PluginConfig] = None,
    ):
        self.plugins = plugins or PluginConfig()

        self.model = model or ModelConfig(
            lora=LoraConfig(
                lora_dir=Path(os.getenv("LORA_DIR", LoraConfig.lora_dir)),
                enabled_loras=os.getenv("ENABLED_LORAS", "").split(",") if os.getenv("ENABLED_LORAS") else [],
                application_probability=float(
                    os.getenv(
                        "LORA_APPLICATION_PROBABILITY",
                        LoraConfig.application_probability,
                    )
                ),
            ),
            ollama_model=os.getenv("OLLAMA_MODEL", ModelConfig.ollama_model),
            ollama_temperature=float(
                os.getenv("OLLAMA_TEMPERATURE", ModelConfig.ollama_temperature)
            ),
            flux_model_path=os.getenv("FLUX_MODEL", ModelConfig.flux_model_path),
            max_sequence_length=int(
                os.getenv("MAX_SEQUENCE_LENGTH", ModelConfig.max_sequence_length)
            ),
        )

        self.image = image or ImageConfig(
            height=int(os.getenv("IMAGE_HEIGHT", ImageConfig.height)),
            width=int(os.getenv("IMAGE_WIDTH", ImageConfig.width)),
            num_inference_steps=int(
                os.getenv("NUM_INFERENCE_STEPS", ImageConfig.num_inference_steps)
            ),
            guidance_scale=float(
                os.getenv("GUIDANCE_SCALE", ImageConfig.guidance_scale)
            ),
            true_cfg_scale=float(
                os.getenv("TRUE_CFG_SCALE", ImageConfig.true_cfg_scale)
            ),
        )

        self.system = system or SystemConfig(
            output_dir=Path(os.getenv("OUTPUT_DIR", SystemConfig.output_dir)),
            log_dir=Path(os.getenv("LOG_DIR", SystemConfig.log_dir)),
            cache_dir=Path(os.getenv("CACHE_DIR", SystemConfig.cache_dir)),
            image_output_dir=Path(
                os.getenv("IMAGE_OUTPUT_DIR", SystemConfig.image_output_dir)
            ),
            cpu_only=bool(os.getenv("CPU_ONLY", SystemConfig.cpu_only)),
            mps_use_fp16=bool(os.getenv("MPS_USE_FP16", SystemConfig.mps_use_fp16)),
        )

        self.logging = logging or LoggingConfig(
            level=os.getenv("LOG_LEVEL", LoggingConfig.level)
        )

        self.temporal_context = temporal_context or TemporalContextConfig(
            enabled=bool(
                os.getenv("TEMPORAL_ENABLED", TemporalContextConfig.enabled)
            ),
            use_day_of_week=bool(
                os.getenv(
                    "TEMPORAL_DAY_OF_WEEK", TemporalContextConfig.use_day_of_week
                )
            ),
            use_time_of_day=bool(
                os.getenv(
                    "TEMPORAL_TIME_OF_DAY", TemporalContextConfig.use_time_of_day
                )
            ),
            use_holidays=bool(
                os.getenv("TEMPORAL_HOLIDAYS", TemporalContextConfig.use_holidays)
            ),
        )
        
    @classmethod
    def from_file(cls, config_path: Path) -> 'Config':
        """Load configuration from a JSON file."""
        if not config_path.exists():
            return cls()
            
        with open(config_path) as f:
            data = json.load(f)
            
        config = cls()
        for section, values in data.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        if isinstance(value, dict) and key.endswith('_dir'):
                            value = Path(value)
                        setattr(section_config, key, value)
                        
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'image': asdict(self.image),
            'plugins': asdict(self.plugins),
            'logging': asdict(self.logging),
            'temporal_context': asdict(self.temporal_context),
            'system': {k: str(v) if isinstance(v, Path) else v
                      for k, v in asdict(self.system).items()}
        }
        
    def save(self, config_path: Path):
        """Save configuration to a JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def validate(self) -> list[str]:
        """
        Validate configuration values.
        
        Returns:
            list[str]: List of validation errors, empty if valid
        """
        errors = []
        
        # Validate image dimensions
        if not (128 <= self.image.height <= 2048):
            errors.append(f"Invalid height: {self.image.height} (must be between 128 and 2048)")
        if not (128 <= self.image.width <= 2048):
            errors.append(f"Invalid width: {self.image.width} (must be between 128 and 2048)")
            
        # Validate model parameters
        if not (1 <= self.image.num_inference_steps <= 150):
            errors.append(f"Invalid inference steps: {self.image.num_inference_steps} (must be between 1 and 150)")
        if not (1.0 <= self.image.guidance_scale <= 30.0):
            errors.append(f"Invalid guidance scale: {self.image.guidance_scale} (must be between 1.0 and 30.0)")
        if not (1.0 <= self.image.true_cfg_scale <= 10.0):
            errors.append(f"Invalid true CFG scale: {self.image.true_cfg_scale} (must be between 1.0 and 10.0)")
            
        # Validate system paths
        for path_attr in ['output_dir', 'log_dir', 'cache_dir', 'image_output_dir']:
            path = getattr(self.system, path_attr)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Invalid {path_attr}: {path} ({str(e)})")
                
        return errors
