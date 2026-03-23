from dataclasses import dataclass

@dataclass
class ConvNeXtConfig:
    batch_size: int = 32   
    epochs: int = 10       
    input_size: int = 224
    
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2    
    
    label_smoothing: float = 0.0
    stochastic_depth_prob: float = 0.1 
    layer_scale_init_value: float = 1e-6
