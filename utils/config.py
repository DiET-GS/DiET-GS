
import sys
sys.path.append("/data/sjlee/gaussian-splatting-event2")
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from threestudio.utils.typing import *

@dataclass
class Config:
    system_type: str = ""
    system: dict = field(default_factory=dict)
        
def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(Config, cfg)
    return scfg

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg

"""
cfg = load_config("config/dietgs.yaml")

import threestudio

diff = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
import pdb; pdb.set_trace()
"""