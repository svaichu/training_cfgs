# training_cfgs

Standalone config manager migrated from `svaichu/robotdataset`.

## Quick start

```python
from training_cfgs import Config

cfg = Config.from_file("config.yaml")
cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2)
sweep = cfg.to_sweep()
```
