Always run `conda activate circuit` before executing any shell command that involves Python, pytest, pip, or any project script. Prefix such commands with `conda run -n circuit` or source the environment first in a compound command, e.g.:

```bash
conda run -n circuit pytest tests/test_prune.py
conda run -n circuit python -m summarization --help
```

Never assume the environment is already active — shell state does not persist between tool calls.
