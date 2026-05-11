You are writing research code. Optimize for readability, traceability of assumptions, and fast iteration.

## Priorities

**Readability over cleverness.** Prefer explicit, flat code over abstractions. A function that does one clear thing is better than a general one that handles hypothetical cases. Inline small helpers when naming them adds no clarity.

**Trace assumptions explicitly.** When a function depends on a non-obvious invariant (tensor shape, adjacency convention, node ordering, threshold semantics), state it in a short inline comment. This is the one place comments are expected. Example: `# adj[target, source] — same convention as AttrGraph`.

**Fast iteration over robustness.** Don't add error handling, fallbacks, or validation for scenarios that can't happen in the current research workflow. Validate at true boundaries (CLI args, file I/O, API responses) but trust internal data that you constructed.

**No defensive programming inside pipelines.** Don't re-check types or shapes mid-pipeline when the caller already guarantees them. Don't write `if x is None` guards for values that are never None in practice.

**Minimal abstraction.** Don't create base classes, registries, or plugin systems speculatively. Three similar blocks of research code is better than a premature abstraction that obscures what each experiment does differently.

**Tests cover behavior, not coverage.** Write tests that verify a concrete research invariant (e.g., pruning retains target logit, clustering produces non-overlapping supernodes). Skip tests for internal helpers unless they encode a subtle invariant worth pinning down.

## Style

- Type hints on function signatures; skip them on obvious one-liners.
- No multi-line docstrings. A single-line summary is enough if the name isn't self-explanatory.
- Use modern Python type syntax: `X | Y`, `X | None`, `list[T]`, `dict[K, V]` — not `typing.*` forms.
- Keep tensor shape comments on the line that creates the tensor when the shape is non-trivial.
