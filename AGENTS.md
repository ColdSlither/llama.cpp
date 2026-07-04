# ColdSlither/llama.cpp — Private Fork

> This is a **private fork** of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) owned by **ColdSlither**.
>
> Working branch: `feature/kv-compression`
>
> Upstream: `https://github.com/ggerganov/llama.cpp`
> Remote:  `coldslither` → `https://github.com/ColdSlither/llama.cpp.git`

This repo is an AI-assisted research fork. All work — commits, pushes, feature development, testing — is done with my approval via an AI coding agent. Treat it as an extension of my engineering workstation, subject to the same quality standards I enforce for myself.

## Workflow

- **Commit and push freely** to `feature/*` branches on the `coldslither` remote. Use `Assisted-by:` in commit messages.
- **PRs to upstream** — those follow upstream's rules. The upstream AGENTS.md is at `AGENTS.md.upstream` (if preserved) or lives at `https://github.com/ggerganov/llama.cpp`. Before opening a PR to `ggerganov/llama.cpp`, pause and confirm with me.
- **Feature branches** are experimental. Push early, push often. No permission needed per-push.
- **Commits** should be concise and accurate. Use conventional commit style (`llama : <short description>`). No unicode, no verbose AI-isms.

## Code Quality Standards (mandatory)

Same as upstream for anything that might eventually go upstream:

- Keep comments concise. Do not restate what the code says.
- Avoid emdashes, unicode arrows, non-ASCII punctuation in commits/comments.
- Prefer reusing existing infrastructure over new subsystems.
- Read relevant files before writing code. Blend in with surrounding patterns.
- For large or invasive changes, pause and confirm scope.

## Working with Upstream (ggerganov/llama.cpp)

When preparing contributions to the upstream repo:

- Check existing issues/PRs first.
- I must understand every change fully before submission.
- PR descriptions, commit messages, and reviewer responses are mine to write.
- Automated PR submissions can result in a contributor ban — never push PRs to upstream without explicit go-ahead from me.

## Useful Resources

- [Upstream contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggerganov/llama.cpp/issues) and [Existing PRs](https://github.com/ggerganov/llama.cpp/pulls)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)
- [Build documentation](docs/build.md)
- [Server usage](tools/server/README.md)
- [Server development](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md)
- [Auto parser](docs/autoparser.md)
- [Jinja engine](common/jinja/README.md)
