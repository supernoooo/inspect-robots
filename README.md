# Colosseum LLM evaluation packages

This branch contains the Inspect Robots core and the YAM EEF plugin from
commit `0865dcac2f38dfdd0a4768a68bf4146ef394b69f` of
[supernoooo/inspect-robots](https://github.com/supernoooo/inspect-robots).
Python runtime source is preserved unchanged. Documentation, personal launch
scripts, tests, integrations, and unrelated plugins are omitted.

The core uses the fixed fork version `0.58.0+colosseum` so installation does not
depend on the original Git tags or history. The EEF plugin version is `0.1.0`.

## Install

Install both packages into the existing Colosseum client environment. Replace
`COMMIT_SHA` with the full commit SHA of this branch for reproducible installs.

```bash
uv pip install --python .venv/bin/python \
  'inspect-robots @ git+https://github.com/supernoooo/inspect-robots.git@COMMIT_SHA' \
  'inspect-robots-yam-eef @ git+https://github.com/supernoooo/inspect-robots.git@COMMIT_SHA#subdirectory=plugins/inspect-robots-yam-eef'
```

For a local checkout:

```bash
uv pip install --python .venv/bin/python . ./plugins/inspect-robots-yam-eef
```

The EEF plugin installs `inspect-robots-yam>=0.37,<0.38`. LLM control also
requires a separately installed `inspect-robots-agent` package.
`inspect-robots colosseum` additionally requires compatible `colosseum-client`
and `colosseum-policy-server` packages in the same environment. This branch
does not bundle those packages or local robot configuration files.

```bash
inspect-robots list embodiments
inspect-robots list policies
inspect-robots doctor --embodiment yam_eef
```

See the [YAM EEF README](plugins/inspect-robots-yam-eef/README.md) for its
coordinate system, workspace limits, and operator requirements. Validate the
robot configuration with an operator at the e-stop before enabling motion.

## License

[MIT](LICENSE). Original attribution is retained in the license file.
