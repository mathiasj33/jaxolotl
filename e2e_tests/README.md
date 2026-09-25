## End-to-end tests

The provided end-to-end tests verify the behaviour of evaluation scripts with pretrained models.
Pretrained models are not bundled with the repository; each test is skipped unless its
final models exist under `runs/<env name>/<alg>/pretrained/models`.

To enable end-to-end tests:
```bash
pixi run pytest --e2e
```

To exclusively run end-to-end tests:
```bash
pixi run pytest --e2e -m e2e
```