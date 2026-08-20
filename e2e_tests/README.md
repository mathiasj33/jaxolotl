## End-to-end tests

The provided end-to-end tests verify the behaviour of evaluation scripts with the pretrained models.

To enable end-to-end tests:
```bash
pixi run pytest --e2e
```

To exclusively run end-to-end tests:
```bash
pixi run pytest --e2e -m e2e
```