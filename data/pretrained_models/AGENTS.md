# Model release requirements

- Follow [the model licensing policy](../../legal/MODEL-LICENSING.md).
- NVIDIA-provided model releases default to the NVIDIA Open Model License;
  source code remains Apache-2.0.
- Every model needs a License section in its model card, a complete license
  copy and an attribution notice. Include these in standalone exports too.
- Alternative licenses require a reviewed exception recorded in both the card
  and the exception manifest. Preserve applicable third-party terms.
- Run `python scripts/check_model_licenses.py` from the repository root.
- Do not modify weights or serialized configs just to change licensing notices.
