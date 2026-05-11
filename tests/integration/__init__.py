"""Opt-in integration tests for pd-ocr-synth.

Tests under this package touch real external services (currently only
the Hugging Face Hub) and are gated behind environment variables so
``make ci`` never reaches the network. See
``tests/integration/README`` (the docstring on
:mod:`tests.integration.test_publish_live_hf`) for the env-var
contract and how to run a single test live.
"""
