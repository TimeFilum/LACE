# LACE

**LACE** is a framework for studying and improving cross-resolution self-consistency in time-series foundation models.

The work investigates whether forecasts produced from aggregation-linked temporal resolutions remain behaviorally consistent, how resolution choice can affect downstream decisions, and how such inconsistencies can be mitigated without collapsing the forecasting hierarchy to a single resolution.

## Repository Status

This repository currently contains the core model implementations used in the LACE study.

The complete experimental pipeline, including the cross-resolution diagnostic framework, evaluation code, decision-level analysis, preprocessing utilities, experiment configurations, and scripts required to reproduce the full study, will be released upon publication of the final version of the work.

## Current Contents

```text
LACE/
├── models/
│   └── ...
└── README.md
```

The released model files contain the main LACE implementations and the interfaces used with the evaluated time-series foundation model backbones.

## Full Release

The final public release is planned to include:

* Cross-resolution self-consistency diagnostics
* Accuracy and coherence evaluation
* Decision-level evaluation
* Experiment configurations
* Reproduction scripts


## Paper

The manuscript describing LACE is currently under submission.

Paper details and citation information will be added following publication.

## License

License information will be provided with the public release.
