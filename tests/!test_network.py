# TODO:Gradient Flow Check: Verify that gradients flow to all network heads (representation, dynamics, prediction) and that no weight matrix receives NaN gradients after an update step.

# TODO: Batch Size 1 Invariance: Ensure model forward passes do not fail when given a single sample batch (1, ...) (often caused by unhandled squeeze/unsqueeze dimensions or batch normalization).