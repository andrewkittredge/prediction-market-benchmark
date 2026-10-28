# Training data cutoff per model, hand-copied from each model's page under
# https://platform.claude.com/docs/en/models/ (the Models API doesn't report
# it). Older models have earlier cutoffs, so more of the dataset's markets
# closed after what they could have seen. Logged on the run so results can be judged against
# the leakage caveat in README.md.
TRAINING_DATA_CUTOFF = {
    "claude-fable-5-1": "2026-06",
    "claude-opus-5-5": "2026-06",
    "claude-sonnet-5-5": "2026-06",
    "claude-haiku-5-5": "2026-06",
    "claude-opus-5": "2026-05",
    "claude-fable-5": "2026-01",
    "claude-sonnet-5": "2026-01",
    "claude-opus-4-8": "2026-01",
    "claude-opus-4-7": "2026-01",
    "claude-sonnet-4-6": "2026-01",
    "claude-opus-4-6": "2025-08",
    "claude-opus-4-5": "2025-08",
    "claude-sonnet-4-5": "2025-07",
    "claude-haiku-4-5": "2025-07",
}
