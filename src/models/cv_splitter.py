from collections.abc import Generator
import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        samples_info: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(len(samples_info))
        test_chunks = np.array_split(indices, self.n_splits)
        embargo_samples = int(len(samples_info) * self.embargo_pct)

        for test_idx in test_chunks:
            if len(test_idx) == 0:
                continue

            test_t0 = samples_info.iloc[test_idx[0]]["entry_time"]
            test_t1 = samples_info.iloc[test_idx[-1]]["exit_time"]

            train_mask = np.ones(len(samples_info), dtype=bool)
            train_mask[test_idx] = False

            for i in np.where(train_mask)[0]:
                sample_t0 = samples_info.iloc[i]["entry_time"]
                sample_t1 = samples_info.iloc[i]["exit_time"]

                if not (sample_t1 < test_t0 or sample_t0 > test_t1):
                    train_mask[i] = False

            test_end_loc = test_idx[-1]
            embargo_end_loc = min(test_end_loc + embargo_samples, len(samples_info))
            train_mask[test_end_loc:embargo_end_loc] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
