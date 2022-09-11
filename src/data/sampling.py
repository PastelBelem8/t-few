from collections import defaultdict

import functools
from logging import warning
import numpy as np


class BalancedLoader:
    def __init__(self, labels):
        self.labels = labels
        self.labels2ids = defaultdict(list)

        for id, label in enumerate(labels):
            self.labels2ids[label].append(id)
    
    def get(self, num_samples: int, seed: int, strict: bool=False) -> list:
        """Get the specified number of samples from each class.
        
        If strict is enabled it returns equal number of examples
        for each class, which goes up to the minimum maximum value
        available. When it is false, it returns the maximum number
        of examples up to ``num_samples`` for each class. For instance,
        if num_samples=5, but class 2 only has 2 examples, then it
        will return the following counts {
            "lab0": 5,
            "lab1": 5,
            "lab2": 2,
        } whereas strict would only return {
            "lab0": 2,
            "lab1": 2,
            "lab2": 2,
        }.
        """
            
        results = {}
        rand = np.random.default_rng()

        for label, ids in self.labels2ids.items():
            # Determine the actual number of samples to get
            # Since it is not guarantee that we'll have enough labels
            # for each label. We therefore sample as much examples of
            # each label as we can.
            actual_num_samples = min(num_samples, len(ids))

            if actual_num_samples < num_samples:
                print(f"Warning: Sampling {actual_num_samples} for label {label} (instead of {num_samples})")

            sample_ids = np.array(ids)
            rand.shuffle(sample_ids)
            results[label] = sample_ids[:actual_num_samples].tolist()
        
        if strict:
            min_val = min(map(len, results.values()))
            results = {lab: ids[:min_val] for lab, ids in results.items()}
        
        return functools.reduce(lambda l1, l2: l1+l2, results.values())

