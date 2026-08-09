import json
import csv
import os
from typing import Dict, Any, List, Optional


class TrainingHistory:
    """Manages epoch-by-epoch training metrics and exports history to CSV/JSON."""
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_epoch(self, epoch_metrics: Dict[str, Any]) -> None:
        """Appends metrics dict for a completed epoch."""
        self.records.append(epoch_metrics)

    def save_json(self, json_path: str) -> None:
        """Saves full history to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def save_csv(self, csv_path: str) -> None:
        """Saves full history to CSV format."""
        if not self.records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        fieldnames = list(self.records[0].keys())
        # Gather any extra keys if present in later epochs
        for r in self.records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def save_summary(
        self,
        summary_path: str,
        best_epoch: int,
        best_val_accuracy: float,
        test_accuracy: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Saves high-level summary JSON of training run."""
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        summary = {
            "best_epoch": best_epoch,
            "best_validation_accuracy": float(best_val_accuracy),
            "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
            "total_epochs": len(self.records),
        }
        if extra_metadata:
            summary.update(extra_metadata)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
