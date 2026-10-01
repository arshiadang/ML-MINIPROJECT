"""Refresh map, supporting tables and report without loading or fitting a model."""
from pathlib import Path
from heatwave.audit import review_saved_results
from heatwave.reporting import write_report

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    summary, _, _ = review_saved_results(root)
    write_report(root)
    print('Refreshed region map, F1 explanation, K comparison, noise support and report.')
    print('No model fitting or lookup edits occurred.')
    print(summary)
