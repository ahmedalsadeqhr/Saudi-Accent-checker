# Accuracy Evaluation Report

Model: badrex/mms-300m-arabic-dialect-identifier
Dataset: ArabicSpeech/ADI17 (test split, via datasets-server.huggingface.co)
Sample plan: {'KSA': 20, 'UAE': 10, 'EGY': 15, 'JOR': 15, 'MOR': 15}
Generated: 2026-07-30T00:00:00 (provenance lines added retroactively to an existing report; see note below)

Overall accuracy: 68.0% (51/75)
Skipped clips: 0

## Per-bucket accuracy
- Egyptian: 33.3%
- Gulf: 73.3%
- Levantine: 60.0%
- Maghrebi: 100.0%

## Confusion matrix (truth bucket rows, predicted label columns)
```
Truth \ Predicted Egyptian    Gulf        Levantine   MSA         Maghrebi    
Egyptian          5           3           4           1           2           
Gulf              0           22          6           0           2           
Levantine         1           2           9           1           2           
Maghrebi          0           0           0           0           15          
```

Note: MSA is not evaluated against ground truth here (the ADI17 source dataset has no MSA samples); it may still appear as a (mis)prediction in the confusion matrix.

Note: the provenance lines above (Model/Dataset/Sample plan/Generated) were added manually after the fact to bring this already-committed report in line with the updated `build_report()` output. The accuracy numbers themselves are from the original live run and were not regenerated in this pass (no live network/model run was performed as part of this fix).