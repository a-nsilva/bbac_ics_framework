# BBAC Framework - Data Directory

This directory contains datasets and data files for the BBAC Framework.

## Dataset Structure

### Expected Files from bbac_ics_dataset

Clone the dataset repository into this directory:

```bash
git clone https://github.com/a-nsilva/bbac_ics_dataset.git
```

Expected dataset structure:
```
bbac_ics_dataset/
├── data/
│   ├── access_logs.csv          # Historical access request logs
│   ├── agent_profiles.json      # Agent behavioral profiles
│   ├── normal_patterns.csv      # Normal behavior baselines
│   ├── anomalies.csv           # Known anomaly examples
│   └── README.md               # Dataset documentation
└── sample_*.csv/json           # Auto-generated sample data (for testing)
```

## Data Format

### access_logs.csv

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Access request timestamp |
| agent_id | string | Unique agent identifier |
| agent_type | string | Type: 'robot' or 'human' |
| resource_id | string | Target resource identifier |
| action | string | Requested action (read/write/execute/etc.) |
| decision | string | Grant/Deny/RequireApproval |
| zone | string | Physical zone (production/maintenance/etc.) |
| context | JSON | Additional contextual information |

### agent_profiles.json

```json
{
  "agent_id": {
    "type": "assembly_robot",
    "normal_behavior": {
      "access_frequency": 120,
      "typical_resources": ["assembly_station_A"],
      "typical_actions": ["read", "write"],
      "active_hours": [8, 9, 10, 11, 13, 14, 15, 16]
    }
  }
}
```

## Sample Data

If the bbac_ics_dataset is not available, the framework will automatically generate sample data for testing:

- `sample_access_logs.csv` - Synthetic access logs
- `sample_agent_profiles.json` - Synthetic agent profiles

## Usage

```python
from src.data.dataset_loader import DatasetLoader

# Load dataset
loader = DatasetLoader(dataset_path="data/bbac_ics_dataset")
loader.load_all()

# Get statistics
stats = loader.get_statistics()
print(stats)
```

## Citation

If using the bbac_ics_dataset, please cite:

```bibtex
@dataset{bbac_ics_dataset,
  author = {Silva, A.N.},
  title = {BBAC ICS Dataset},
  year = {2025},
  url = {https://github.com/a-nsilva/bbac_ics_dataset}
}
```
