# apprat

Cluster applications by similarity and rank them using proportional similarity analysis.

## Description

**apprat** is a Python desktop application that assists in application portfolio rationalization. It performs multi-dimensional clustering to identify similar applications that may be candidates for consolidation.

### Key Features

- **CSV-based data input**: Load application scores and dimension mappings from CSV files
- **Proportional similarity**: Advanced algorithm that correctly weighs similarity - two simple apps with 5/5 matching features rank higher than complex apps with 50/100 matches
- **Multiple similarity methods**: Choose from proportional, Jaccard, or cosine similarity
- **Automatic clustering** 🎯: Dynamically determines optimal number of clusters - no guessing required!
  - Silhouette-optimized (best quality)
  - Threshold-based (natural groups)
  - DBSCAN (finds outliers)
- **Cluster feature analysis** 📊: Understand why apps are grouped together
  - Identifies significant features per cluster
  - Shows which capabilities connect applications
  - Helps explain consolidation recommendations
- **Manual clustering**: Optional manual control with configurable cluster counts
- **Desktop GUI**: User-friendly PyQt6 interface with split-view cluster details
- **Comprehensive analysis**: View similarity rankings, clusters, and detailed comparisons

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install as Package

```bash
pip install -e .
```

## Usage

### Quick Start

1. Run the application:
```bash
python run.py
```

2. Load your data:
   - Click "Browse..." next to "Application Matrix" and select your application CSV file
   - Optionally, click "Browse..." next to "Dimension Mapping" and select your dimension mapping CSV
   - Click "Load Data"

3. Analyze:
   - **Similarity Rankings**: Calculate and view the most similar application pairs
   - **Clusters**: Group applications into clusters based on similarity

### Sample Data

Sample CSV files are provided in the `data/` directory:
- `sample_applications.csv`: Example application matrix with 12 applications
- `sample_dimensions.csv`: Example dimension mappings

Try loading these files to explore the application's capabilities.

## CSV File Format

### Application Matrix CSV

Contains applications (rows) and their scores across dimension indices (columns):

```csv
Application,0,1,2,3,4
App A,5,3,0,4,2
App B,4,3,1,5,2
App C,0,0,5,0,4
```

- **First column**: Application names
- **Remaining columns**: Numeric dimension indices (0, 1, 2, ...)
- **Values**: Numeric scores for each dimension

### Dimension Mapping CSV (Optional)

Maps dimension indices to human-readable names:

```csv
Index,Dimension
0,Cloud Native
1,Mobile Support
2,Data Analytics
3,User Management
4,API Integration
```

- **First column**: Dimension index
- **Second column**: Dimension name/description

## Similarity Algorithm

The proportional similarity algorithm ensures fair comparison with a bias towards applications WITH features:

### Key Features

1. **Proportional Fairness**
   - Simple apps with 5/5 matching dimensions (100%) > Complex apps with 50/100 matching dimensions (50%)
   - Normalizes by applicable features, not absolute feature counts

2. **Feature-Biased Scoring** 🎯
   - **Apps with no features are NOT similar** (similarity = 0.0)
   - Apps with no shared features get very low similarity (< 0.1)
   - Only shared active features contribute to similarity
   - Prevents false positives from undefined/empty applications

3. **Shared Feature Emphasis**
   - Focuses on dimensions where BOTH apps have non-zero scores
   - Penalizes asymmetric dimensions (only one app has the feature)
   - Rewards high overlap in active dimensions

### Example Comparisons

| Scenario | App 1 Features | App 2 Features | Similarity | Interpretation |
|----------|----------------|----------------|------------|----------------|
| Both empty | None (all 0s) | None (all 0s) | 0.00 | NOT similar (undefined) |
| No overlap | {0,1,2} | {3,4,5} | ~0.01 | NOT similar (different) |
| Partial overlap | {0,1,2,3} | {0,1,4,5} | ~0.67 | Similar (2/4 shared) |
| High overlap | {0,1,2} (identical) | {0,1,2} (identical) | 1.00 | Very similar |
| Simple 100% | 3/3 features match | 3/3 features match | 1.00 | Perfect match |
| Complex 50% | 3/6 features match | 3/6 features match | ~0.67 | Partial match |

**Business Value**: The algorithm only suggests consolidation when applications truly share features, avoiding false recommendations from undefined or completely different applications.

## Project Structure

```
apprat/
├── src/
│   ├── core/              # Data models and CSV loading
│   ├── analysis/          # Similarity and clustering algorithms
│   ├── ui/                # Desktop interface
│   └── main.py            # Application entry point
├── data/                  # Sample data files
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── run.py                 # Launch script
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development notes.

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.