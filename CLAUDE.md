# CLAUDE.md - Application Rationalization Tool

## Project Overview

**apprat** is a Python desktop application that assists in application portfolio rationalization by clustering applications based on multi-dimensional similarity analysis.

### Core Functionality

1. **Application Ingestion**: Import portfolio of applications via CSV files
2. **Feature Matrix Generation**: Research applications via internet to extract features (or load pre-scored matrix from CSV)
3. **Multi-dimensional Scoring**: Score each application across all identified features
4. **Clustering Analysis**: Identify similar applications using proximity analysis
5. **Similarity Ranking**: Provide ranked suggestions for consolidation opportunities

### Data Input Format

The application supports CSV-based data input with two files:

1. **Application Matrix CSV**: Contains applications and their scores across dimensions
   - Rows: Applications
   - Columns: Numeric dimension indices (e.g., 0, 1, 2, 3, ...)
   - Values: Numeric scores for each application on each dimension

2. **Dimension Mapping CSV**: Links dimension indices to their actual meanings
   - Column 1: Dimension index (0, 1, 2, ...)
   - Column 2: Dimension name/description

Example Application Matrix CSV:
```
Application,0,1,2,3,4
App A,5,3,0,4,2
App B,4,3,1,5,2
App C,0,0,5,0,4
```

Example Dimension Mapping CSV:
```
Index,Dimension
0,Cloud Native
1,Mobile Support
2,Data Analytics
3,User Management
4,API Integration
```

### Key Design Consideration

**Proportional Similarity**: The clustering algorithm must use proportional similarity rather than absolute feature counts. Two simple applications with 5/5 matching features should rank as more similar than two complex applications with 50/100 matching features, even though the complex apps share more absolute features.

## Technology Stack

### Core Technologies
- **Language**: Python 3.x
- **Desktop Framework**: TBD (options: PyQt6, Tkinter, or web-based with Electron)
- **Data Processing**: pandas, numpy
- **Clustering**: scikit-learn (DBSCAN, K-means, or hierarchical clustering)
- **Web Research**: TBD (options: requests + BeautifulSoup, Selenium, or API integrations)
- **Data Visualization**: matplotlib, seaborn, or plotly

## Architecture

### Data Flow
```
Applications Input
    ↓
Feature Extraction (Internet Research)
    ↓
Feature Matrix Generation
    ↓
Multi-dimensional Scoring
    ↓
Clustering Analysis
    ↓
Proximity Ranking
    ↓
Results Display
```

### Core Components

1. **Application Manager**: Handle input and storage of application data
2. **Research Engine**: Automated internet research to extract features
3. **Feature Matrix**: Store and manage feature dimensions
4. **Scoring Engine**: Calculate scores for each app across dimensions
5. **Clustering Engine**: Perform similarity analysis
6. **Visualization Layer**: Display results and rankings
7. **Desktop UI**: User interface for interaction

## Similarity Algorithm

The clustering must account for proportional similarity with a bias towards applications WITH features.

### Proportional Similarity (Primary Method)

**Core Principle**: Applications are similar when they share active features, not when they both lack features.

**Algorithm**:
1. Identify shared active dimensions (both apps have score > 0)
2. Calculate similarity for shared dimensions only
3. Apply penalty based on proportion of shared vs. total dimensions
4. Return 0.0 for apps with no features
5. Return very low similarity for apps with no shared active dimensions

**Formula**:
```
shared_dims = active_dims(app1) ∩ active_dims(app2)
base_similarity = weighted_avg_of_score_similarity_in_shared_dims
shared_ratio = len(shared_dims) / len(active_dims(app1) ∪ active_dims(app2))
final_similarity = base_similarity × (0.5 + 0.5 × shared_ratio)
```

**Key Features**:
- Two apps with 5/5 matching dimensions score higher than apps with 50/100 matching
- Apps with all zeros are NOT similar (score = 0.0)
- Apps with no shared features get very low similarity (< 0.1)
- Normalizes by applicable features per pair, not global feature set

### Other Similarity Methods

**Jaccard Similarity**:
- Based on set intersection/union of active dimensions
- Ignores score magnitudes
- Returns 0.0 for apps with no features

**Cosine Similarity**:
- Traditional vector-based similarity
- Uses all dimensions (including zeros)
- Good for comparing score patterns

## Development Phases

### Phase 1: Core Data Pipeline
- Application data model
- Feature extraction framework
- Feature matrix structure

### Phase 2: Research & Scoring
- Internet research automation
- Feature scoring system
- Data persistence

### Phase 3: Clustering & Analysis
- ✅ Implement proportional clustering algorithm
- ✅ Proximity calculation
- ✅ Ranking system
- ✅ Automatic cluster determination (3 methods)
  - Silhouette-optimized (best quality)
  - Threshold-based (natural groups)
  - DBSCAN (density-based, finds outliers)

### Phase 4: Desktop Interface
- GUI framework selection
- Application management interface
- Results visualization
- Export functionality

## Project Structure (Proposed)

```
apprat/
├── src/
│   ├── core/
│   │   ├── application.py       # Application data model
│   │   ├── feature_matrix.py    # Feature matrix management
│   │   └── database.py          # Data persistence
│   ├── research/
│   │   ├── researcher.py        # Internet research engine
│   │   └── feature_extractor.py # Feature extraction logic
│   ├── analysis/
│   │   ├── scoring.py           # Scoring engine
│   │   ├── clustering.py        # Clustering algorithms
│   │   └── similarity.py        # Similarity calculations
│   ├── ui/
│   │   ├── main_window.py       # Main application window
│   │   ├── visualization.py     # Results visualization
│   │   └── components/          # UI components
│   └── utils/
│       ├── config.py            # Configuration management
│       └── helpers.py           # Utility functions
├── tests/
├── data/                        # Local data storage
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Next Steps

1. Choose desktop framework (PyQt6 recommended for native desktop feel)
2. Design application data schema
3. Implement basic application CRUD operations
4. Build research engine prototype
5. Develop proportional similarity algorithm
6. Create basic UI prototype

## Notes

- Consider API rate limits for internet research
- May need LLM integration for feature extraction from unstructured data
- Security: Handle API keys and credentials properly
- Performance: Cache research results to avoid redundant API calls
- Consider offline mode with manual feature entry
