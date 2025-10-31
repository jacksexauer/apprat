# Quick Start Guide

Get started with **apprat** in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run the Application

```bash
python run.py
```

## Step 3: Load Sample Data

1. In the application window, click the first "Browse..." button (next to "Application Matrix")
2. Navigate to the `data/` folder and select `sample_applications.csv`
3. Click the second "Browse..." button (next to "Dimension Mapping")
4. Select `sample_dimensions.csv` from the `data/` folder
5. Click "Load Data"

You should see a success message showing 12 applications loaded!

## Step 4: Explore Similarity Rankings

1. Click on the "Similarity Rankings" tab
2. Select a similarity method (try "proportional" first)
3. Set how many pairs to show (default: 20)
4. Click "Calculate Similarity"

You'll see a ranked list of application pairs, sorted by similarity. The top pairs are the most similar applications that might be candidates for consolidation.

## Step 5: Try Clustering

1. Click on the "Clusters" tab
2. Set the number of clusters (try 3)
3. Select the clustering method (try "proportional")
4. Click "Calculate Clusters"

The applications will be grouped into clusters based on their similarity.

## Understanding the Results

### Similarity Scores

- **1.0**: Identical applications
- **0.8-1.0**: Very similar - strong consolidation candidates
- **0.6-0.8**: Similar - consider consolidation
- **0.4-0.6**: Somewhat similar - review for overlap
- **< 0.4**: Not very similar

### Proportional Similarity

The proportional similarity algorithm ensures fair comparison:

- Simple applications with high feature overlap rank higher than complex applications with lower proportional overlap
- Example:
  - HR Management ↔ Employee Portal (similar core features) = **High Similarity**
  - CRM System ↔ Email Marketing (many features, partial overlap) = **Medium Similarity**

## Using Your Own Data

### Create Your Application Matrix CSV

1. First column: Application names
2. Remaining columns: Numeric dimension indices (0, 1, 2, ...)
3. Values: Scores for each dimension (0 = not present, higher = stronger)

Example:
```csv
Application,0,1,2,3,4
My App 1,5,4,3,0,2
My App 2,4,5,3,1,1
My App 3,0,0,5,4,5
```

### Create Your Dimension Mapping CSV (Optional)

```csv
Index,Dimension
0,Cloud Native Architecture
1,Mobile Application Support
2,Advanced Analytics
3,Integration Capabilities
4,Security Features
```

### Load and Analyze

1. Load your CSV files using the Browse buttons
2. Explore the three tabs:
   - **Application Matrix**: View your raw data
   - **Similarity Rankings**: Find similar application pairs
   - **Clusters**: Group applications into clusters

## Tips

- Start with the "proportional" similarity method - it handles applications of different complexity fairly
- Use the "Jaccard" method if you only care about which dimensions are present (not their scores)
- Use the "cosine" method for traditional vector-based similarity
- Export results by selecting rows and copying to your clipboard

## Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Check [CLAUDE.md](CLAUDE.md) for technical architecture details
- Run tests: `pytest tests/`

## Troubleshooting

**"No module named 'PyQt6'"**
- Run: `pip install -r requirements.txt`

**Application won't start**
- Make sure you have Python 3.8 or higher: `python --version`
- Try: `python3 run.py` instead of `python run.py`

**CSV loading error**
- Check that your CSV has headers
- Ensure dimension columns are numeric (0, 1, 2, not "A", "B", "C")
- Verify no empty rows or columns

Need help? Open an issue on GitHub!
