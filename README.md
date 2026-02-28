# CineMatch v2 — Movie Recommender Web App

## Setup

1. Download your PKLs from Colab and place them in the `models/` folder:

```python
# Run this in Colab after training
from google.colab import files
files.download('/content/models/combined_matrix.pkl')
files.download('/content/models/movie_meta.pkl')
files.download('/content/models/title_index.pkl')
files.download('/content/models/titles_list.pkl')
```

2. Install and run:
```bash
pip install -r requirements.txt
python app.py
```

3. Open http://localhost:5000

