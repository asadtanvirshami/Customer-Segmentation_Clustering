# Customer-Segmentation_Clustering
# README.md

# Customer Segmentation with KMeans

This project uses KMeans clustering on the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) dataset to segment customers for marketing insights.

## Project Structure

customer-segmentation/
├── data/
│ └── marketing_campaign.csv
├── notebooks/
│ └── EDA_and_Modeling.ipynb
├── models/
│ ├── kmeans_model.pkl
│ └── scaler.pkl
├── visuals/
│ └── cluster_plots.png
├── requirements.txt
└── README.mdQ

## Dataset Features Used

- Income
- Recency
- Age
- Total Children
- Spending (on Wines, Fruits, Meat, Fish, Sweets, Gold)
- Deal, Web, Catalog, Store Purchases
- Web Visits/Month

## How to Run

1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Make sure `data/marketing_campaign.csv` is in place
4. Run Streamlit app:  
   `streamlit run app/streamlit_app.py`

## Model

- Clustering using **KMeans** (elbow method to select `k=5`)
- Visualization using PCA
- Scaler: `StandardScaler`

## Output

- Cluster visualization plot in `visuals/`
- Saved model and scaler in `models/`

## Bonus Ideas

- Cluster profiling for personas
- Advanced clustering: DBSCAN, Agglomerative
- Add classification to predict cluster from new customer profiles
#