# Online Shopping Data Analysis
### Customer Behaviour, Product Performance & Peak Sales Periods

An end-to-end Python data analytics and machine learning project using the **UCI Online Retail** transaction dataset.

## Project overview

This project analyzes historical online retail transactions to understand:
- Customer purchasing behaviour and customer value.
- Product demand and revenue contribution.
- Sales patterns by month, weekday and hour.
- Customer segments discovered using RFM analysis and K-Means clustering.

The project is designed to be reproducible: the source data is downloaded from the UCI Machine Learning Repository when the pipeline runs, and the analysis outputs are generated locally.

## Research questions

1. What are the overall sales and order patterns in the transaction data?
2. Which products and countries contribute most to recorded revenue?
3. When are sales and orders most active?
4. How do customers differ in recency, frequency and monetary value?
5. Can customers be grouped into interpretable segments using K-Means?

## Dataset

**UCI Online Retail**  
Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C5BW33  
Dataset page: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains 541,909 transaction-line records from a UK-based non-store online retailer, covering 1 December 2010 to 9 December 2011. It includes invoice number, stock code, product description, quantity, invoice date, unit price, customer ID and country.

The dataset is licensed under **CC BY 4.0**. Attribution is provided above. The raw dataset is not committed to this repository; the pipeline retrieves it from UCI.

## Technology

- Python 3.10+
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- ucimlrepo

## Repository structure

```text
Online_Shopping_Data_Analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   └── analysis.py
├── notebooks/
│   └── Online_Shopping_Data_Analysis.ipynb
├── reports/
│   ├── Project_Report.md
│   └── Viva_Preparation.md
├── data/
│   └── README.md
└── outputs/
    ├── README.md
    └── figures/
```

## How to run

### Option A: Google Colab (quickest)
1. Upload `notebooks/Online_Shopping_Data_Analysis.ipynb` to Google Colab.
2. Run all cells from top to bottom.
3. The notebook installs dependencies, retrieves the dataset and performs the analysis.
4. Review the model-selection plots and cluster profiles before interpreting the final segments.

### Option B: Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/analysis.py
```

The script creates CSV summaries and PNG figures in `outputs/`.

## Methodology

1. **Data audit:** inspect data types, missing values, duplicates, cancellations and invalid sales lines.
2. **Preprocessing:** remove exact duplicates; parse timestamps and numeric fields; isolate completed positive-quantity, positive-price sales lines.
3. **Feature engineering:** calculate line revenue as `Quantity × UnitPrice`.
4. **EDA:** calculate KPIs and analyze sales by time, product and country.
5. **RFM analysis:** calculate customer recency, frequency and monetary value.
6. **Clustering:** log-transform and standardize RFM features; compare K values with inertia and silhouette score; fit K-Means and profile the resulting groups.

## Data handling decisions and limitations

- Invoice numbers beginning with `C` indicate cancellations and are excluded from completed-sales analysis.
- Positive quantities and positive unit prices are retained for the primary sales analysis. Negative quantities/returns are not included in this sales measure; they should be analyzed separately if return behaviour is a research objective.
- Records without a CustomerID can contribute to product and time analysis but cannot be assigned to customer-level RFM segments.
- `LineRevenue` is an estimate of gross line sales (`Quantity × UnitPrice`), not profit. Product costs, shipping, tax and operating expenses are not supplied.
- The dataset has product codes and descriptions but no formal product-category column.
- The dataset does not contain customer demographics, website visits, clickstream, marketing campaign exposure or inventory levels.
- December 2011 is incomplete (data ends on 9 December 2011); monthly comparisons must flag this partial month.
- This is historical data from 2010–2011 and should not be treated as a representation of today's retail market.
- K-Means is an unsupervised descriptive segmentation method, not a future-purchase prediction model. Cluster numbers are arbitrary and segment names must be assigned from the measured profiles.

## Expected generated outputs

After running the analysis:
- `outputs/kpi_summary.csv`
- `outputs/monthly_sales.csv`
- `outputs/daily_sales.csv`
- `outputs/hourly_sales.csv`
- `outputs/country_performance.csv`
- `outputs/top_products.csv`
- `outputs/customer_rfm_segments.csv`
- `outputs/cluster_profiles.csv`
- Figures for monthly trends, weekday/hourly patterns, top countries, top products, elbow and silhouette evaluation, and customer segments.

## Academic integrity and reproducibility

Do not add made-up findings, charts or evaluation scores to the report. Run the notebook or script and copy the generated values into the report. Include the dataset citation. The `outputs/` folder is intentionally a placeholder until the analysis has been executed.

## References

1. Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*, 197–208. https://doi.org/10.1057/dbm.2012.17
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
