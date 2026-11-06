# Project Report
## Online Shopping Data Analysis: Customer Behaviour, Product Performance and Peak Sales Periods

**Submitted by:** 
1.Noor Fatima Kalam **enrollment number:** 2500100428
2.Nooreen Fatima **enrollment number:** 2500100051
**Course / programme:** B.tech in CSE (CCAI)
**institution:**   Integral University
**Academic year:** 2026-27

## Abstract

This project presents an exploratory and machine learning-based analysis of historical online retail transaction data. The objective is to examine customer purchasing behaviour, product performance and temporal sales patterns, and to identify customer groups using Recency, Frequency and Monetary (RFM) analysis with K-Means clustering. The dataset is obtained from the UCI Machine Learning Repository and contains transaction-level information such as invoice number, product code, quantity, invoice timestamp, unit price, customer identifier and country. Data preparation includes duplicate removal, timestamp and numeric type conversion, exclusion of cancellation invoices from completed-sales analysis, and calculation of line revenue. Descriptive analysis is used to study sales trends across months, weekdays and hours, as well as country and product contributions. Customer-level RFM features are transformed and standardized before clustering. The Elbow Method and Silhouette Score are used to support cluster selection. The final findings and recommendations are to be populated from the reproducible analysis outputs.

**Keywords:** Online retail, data analysis, customer behaviour, RFM, K-Means, sales trends, Python.

## Chapter 1: Introduction

### 1.1 Background

Online retail platforms produce transaction data that can support decisions about customer retention, product planning and sales operations. By examining historical orders, an organization can identify high-demand products, understand when customers place orders and distinguish customer groups by purchasing activity and spending.

### 1.2 Problem statement

Raw transaction records are difficult to interpret directly. A structured analytical process is required to clean the records, summarize sales behaviour, identify temporal and product-level patterns, and derive interpretable customer segments.

### 1.3 Aim

To analyze online shopping transaction data to understand customers, products and peak sales periods, and to apply an unsupervised learning method for customer segmentation.

### 1.4 Objectives

1. Inspect and preprocess the transaction dataset.
2. Calculate key sales and order indicators.
3. Analyze monthly, weekday and hourly sales patterns.
4. Identify products and countries with the highest recorded revenue.
5. Calculate customer Recency, Frequency and Monetary metrics.
6. Apply K-Means clustering and evaluate candidate cluster counts.
7. Present evidence-based business recommendations.

### 1.5 Research questions

- Which products and countries contribute the greatest recorded revenue?
- Which months, weekdays and hours show the highest sales activity?
- How do customers differ in recency, order frequency and monetary value?
- Can RFM features reveal useful customer segments?

## Chapter 2: Dataset Description

### 2.1 Source

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

The dataset describes transactions from a UK-based non-store online retailer selling mainly unique all-occasion gifts. It contains 541,909 transaction-line records and spans 1 December 2010 to 9 December 2011.

### 2.2 Variables

| Variable | Description |
|---|---|
| InvoiceNo | Invoice identifier; values beginning with C indicate cancellations |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Number of units on a transaction line |
| InvoiceDate | Date and time of the invoice |
| UnitPrice | Unit selling price in GBP |
| CustomerID | Customer identifier; may be missing |
| Country | Customer's country |

### 2.3 Scope and limitations

The dataset does not provide product cost, profit margin, formal product categories, customer demographics, website browsing sessions, advertising exposure or inventory levels. Revenue is therefore calculated as transaction quantity multiplied by unit price; it is not profit. December 2011 is partial because the dataset ends on 9 December 2011.

## Chapter 3: Methodology

### 3.1 Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn and ucimlrepo.

### 3.2 Data preprocessing

The analysis removes exact duplicate rows, converts timestamps and numeric fields to appropriate data types, and excludes invoice numbers beginning with C from completed-sales calculations. The main sales analysis retains positive quantities and positive unit prices. Records with missing CustomerID remain eligible for product and time analyses but are excluded from customer-level RFM calculations.

### 3.3 Revenue calculation

For each transaction line:

\[
\text{Line Revenue} = \text{Quantity} \times \text{Unit Price}
\]

Total recorded revenue is the sum of line revenue over the retained sales records.

### 3.4 Exploratory data analysis

The project calculates overall indicators and groups transactions by month, weekday, hour, country and product. Graphs are generated to make the patterns easier to interpret.

### 3.5 RFM analysis

- **Recency:** Number of days between the customer's latest purchase and the analysis snapshot date.
- **Frequency:** Number of distinct invoices associated with the customer.
- **Monetary:** Sum of recorded line revenue for the customer.

The snapshot date is set to one day after the latest transaction date in the identified-customer sales records.

### 3.6 K-Means clustering

The three RFM variables are log-transformed using `log1p` and standardized with `StandardScaler`. Candidate K values are evaluated using inertia and Silhouette Score. The final cluster solution is profiled using average and median RFM values. Cluster labels are arbitrary identifiers; descriptive segment names must be assigned only after reviewing the resulting profiles.

### 4.1 Data quality and cleaning

- Original rows: [from data_cleaning_audit.csv]
- Rows after exact deduplication: [value]
- Cancellation rows excluded: [value]
- Positive sales rows retained: [value]
- Sales rows missing CustomerID: [value]

Discuss how missing customer identifiers and cancellation handling affect the analysis.

### 4.2 Overall performance

- Total recorded revenue: £[value]
- Distinct invoices: [value]
- Identified customers: [value]
- Distinct products: [value]
- Countries represented: [value]

### 4.3 Sales timing

Insert `monthly_revenue.png` and `weekday_hourly_patterns.png`. Identify the strongest month, weekday and hour from the generated summaries. Explicitly note that December 2011 is incomplete.

### 4.4 Product and country performance

Insert `top_products.png` and `top_countries.png`. Discuss the leading products and countries by recorded revenue. Avoid describing revenue as profit.

### 4.5 Customer segmentation

Report the selected K, its Silhouette Score, and the cluster profile table. Describe segments by their observed Recency, Frequency and Monetary values rather than cluster number alone.

### 4.6 Summary of findings

1. [Finding supported by monthly/daily/hourly output]
2. [Finding supported by product/country output]
3. [Finding supported by RFM/cluster profile output]

## Chapter 5: Recommendations and Conclusion

### 5.1 Recommendations

Use the analysis to support recommendations such as:
- Plan promotional campaigns around observed periods of stronger order activity.
- Prioritize inventory review for products with consistently high demand.
- Consider retention initiatives for customer segments with high monetary value and frequent purchasing.
- Consider reactivation communication for groups with high recency and low frequency, subject to appropriate customer consent and marketing rules.
- Improve customer identifier completeness to strengthen customer-level analytics.


### 5.2 Conclusion

This project demonstrates how transaction-level online retail data can be transformed into useful descriptive insights and customer segments. It combines data preprocessing, exploratory analysis, feature engineering and unsupervised learning. The practical value of the findings depends on the historical coverage and limitations of the source data. Future work could use more recent multi-year data, product costs, formal product categories, inventory data and campaign information.

### 5.3 Future scope

- Compare K-Means with hierarchical clustering and DBSCAN.
- Add purchase forecasting using a time-series model if sufficient continuous history is available.
- Analyze returns separately rather than excluding them from the main completed-sales measure.
- Incorporate cost and inventory data to study profit and stock planning.
- Use newer transaction datasets to assess whether patterns persist.

## References

1. Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*, 197–208. https://doi.org/10.1057/dbm.2012.17
3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
