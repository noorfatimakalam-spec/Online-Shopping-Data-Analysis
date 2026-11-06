# Viva Preparation

## Project basics

**1. What is the objective of this project?**  
To analyze historical online retail transactions to understand customer purchasing behaviour, product performance and peak sales periods, and to identify customer segments using RFM and K-Means.

**2. Which dataset is used?**  
The UCI Online Retail dataset, containing 541,909 transaction-line records from 1 December 2010 to 9 December 2011.

**3. Why was this dataset selected?**  
It contains invoice, product, quantity, timestamp, price, customer and country information, which supports sales analysis and customer segmentation.

## Data preparation

**4. What is data preprocessing?**  
The process of preparing raw data for analysis by correcting data types, handling missing values, removing duplicates and applying justified filters.

**5. Why are cancellation invoices excluded from completed sales?**  
The UCI documentation states that invoice numbers beginning with C indicate cancellations. Treating them as ordinary completed sales would distort the sales measure.

**6. Why remove non-positive quantities and prices from the main sales analysis?**  
The primary analysis focuses on positive completed sales lines. Negative quantities can represent returns or adjustments and should be studied separately if return analysis is required.

**7. How is revenue calculated?**  
Line revenue is Quantity multiplied by UnitPrice. Total recorded revenue is the sum of line revenue.

**8. Is revenue the same as profit?**  
No. Profit requires cost information. The dataset supplies unit selling price but not product cost or operating expenses.

**9. Why are some customers excluded from RFM?**  
RFM requires a customer identifier to group transactions by customer. Rows with missing CustomerID cannot reliably be assigned to a customer.

## Machine learning

**10. What is RFM analysis?**  
Recency measures how recently a customer purchased, Frequency counts distinct orders, and Monetary measures total recorded spending.

**11. What is K-Means clustering?**  
An unsupervised learning algorithm that partitions observations into K groups by minimizing the within-cluster sum of squared distances to cluster centroids.

**12. Why is K-Means called unsupervised learning?**  
It finds structure in data without using predefined target labels.

**13. Why log-transform RFM features?**  
Spending and order frequency are often right-skewed. A logarithmic transformation reduces skew and lessens the influence of extreme values.

**14. Why standardize the features?**  
K-Means uses distance calculations. Standardization prevents variables with larger numerical scales from dominating the distance.

**15. What is the Elbow Method?**  
It plots the model's inertia for different K values. The bend in the curve can indicate a point after which additional clusters provide diminishing improvements.

**16. What is the Silhouette Score?**  
It measures how similar a sample is to its own cluster compared with other clusters. It ranges from -1 to 1; higher values generally indicate better separation.

**17. Does this model predict future customer purchases?**  
No. This project uses K-Means for descriptive segmentation, not supervised prediction.

**18. Why can't cluster 0 automatically be called VIP?**  
Cluster IDs are arbitrary. Names must be assigned after examining each cluster's actual RFM profile.

## Critical thinking

**19. What are the limitations of the dataset?**  
It is historical, December 2011 is incomplete, some customer IDs are missing, and it does not contain product cost, formal categories, demographics or browsing data.

**20. What is the practical use of the project?**  
It can help a retailer understand valuable customer groups, identify periods of higher activity and prioritize products for inventory review and marketing planning.

**21. What could be improved in future?**  
Use newer multi-year data, analyze returns separately, add product costs and inventory information, and compare multiple clustering methods.
