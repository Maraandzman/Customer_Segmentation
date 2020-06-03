# Customer_Segmentation: Projec Overview
---

- Create a tool that runs monthly to categorize each customer_id(msisdn)
- The measures used to categorize these customers are: Revenue, Recency & Frequency aka RFM
- The aim is to have 3 categories: High Value Customers, Low Value Customers, Mid Value Customers
- Present to marketing team to target specific customers for specific campaigns


# Code and Resources Used
---

Python Version: 3.7
Packages: pandas, numpy, sklearn, KMeans, matplotlib, seaborn

# Data
---
 -   MSISDN              object        
 -   LAST_RECHARGE_DATE  datetime64[ns]
 -   BTS_MU_CITY         object        
 -   BTS_MU_LAT          object        
 -   BTS_MU_LON          object        
 -   COUNTRY             object        
 -   TARIFF_TYPE         object        
 -   ASPU                float64       
 -   RCH_COUNT_VOUCHER   float64       
 -   REGION_CLUSTER      object        
 -  TERRITORY           object
 
 # EDA
 ---
 ![](Recency_hist.png)
