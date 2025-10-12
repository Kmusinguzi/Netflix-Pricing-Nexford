# Netflix Pricing Optimization – Data Collection & EDA

## 🎯 Project Overview
This project aims to optimize Netflix’s subscription revenue and reduce churn through a dynamic pricing model that integrates user behavior and market data.

## 📊 Objective
To develop a predictive pricing system that aligns subscription costs with engagement, sentiment, and regional economic trends.

## 🧩 Dataset Overview
| Source | Description | Type | Access Method |
|---------|--------------|------|----------------|
| Netflix Subscriber Data | User demographics, plans, churn data | Internal | CSV |
| Competitor Pricing | Pricing tiers & regional offers | External | Web APIs |
| Social Media Sentiment | Pricing opinions | External | API/Scraping |
| Economic Indicators | GDP, inflation | Public | World Bank API |

Final cleaned dataset: **`netflix_pricing_final.csv`**

## ⚙️ Tools & Technologies
Python, Pandas, NumPy, BeautifulSoup, Seaborn, MySQL, Google Cloud Storage.

## 🧼 Data Processing Steps
1. **Data Cleaning:** Deduplication, missing value imputation, normalization, and error correction.
2. **Feature Engineering:** Created features such as `EngagementScore`, `ValuePerHour`, and `ValueSegment`.
3. **Integration:** Merged multiple sources using keys (`country`, `plan_type`, `date_key`).
4. **Quality Assurance:** Verified completeness, accuracy, and consistency.

## 🧠 Exploratory Data Analysis
- Visualized churn distribution, tenure vs churn, and correlations.
- Conducted t-tests for statistical significance.
- Identified strong negative correlation between price increases and churn.

## 🧑‍⚖️ Ethical Considerations
- All data anonymized and compliant with GDPR.
- Bias detection checks (e.g., churn distribution by gender).

## 📂 Repository Contents
/Netflix-Pricing-EDA
├── netflix_pricing_final.csv
├── Netflix_Pricing_EDA.ipynb
├── README.md
└── Presentation.pptx

## 🚀 Next Steps
- Model building (regression and machine learning)
- Elasticity modeling per region
- Dashboard integration with API