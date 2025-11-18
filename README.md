# 🏙️ Tokyo Urban Isolation Index  
### Social Isolation Risk Modeling for Tokyo’s 23 Wards

This project builds a data-driven **Isolation Risk Index (IRI)** for Tokyo, integrating demographic, socioeconomic, accessibility, transit, and spatial clustering methods. It provides a replicable urban analytics workflow for identifying areas where older adults may face elevated isolation risk.

---

## 📌 Project Overview

Two complementary indices are constructed:

### 1. **Designed Isolation Index (D-IRI)**  
A theory-driven composite index using standardized indicators:
- 👵 Percent of residents aged 65+
- 🏠 Percent of older adults living alone
- 💴 Poverty rate
- 🏥 Accessibility score (distance to community resources)
- 🚉 Transit density (rail station density)

### 2. **PCA-Based Isolation Index**  
A data-driven index based on **PC1** from principal component analysis.

These allow comparison between a conceptual model and an empirical (unsupervised) model.

---

## ⚙️ Features & Capabilities

### **Data Ingestion**
- Loads demographics, SES data  
- Imports accessibility proxies  
- Processes railway station geodata and computes station density  

### **Feature Engineering**
- Z-score scaling  
- Access score computation  
- Transit score computation  
- Full dataset merging with spatial geometries  

### **Index Construction**
- Designed Isolation Index (D-IRI)  
- PCA Isolation Index (PC1)  

### **Statistical Analysis**
- 📈 Correlation analysis  
- 🧮 PCA loadings  
- 📉 OLS regression diagnostics  
- 🌐 Global Moran’s I (spatial autocorrelation)  
- 🗺️ LISA cluster detection (local spatial clusters)  

### **Visualization**
- Choropleth: Designed IRI  
- Choropleth: PCA IRI  
- Side-by-side comparison maps  
- LISA cluster map (High-High, Low-Low, etc.)

---

## 🔍 Key Findings (Current Results)

- The **D-IRI and PCA index strongly agree**, validating the conceptual model.  
- **Spatial clustering is significant** (Moran’s I ≈ 0.30, p ≈ 0.003).  
- Central wards form **Low-Low clusters** (low isolation surrounded by low).  
- Several outer wards show **higher isolation**, influenced by aging, single-living rates, poverty, and reduced transit/access.

---




