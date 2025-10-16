# 🌤 Lansing Temperature & ENSO Dashboard

**Author:** Joshua Nicholson  
**Course:** CMSE 830 — Foundations of Data Science
**Semester:** Fall 2025  

---

## 📘 Overview

This interactive Streamlit web app explores **temperature patterns in Lansing, Michigan** and how they relate to the **El Niño–Southern Oscillation (ENSO)**.  
It visualizes historical climate data and seasonal temperature variability through multiple lenses — from absolute averages and anomalies to winter patterns driven by ENSO phase.

---

## 🔍 What’s Inside

The app is divided into two main sections:

### **1️⃣ ENSO Analysis**
- **Absolute Temperature Trends:** Displays average high temperatures over time.  
- **Temperature Anomalies:** Highlights deviations from long-term averages to visualize warming or cooling periods.  
- **Winter ENSO Relationships:** Uses violin plots to compare winter (DJF) average high temperatures across ENSO phases — from strong El Niño (–4) to strong La Niña (+3).  
- **ENSO Phase Oscillation:** Visualizes how ENSO fluctuates through time, showing positive (red) and negative (blue) phases.

### **2️⃣ Lansing Trends**
- **Freeze Day Counts:** Shows the number of days below freezing for each year.  
- **Long-Term Trends:** Displays general warming trends and climate variability in Lansing.  
- Includes static figures for clarity and reproducibility.

---

## 🌎 About ENSO

The **El Niño–Southern Oscillation (ENSO)** is a recurring climate pattern in the Pacific Ocean that alternates between **El Niño (warm phase)** and **La Niña (cool phase)**.  
These shifts affect weather worldwide — including **winter temperatures in the U.S. Midwest**, where Lansing often experiences warmer winters during El Niño years and colder ones during La Niña.

---

## 🧮 Data

- **Source:** NOAA and derived local datasets (1960–2024).  
- **Variables Used:**
  - `Year`, `Month`, `high` (average monthly high temperature)
  - `ENSO_encoded` (ENSO phase index)
  - Derived columns: `Season_Year`, temperature anomalies, freeze-day counts. ENSO PHASE correlations

---
