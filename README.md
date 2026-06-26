\# 🛡️ FraudGuard AI: Real-Time Anomaly Detection Engine



<div align="center">



\[!\[Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)

\[!\[XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-blue?logo=xgboost)](https://xgboost.readthedocs.io/)

\[!\[Streamlit App](https://img.shields.io/badge/Streamlit-Production\_UI-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\*An end-to-end machine learning pipeline and interactive web dashboard engineered to detect fraudulent credit card transactions in extreme class-imbalance environments.\*



</div>



\---



\## 📖 The "Accuracy Paradox" (Problem Statement)

In financial fraud detection, standard classification accuracy is a dangerous metric. In this dataset of \*\*284,807 transactions, only 0.17% (492 cases) are fraudulent\*\*. 



A baseline model that simply predicts "Normal" for every transaction achieves an accuracy of \*\*99.83%\*\*, while allowing \*\*100% of fraudulent activity\*\* to pass through undetected. This project abandons accuracy in favor of optimizing the \*\*F1-Score, Precision, and Recall\*\*, balancing the detection of cybercrime against the operational cost of blocking innocent customers.



\---



\## 🧠 System Architecture \& Engineering



\### 1. Data Pipeline \& Preprocessing

\* \*\*Feature Scaling:\*\* Applied `RobustScaler` to strictly normalize highly skewed and unbounded features (`Time` and `Amount`), ensuring extreme transaction values do not disrupt model gradients.

\* \*\*Stratified Partitioning:\*\* Engineered customized train/test splits that strictly enforce the exact 99.83% / 0.17% class distribution across all validation subsets to prevent data leakage.

\* \*\*Dimensionality Reduction:\*\* Leveraged pre-computed PCA components ($V\_1$ through $V\_{28}$) to maintain data privacy while capturing latent variance.



\### 2. Algorithmic Evaluation \& Tuning

Models were trained using algorithmic class penalization (`class\_weight='balanced'` and `scale\_pos\_weight`) to force gradient optimization toward the minority class. 



| Classifier Architecture | Precision | Recall (Sensitivity) | F1-Score | False Positives (Friction) |

| :--- | :---: | :---: | :---: | :---: |

| Baseline Logistic Regression | 0.06 | \*\*0.92\*\* | 0.11 | 1,389 |

| Tuned Logistic Regression | 0.16 | 0.89 | 0.27 | 460 |

| XGBoost Gradient Engine | 0.50 | 0.85 | 0.63 | 83 |

| 🏆 \*\*Random Forest (Ensemble)\*\*| \*\*0.96\*\* | 0.76 | \*\*0.85\*\* | \*\*3 (Near-Perfect)\*\* |



\### 3. The Business Impact (Production Decision)

While Logistic Regression successfully flagged the most total fraud (highest Recall), it generated \*\*1,389 false alarms\*\*, which would severely damage customer trust and increase support ticket overhead. 



The \*\*Random Forest Ensemble\*\* was chosen for production deployment because it achieved a massive \*\*0.85 F1-Score\*\*, catching the majority of fraud while dropping false alarms down to just \*\*3 instances\*\* out of \~57,000 baseline checks.



\---



\## 💻 Live Interactive Dashboard

The project features a sleek, dark-mode risk assessment portal built with \*\*Streamlit\*\*. 



\*\*Dashboard Features:\*\*

\* \*\*Real-Time Inference:\*\* Adjust transaction parameters via UI sliders to see live risk probability outputs.

\* \*\*Dynamic Color-Coding:\*\* Instant visual alerts for approved vs. intercepted transactions.

\* \*\*Analytics Engine:\*\* Integrated `matplotlib` and `seaborn` visualizations comparing model benchmarks on logarithmic scales.



\---



\## 🚀 Local Deployment Guide



Follow these steps to run the pipeline and UI on your local machine.



\### Prerequisites

\* Python 3.9+

\* Git



\### Installation

1\. \*\*Clone the repository:\*\*

&#x20;  ```bash

&#x20;  git clone \[https://github.com/YOUR\_USERNAME/Credit\_Card\_Fraud\_Detection.git](https://github.com/YOUR\_USERNAME/Credit\_Card\_Fraud\_Detection.git)

&#x20;  cd Credit\_Card\_Fraud\_Detection

