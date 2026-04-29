🚀 OPS-SAT Satellite Anomaly Detection – Predictive Analytics Project
=====================================================================

This repository contains a **machine learning anomaly detection project** using the **ESA OPS-SAT satellite telemetry dataset**, sourced from Zenodo. The aim of this project is to apply predictive analytics techniques to classify spacecraft signal behaviour into anomalous vs normal categories and identify the most accurate machine learning model.

📌 Project Objective
--------------------

To design a predictive system that:

*   Preprocesses satellite telemetry data
    
*   Performs Exploratory Data Analysis
    
*   Trains multiple ML models
    
*   Classifies anomaly vs normal signal behaviour
    
*   Compares model accuracy
    
*   Identifies the best performing algorithm
    

📂 Dataset Source
-----------------

*   **Dataset Name:** OPS-SAT Satellite Telemetry Anomaly Dataset
    
*   **Provider:** European Space Agency
    
*   **Source Link:** [https://zenodo.org/records/15108715](https://zenodo.org/records/15108715)
    
*   **Dataset Type:** Binary Classification (anomaly = 0/1)
    
*   **Attributes Count:** 23 features
    

📊 Features Used in Training
----------------------------

Some key dataset attributes:

*   mean
    
*   kurtosis
    
*   skew
    
*   duration
    
*   len
    
*   variance
    
*   n\_peaks
    
*   diff2\_peaks
    

Target variable:

*   anomaly
    

🧠 Techniques Applied
---------------------

This project includes models and techniques from multiple predictive analytics domains:

### ✔ Dataset Preprocessing

*   Missing value check
    
*   Feature scaling
    
*   Label encoding
    
*   Train-test split
    

### ✔ Exploratory Data Analysis

*   Correlation heatmap
    
*   Class distribution
    
*   PCA clustering
    

### ✔ Machine Learning Models

Trained models include:

ModelStatusLogistic Regression✔ TestedRandom Forest✔ TestedSVM✔ TestedNaive Bayes✔ TestedDecision Tree✔ TestedKNN✔ Tested

🏆 Best Model Result
--------------------

**Random Forest Classifier produced the highest accuracy**, outperforming:

*   Logistic Regression
    
*   SVM
    
*   Naive Bayes
    
*   KNN
    
*   Decision Tree
    

The model was selected based on validation accuracy performance.

🔧 Requirements
---------------

Use the following Python libraries:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pandas  numpy  sklearn  seaborn  matplotlib   `

▶️ Running the Notebook
-----------------------

1.  Download repository
    
2.  Open notebook in Jupyter/Colab
    
3.  Upload ops\_sat.csv
    
4.  Run cells sequentially
    

📸 Visual Outputs Included
--------------------------

*   PCA scatter plot
    
*   Class distribution graph
    
*   Correlation heatmap
    
*   Model performance comparison
    

All visualisations are documented clearly inside the notebook.

📈 Accuracy Comparison Summary (Example)
----------------------------------------

ModelAccuracy ScoreRandom Forest0.97SVM0.95Decision Tree0.94Logistic Regression0.92KNN0.90Naive Bayes0.85

_(Actual results may vary depending on random state & preprocessing.)_

🔍 Why This Project Matters
---------------------------

Satellite failure detection is critical to:

*   prevent mission losses
    
*   identify communication disruptions
    
*   enhance spacecraft performance
    
*   automate signal analysis
    

This project demonstrates how **machine learning can detect anomalies from space signal data**.

🚀 Future Improvements
----------------------

*   Apply Neural Networks (MLP, CNN)
    
*   Advanced hyperparameter tuning
    
*   Deploy real-time anomaly alerts
    
*   Implement SMOTE to fix imbalance
    
*   Expand dataset using raw waveform data
    

📝 Author
---------

*   **Your Name**
    
*   Predictive Analytics Student
    
*   Year: 2024–2025
    

📜 Citation
-----------

If you use this dataset, cite ESA Zenodo OPS-SAT release:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   OPS-SAT Anomaly Dataset  Zenodo Repository, 2024  https://zenodo.org/records/15108715   `

⭐ If this Research Helps You
----------------------------

Please ⭐ star the repository — it motivates further development! 😊
