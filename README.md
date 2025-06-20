# EduScoreML

## Project Overview
EduScoreML is a machine learning project aimed at predicting student performance (math scores) based on factors such as gender, race/ethnicity, parental education, lunch type, and test preparation course. The project follows the complete ML lifecycle, including exploratory data analysis (EDA), data preprocessing, model training, and evaluation, with a web application for predictions.

This project demonstrates skills in:
- **Data Manipulation**: Using Pandas for EDA and preprocessing.
- **Machine Learning**: Training and evaluating regression models with Scikit-learn and XGBoost.
- **MLOps**: Structuring code with modular pipelines for scalability.
- **Web Development**: Deploying a Flask-based web app for predictions.

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977) and contains 1000 rows with 8 columns.

## Repository Structure
```
EduScoreML/
├── .gitignore                  # Ignored files
├── LICENSE                     # License file
├── README.md                   # Project documentation
├── app.py                      # Flask web app for predictions
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup script
├── logs/                       # Log files
│   ├── 06_17_2025_01_27_21.log
│   └── 06_17_2025_01_27_54.log
├── notebook/                   # Jupyter notebooks
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   ├── catboost_info/         # CatBoost training logs
│   └── data/
│       └── stud.csv           # Dataset
├── src/                       # Source code
│   ├── __init__.py
│   ├── exception.py           # Custom exceptions
│   ├── logger.py              # Logging utility
│   ├── utils.py               # Helper functions
│   ├── components/            # ML components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/              # ML pipelines
│       ├── __init__.py
│       ├── predict_pipeline.py
│       └── train_pipeline.py
└── templates/                 # HTML templates for Flask app
    ├── home.html
    └── index.html
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Monish-Nallagondalla/EduScoreML.git
   cd EduScoreML
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the dataset (`stud.csv`) is in the `notebook/data/` directory.

## Usage
1. **Exploratory Data Analysis**:
   - Open `notebook/1. EDA STUDENT PERFORMANCE.ipynb` to explore the dataset.
   - Key insights:
     - Females outperform males in overall scores, but males score higher in math.
     - Standard lunch correlates with better performance.
     - Students from Group E perform best; Group A performs worst.
     - Parental education (master’s/bachelor’s) positively impacts scores.
     - Test preparation courses improve scores across subjects.

2. **Model Training**:
   - Run `notebook/2. MODEL TRAINING.ipynb` to train regression models.
   - Models evaluated: Linear Regression, Lasso, Ridge, K-Neighbors, Decision Tree, Random Forest, XGBoost, AdaBoost.
   - Features are preprocessed using `ColumnTransformer` with `OneHotEncoder` for categorical variables and `StandardScaler` for numerical variables.
   - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.
   - Linear Regression achieved the highest accuracy (example: ~88% R² score on test data).

3. **Prediction Pipeline**:
   - Run the Flask app for predictions:
     ```bash
     python app.py
     ```
   - Access the web app at `http://localhost:5000` to input features and predict math scores.
   - The app uses `predict_pipeline.py` to load the trained model and preprocess inputs.

4. **Training Pipeline**:
   - Use `src/pipeline/train_pipeline.py` to automate data ingestion, transformation, and model training.
   - Logs are saved in the `logs/` directory.

## Key Features
- **Data Preprocessing**: Handles categorical (OneHotEncoder) and numerical (StandardScaler) features.
- **Model Evaluation**: Compares multiple regression models using MAE, RMSE, and R² metrics.
- **Modular Code**: Organized with separate components for data ingestion, transformation, and training.
- **Web Interface**: Flask app with HTML templates for user-friendly predictions.

## Insights from EDA
- **Gender**: Females have higher average scores; males excel in math.
- **Lunch**: Standard lunch students outperform free/reduced lunch students.
- **Race/Ethnicity**: Group E students score highest; Group A scores lowest.
- **Parental Education**: Higher education (master’s/bachelor’s) correlates with better student performance.
- **Test Preparation**: Completing the course improves scores in all subjects.
- **Score Distribution**: Math scores range from 0 to 100, with most students scoring 60–80.

## Model Performance
- Linear Regression performed best with an R² score of ~88% on the test set.
- Random Forest and XGBoost also showed strong performance but were slightly less accurate.
- Visualizations (scatter plots, regression plots) confirm predictions align closely with actual values.

## Requirements
Key dependencies (see `requirements.txt` for full list):
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `flask`
- `matplotlib`
- `seaborn`

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make changes and commit: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, contact [Monish Nallagondalla](mailto:nsmonish@gmail.com) or open an issue on GitHub.


