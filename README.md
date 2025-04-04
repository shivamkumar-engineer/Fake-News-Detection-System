# Fake-News-Detection-System

A machine learning application that uses various classification algorithms to detect fake news articles based on their title and content.

## Introduction

The Fake-News-Detection-System is a Flask web application that leverages multiple machine learning algorithms to analyze news articles and classify them as either real or fake. The system preprocesses text data, extracts features using TF-IDF vectorization, and employs various classification models to achieve high accuracy in fake news detection.

## Features

- **Text Preprocessing**: Cleans and normalizes news text by removing URLs, mentions, hashtags, punctuation, and numbers
- **Feature Extraction**: Combines title and article text, then transforms using TF-IDF vectorization
- **Multiple Classification Models**: Implements and compares 7 different machine learning algorithms
- **Cross-Validation**: Ensures model robustness through 5-fold cross-validation
- **Performance Metrics**: Provides detailed model performance reports including accuracy, confusion matrix, and classification reports
- **Web Interface**: User-friendly Flask application for real-time news analysis
- **Prediction History**: Stores recent predictions for user reference
- **Dataset Analysis**: Displays statistics about the training dataset

## Machine Learning Models

The application trains and evaluates the following classification models:

1. **Logistic Regression**: A linear model that uses a logistic function to estimate probabilities, configured with SAG solver for efficiency with large datasets.

2. **Multinomial Naive Bayes**: A probabilistic classifier that applies Bayes' theorem with strong independence assumptions, particularly effective for text classification.

3. **SGD Classifier**: Stochastic Gradient Descent classifier with logistic loss function, efficient for large-scale and online learning problems.

4. **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions, effective at handling non-linear relationships.

5. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on the majority class of the k nearest training samples.

6. **Support Vector Machine (SVM)**: Uses a linear kernel to find the hyperplane that best separates fake and real news articles in the feature space.

7. **Passive Aggressive Classifier**: An online learning algorithm that remains passive for correct classifications and aggressive for incorrect ones, well-suited for text classification.

The system automatically selects the best-performing model based on accuracy metrics.

## TF-IDF Vectorizer

The application uses Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to convert text data into numerical features:

- **Feature Extraction**: Converts text into a matrix of TF-IDF features
- **Stop Words**: Removes common English words that don't carry significant meaning
- **N-grams**: Includes both single words and pairs of adjacent words (1-2 n-grams)
- **Feature Limitation**: Restricts to the top 5,000 features to balance performance and resource usage

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- Flask
- joblib
- re (regular expressions)

## Installation

### Setting Up a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Installing Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

## Usage

### Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/shivamkumar-engineer/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

### Running the Application

```bash
# Run the Flask application
python model.py
```

Navigate to `http://127.0.0.1:5000/` in your web browser to access the application.

### Data Requirements

The application expects a CSV file named `news.csv` with at least the following columns:
- `title`: The headline of the news article
- `text`: The body content of the news article
- `label`: The classification label ('FAKE' or 'REAL')

## Project Structure

```
Fake-News-Detection-System/
├── app.py                  # Main Flask application
├── models/                 # Directory for saved models
│   ├── model.pkl           # Best performing trained model
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   └── reports.pkl         # Performance reports and dataset statistics
├── templates/              # Flask HTML templates
│   └── index.html          # Main web interface
├── static/                 # CSS, JavaScript, and static files
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Contributing

Contributions to improve the Fake-News-Detection-System are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Run tests to ensure functionality
5. Commit your changes (`git commit -m 'Add new feature'`)
6. Push to the branch (`git push origin feature-branch`)
7. Create a Pull Request
