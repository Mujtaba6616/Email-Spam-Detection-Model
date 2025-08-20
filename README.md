# 📧 Email Spam Detection Model

A **machine learning-based email spam detection system** that classifies emails as **spam** or **ham** (legitimate) using advanced text processing and classification algorithms. The model effectively identifies and filters unwanted emails to enhance email security and user experience.

## 🎯 Project Overview

* **Objective:** Build an intelligent system to automatically detect and classify spam emails
* **Dataset:** Email corpus with labeled spam/ham messages
* **Approach:** Text preprocessing, feature extraction, and machine learning classification
* **Output:** Trained model with interactive interface and comprehensive evaluation metrics
* **Accuracy:** Achieved ~95%+ accuracy on test dataset

## ⚙️ Installation & Requirements

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mujtaba6616/Email-Spam-Detection-Model.git
cd Email-Spam-Detection-Model
pip install -r requirements.txt
```

**Requirements:**
* Python 3.7+
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* nltk
* joblib
* IPython

## 📂 Project Structure

```
Spam_Email_Detection.ipynb      # Main Jupyter notebook with complete workflow
email_spam_detector.pkl         # Trained model file
requirements.txt                # Python dependencies
README.md                       # Project documentation
```

## 🚀 Workflow

### 1. **Data Preprocessing**
   * Text cleaning and normalization
   * Removal of special characters, URLs, and HTML tags
   * Converting to lowercase
   * Tokenization and stemming/lemmatization

### 2. **Feature Engineering**
   * **TF-IDF Vectorization:** Convert text to numerical features
   * **N-gram Analysis:** Unigrams and bigrams for better context
   * **Feature Selection:** Identify most informative features

### 3. **Model Training & Evaluation**
   * **Algorithms Tested:** Naive Bayes, Logistic Regression, Random Forest, SVM
   * **Cross-validation:** K-fold validation for robust performance
   * **Hyperparameter Tuning:** Grid search for optimal parameters
   * **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

### 4. **Model Deployment**
   * Interactive web interface for real-time email classification
   * Confidence scoring for predictions
   * User-friendly design with modern CSS styling

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.2% |
| **Precision** | 94.8% |
| **Recall** | 95.6% |
| **F1-Score** | 95.2% |

## 🖥️ Interactive Interface

The project includes a **modern web-based interface** featuring:

* **Real-time Analysis:** Instant spam/ham classification
* **Confidence Scoring:** Probability percentages for predictions
* **Responsive Design:** Clean, professional UI with dark theme
* **Easy to Use:** Simple text input with one-click analysis

### Usage Example:

```python
# Load the trained model
import joblib
detector = joblib.load('email_spam_detector.pkl')

# Analyze an email
sample_email = "Congratulations! You've won $1000000! Click here now!"
prediction = detector.predict([sample_email])
confidence = detector.predict_proba([sample_email])

print(f"Prediction: {'SPAM' if prediction[0] == 1 else 'HAM'}")
print(f"Confidence: {max(confidence[0]) * 100:.1f}%")
```

## 📈 Key Features

* **High Accuracy:** 95%+ spam detection rate
* **Fast Processing:** Real-time email classification
* **Robust Preprocessing:** Handles various email formats
* **Feature Engineering:** Advanced text vectorization techniques
* **Model Comparison:** Multiple algorithms tested and evaluated
* **Interactive UI:** User-friendly web interface
* **Confidence Scoring:** Probability-based predictions

## 🔧 Technical Implementation

### Text Preprocessing Pipeline:
1. HTML tag removal
2. URL and email address normalization
3. Special character cleaning
4. Tokenization and stop word removal
5. Stemming/Lemmatization

### Machine Learning Pipeline:
1. TF-IDF vectorization with n-grams
2. Feature selection and dimensionality reduction
3. Model training with cross-validation
4. Hyperparameter optimization
5. Performance evaluation and model selection

## 📱 How to Use

1. **Open the Notebook:** Run `Spam_Email_Detection.ipynb` in Jupyter
2. **Train the Model:** Execute all cells to train and save the model
3. **Use Interactive Interface:** Use the built-in web interface for real-time detection
4. **Custom Analysis:** Load the saved model and use `analyze_email()` function

## 🔮 Future Enhancements

* **Deep Learning:** Implement LSTM/BERT models for better accuracy
* **Real-time Integration:** Email client plugin development
* **Multilingual Support:** Extend to non-English emails
* **Advanced Features:** Sender reputation, attachment analysis
* **Mobile App:** Cross-platform mobile application
* **API Development:** RESTful API for integration

## 📊 Dataset Information

* **Source:** Publicly available email spam datasets
* **Size:** Thousands of labeled email samples
* **Classes:** Binary classification (Spam vs Ham)
* **Format:** Raw email text with metadata

## 🧑‍💻 Author

**Mujtaba Ahmed**  
Senior Computer Science Student — FAST-NUCES Lahore

[![GitHub](https://img.shields.io/badge/GitHub-Mujtaba6616-blue?style=flat&logo=github)](https://github.com/Mujtaba6616)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/mujtaba-ahmed)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any improvements or suggestions.

## ⭐ Acknowledgments

* Scikit-learn team for excellent machine learning tools
* Open-source email datasets contributors
* FAST-NUCES for academic support and resources
