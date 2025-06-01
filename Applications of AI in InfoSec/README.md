# Applications of AI in InfoSec

This folder contains Jupyter notebooks demonstrating practical applications of artificial intelligence and machine learning techniques in information security contexts. These notebooks cover real-world cybersecurity challenges and show how AI can be leveraged to detect threats, classify malware, and analyze security data.

## Notebooks Overview

### 🎯 Skills Assessment
**File**: `Skills Assessment.ipynb`
**Topic**: Sentiment Analysis for Text Classification
- **Dataset**: IMDB Movie Reviews (25,000 samples)
- **Objective**: Build a binary classifier to predict positive/negative sentiment
- **Techniques**: 
  - Text preprocessing (tokenization, stemming, stop word removal)
  - TF-IDF vectorization
  - Logistic Regression with hyperparameter tuning
  - Model evaluation and deployment
- **Real-world Application**: Content moderation, social media monitoring, threat intelligence analysis

### 📧 Spam Detection
**Topic**: Email/SMS Spam Classification
- **Dataset**: SMS Spam Collection (5,574 messages)
- **Objective**: Classify messages as spam or legitimate (ham)
- **Techniques**:
  - Bag-of-words and N-gram features
  - Multinomial Naive Bayes classifier
  - Pipeline creation with CountVectorizer
  - Cross-validation and performance metrics
- **Security Relevance**: Email security, phishing detection, malicious content filtering

### 🌐 Network Anomaly Detection
**Topic**: Network Traffic Analysis and Intrusion Detection
- **Dataset**: Modified NSL-KDD dataset
- **Objective**: Multi-class classification of network attacks
- **Attack Categories**:
  - DoS (Denial of Service)
  - Probe (Network scanning)
  - Privilege Escalation
  - Unauthorized Access
- **Techniques**:
  - Random Forest classifier
  - Feature engineering for network data
  - Multi-class classification metrics
  - Confusion matrix analysis
- **Security Applications**: SIEM systems, network monitoring, threat detection

### 🦠 Malware Classification
**Topic**: Malware Family Classification using Computer Vision
- **Dataset**: Malimg dataset (9,339 malware images across 25 families)
- **Objective**: Classify malware samples by family using visual patterns
- **Techniques**:
  - Convolutional Neural Networks (CNNs)
  - Transfer learning with pre-trained ResNet50
  - Image preprocessing and data augmentation
  - PyTorch implementation
- **Innovation**: Converting binary files to grayscale images for visual analysis
- **Security Impact**: Automated malware analysis, threat intelligence, incident response

## Key Learning Objectives

### Technical Skills
- **Text Processing**: NLTK, regular expressions, feature extraction
- **Machine Learning**: Scikit-learn pipelines, model selection, evaluation
- **Deep Learning**: PyTorch, CNN architectures, transfer learning
- **Data Handling**: Pandas, NumPy, dataset preprocessing

### Security Applications
- **Threat Detection**: Automated identification of malicious content
- **Behavioral Analysis**: Pattern recognition in network traffic
- **Content Classification**: Distinguishing legitimate vs. malicious data
- **Scalable Solutions**: ML models for large-scale security monitoring

## Getting Started

### Prerequisites
```bash
# Core ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# NLP libraries
pip install nltk

# Deep learning (for malware classification)
pip install torch torchvision

# Utilities
pip install requests joblib split-folders
```

### Quick Start
1. **Begin with Spam Detection** - introduces fundamental text classification concepts
2. **Progress to Network Anomaly Detection** - covers structured data and multi-class problems
3. **Advance to Malware Classification** - explores deep learning and computer vision
4. **Complete Skills Assessment** - applies learned concepts to a comprehensive project

### Evaluation Framework
Each notebook includes integration with evaluation portals for automated model testing:
- **Spam Detection**: `localhost:8000/api/upload`
- **Network Anomaly**: `localhost:8001/api/upload` 
- **Malware Classification**: `localhost:8002/api/upload`
- **Skills Assessment**: `localhost:5000/api/upload`

## Real-World Applications

### Enterprise Security
- **Email Security Gateways**: Spam/phishing detection
- **Network Security Monitoring**: Anomaly detection in traffic patterns
- **Endpoint Protection**: Malware family identification and response
- **Content Filtering**: Automated classification of suspicious content

### Threat Intelligence
- **Malware Research**: Automated family classification for threat analysis
- **Social Engineering Detection**: Sentiment analysis for phishing attempts
- **Network Forensics**: Pattern recognition in attack signatures
- **Incident Response**: Rapid classification and prioritization of threats

## Performance Expectations

| Notebook | Expected Accuracy | Key Metrics |
|----------|------------------|-------------|
| Spam Detection | >95% | Precision, Recall, F1-Score |
| Network Anomaly | >85% | Multi-class accuracy, Confusion Matrix |
| Malware Classification | >85% | Image classification accuracy |
| Skills Assessment | >80% | Sentiment classification accuracy |

## Security Considerations

⚠️ **Important Notes**:
- Malware classification uses image representations for safety
- Network datasets are sanitized for educational use
- Models should be validated on diverse datasets before production deployment
- Consider adversarial attacks and model robustness in real-world applications