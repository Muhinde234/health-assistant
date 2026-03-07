# ❤️ AI Heart Disease Risk Assessment System

![Version](https://img.shields.io/badge/version-1.0_Pro_Max-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.55.0-red)

A professional, AI-powered cardiac risk assessment system designed to assist healthcare professionals in evaluating heart disease probability using machine learning.

## 🌟 Features

### 🔍 **Advanced Risk Assessment**
- Real-time heart disease risk prediction using ML models
- 13 comprehensive clinical parameters
- Interactive risk gauge visualization
- Detailed risk factor breakdown
- Personalized recommendations

### 📊 **Analytics Dashboard**
- Patient assessment statistics
- Risk distribution charts
- Risk score trend analysis
- Feature importance visualization
- Historical data tracking

### 📋 **Patient History Management**
- Complete assessment history
- Patient identification system
- Timestamp tracking
- Quick access to previous assessments
- Export capability for reports

### 📄 **Professional Reporting**
- Downloadable assessment reports
- Detailed clinical parameter documentation
- Risk factor analysis
- Professional recommendations
- Timestamp and patient information

### 🎨 **Modern UI/UX**
- Professional color scheme
- Responsive design
- Interactive visualizations using Plotly
- Tab-based navigation
- Sidebar dashboard
- Tooltips and help text for all inputs

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/health-AI-assist.git
cd health-AI-assist
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (if not already trained)
```bash
python src/train_model.py
```

5. **Run the application**
```bash
streamlit run src/dashboard/hospital_dashboard.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
health-AI-assist/
├── models/
│   └── heart_model.pkl          # Trained ML model
├── src/
│   ├── dashboard/
│   │   └── hospital_dashboard.py # Main Streamlit app
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing
│   ├── train_model.py          # Model training script
│   ├── evaluate_model.py       # Model evaluation
│   ├── predict.py              # Prediction utilities
│   └── config.py               # Configuration settings
├── notebook/
│   └── work.ipynb              # Jupyter notebook for analysis
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                   # This file
```

## 🔬 Technical Details

### Machine Learning Models
- **Primary**: Logistic Regression
- **Secondary**: Random Forest Classifier
- **Training Data**: Cleveland Heart Disease Database (UCI ML Repository)
- **Features**: 13 clinical parameters
- **Accuracy**: ~85% on test data

### Input Parameters

| Parameter | Description | Range/Options |
|-----------|-------------|---------------|
| Age | Patient age in years | 20-100 |
| Sex | Biological sex | Male/Female |
| Chest Pain Type | Type of chest pain | Asymptomatic, Atypical Angina, Non-Anginal, Typical Angina |
| Resting BP | Resting blood pressure | 80-200 mm Hg |
| Cholesterol | Serum cholesterol | 100-600 mg/dl |
| Fasting Blood Sugar | >120 mg/dl | Yes/No |
| Resting ECG | ECG results | Normal, ST-T Abnormality, LV Hypertrophy |
| Max Heart Rate | Maximum heart rate achieved | 60-220 bpm |
| Exercise Angina | Exercise-induced angina | Yes/No |
| ST Depression | Oldpeak value | 0-6 mm |
| ST Slope | ST segment slope | Upsloping, Flat, Downsloping |
| Major Vessels | Colored by fluoroscopy | 0-4 |
| Thalassemia | Blood disorder status | Normal, Fixed Defect, Reversible Defect |

## 🎯 Use Cases

✅ **Clinical Decision Support** - Assist doctors in preliminary risk assessment  
✅ **Patient Screening** - Identify high-risk patients for further testing  
✅ **Health Monitoring** - Track patient cardiac health over time  
✅ **Medical Education** - Demonstrate ML applications in healthcare  
✅ **Research** - Analyze patterns in cardiac risk factors  

## 📊 Screenshot Features

- **Risk Assessment Tab**: Interactive input forms with real-time validation
- **Analytics Dashboard**: Comprehensive charts and statistics
- **Patient History**: Detailed assessment records
- **Information Tab**: Complete documentation and medical disclaimer

## 🔒 Security & Privacy

⚠️ **Important Notes**:
- This is a decision support tool, not a diagnostic system
- All patient data is stored locally in session state
- No data is transmitted to external servers
- Clear session history option available
- Suitable for HIPAA-compliant environments when properly deployed

## ⚠️ Medical Disclaimer

This system is designed as a **decision support tool** for healthcare professionals and should NOT be used as the sole basis for medical diagnosis or treatment decisions.

- ✅ Use as part of comprehensive clinical evaluation
- ✅ Combine with other diagnostic tests
- ✅ Review with qualified medical professionals
- ❌ Do not use for self-diagnosis
- ❌ Do not replace professional medical advice

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path: `src/dashboard/hospital_dashboard.py`
5. Deploy!

### Docker (Coming Soon)

```dockerfile
# Dockerfile configuration coming soon
```

## 🛠️ Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Running Tests

```bash
# Coming soon: pytest integration
python -m pytest tests/
```

## 📈 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Multi-language support
- [ ] Mobile-responsive design improvements
- [ ] Integration with EHR systems
- [ ] Real-time monitoring dashboard
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Advanced data visualization
- [ ] PDF report generation with charts
- [ ] User authentication system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed for educational and research purposes.

## 👨‍💻 Author

Created with ❤️ for healthcare innovation

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Cleveland Clinic Foundation
- Streamlit community
- Scikit-learn developers

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Version**: 1.0 Pro Max | **Last Updated**: March 2026 | **Status**: Production Ready
