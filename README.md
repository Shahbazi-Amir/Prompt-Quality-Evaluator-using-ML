[START_OF_REPORT]
# Prompt Quality Evaluator with ML

## 🎯 Project Overview
A machine learning pipeline to evaluate the quality of LLM responses based on prompt engineering.

## 📊 Dataset Summary
- **Total Prompts:** 104
- **Responses per Prompt:** 3 (Bad, Average, Good)
- **Total Samples:** 312
- **Labels:** 0 = Bad, 1 = Average, 2 = Good
- **Features Extracted:**
  - `length`: Length of the response
  - `prompt_response_sim`: Cosine similarity between prompt and response
  - `repeated_words`: Number of repeated words in response

## 🧪 Models Tested
| Model              | Accuracy | F1-Score (Macro) | Type           |
|--------------------|----------|------------------|----------------|
| Random Forest      | **90%**  | **0.90**         | Classification |
| SVM (RBF)          | 89%      | 0.88             | Classification |
| Linear Regression  | R² = 0.66| —                | Regression     |

## 📈 Key Insights
- **Random Forest** showed the best performance.
- `prompt_response_sim` was the most important feature.
- Linear Regression was less suitable for classification without discretization.

## 📁 Files
- `prompt_data_updated.csv`: Labeled dataset with extracted features

## 🚀 Next Steps (Optional)
- Add more features (e.g., BERT embeddings, grammar score)
- Use deep learning models for better accuracy
- Build a UI with Streamlit for real-time evaluation
[END_OF_REPORT]