import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options

# Define feature names
feature_names = ['C','N%','SM','PR','IM']

# Streamlit user interface
st.title(" Semen quality test Predictor")

# 精液浓度: numerical input
C = st.number_input("C:", min_value=1, max_value=300, value=50)

# N: categorical selection
N = st.number_input("N%:", min_value=1, max_value=10, value=3)

# SM: categorical selection
SM =st.number_input("SM:", min_value=10, max_value=100, value=50)

# PR: numerical input
PR = st.number_input("PR:", min_value=1, max_value=150, value=30)

# IM: numerical input
IM = st.number_input("IM:", min_value=0, max_value=100, value=50)


# Process inputs and make predictions
feature_values = [C, N, SM, PR, IM]
features = np.array([feature_values])

if st.button("Predict"):  
    # Predict class and probabilities  
    predicted_class = model.predict(features)[0]  
    predicted_proba = model.predict_proba(features)[0]  
  
    # Display prediction results  
    class_labels = ['Astheno', 'AsthenoTerato', 'Normal', 'OAT', 'Terato'] # 假设的类别标签  
    predicted_class_label = class_labels[predicted_class]  
    st.write(f"**Predicted Class**: {predicted_class_label}")  
    st.write(f"**Prediction Probabilities**: {predicted_proba}")  
  
    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100  
  
    # 根据不同的类别生成不同的建议  
    advice = ""  
    if predicted_class == 0:  
        advice = (  
            f"根据我们的模型，您属于{predicted_class_label}。模型预测您属于此类的概率为{probability:.1f}%。"  
            "具体建议需要根据该类别的含义来确定。"  
        )  
    elif predicted_class == 1:  
        advice = (  
            f"根据我们的模型，您可能面临较高的风险（属于{predicted_class_label}）。"  
            f"模型预测您属于此类的概率为{probability:.1f}%。"  
            "建议您立即咨询相关专家以获取进一步评估和治疗建议。"  
        )  
    elif predicted_class == 2:  
        advice = (  
            f"根据我们的模型，您属于{predicted_class_label}的概率较高，为{probability:.1f}%。"  
            "请继续关注您的健康状况，并根据需要寻求医疗建议。"  
        )  
    elif predicted_class == 3:  
        advice = (  
            f"模型预测您属于{predicted_class_label}的概率为{probability:.1f}%。"  
            "请注意保持健康的生活方式，并根据需要咨询医生。"  
        )  
    elif predicted_class == 4:  
        advice = (  
            f"模型预测您属于{predicted_class_label}的概率为{probability:.1f}%。"  
            "具体的行动建议需要根据该类别的具体情况来确定。"  
        )  
    else:  
        advice = "无法识别的类别，请检查模型输出。"  
  
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value[predicted_class], shap_values[:,:,predicted_class], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
