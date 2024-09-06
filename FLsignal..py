import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoostFL.pkl')

# Define feature options

# Define feature names
feature_names = ['P1', 'P1.1', 'P2', 'P2.1', 'P3', 'P3.1', 'P4','P4.1', 'P5', 
                 'P5.1', 'P6', 'P6.1', 'P7', 'P7.1', 'P8', 'P8.1', 
                 'P9','P9.1', 'P10', 'P10.1', 'P11', 'P11.1', 'P12',
                 'P12.1', 'P13', 'P13.1','P14', 'P14.1']
# Streamlit user interface
st.title("Fluorescence Signal Semen quality test Predictor")

P1378 = st.number_input("P1-378:", min_value=0.0, max_value=1.0, value=0.01)
P1396 = st.number_input("P1-396:", min_value=0.0, max_value=1.0, value=0.2)

P2378 = st.number_input("P2-378:", min_value=0.0, max_value=1.0, value=0.01)
P2396 = st.number_input("P2-396:", min_value=0.0, max_value=1.0, value=0.2)

P3378 = st.number_input("P3-378:", min_value=0.0, max_value=1.0, value=0.01)
P3396 = st.number_input("P3-396:", min_value=0.0, max_value=1.0, value=0.2)

P4378 = st.number_input("P4-378:", min_value=0.0, max_value=1.0, value=0.01)
P4396 = st.number_input("P4-396:", min_value=0.0, max_value=1.0, value=0.2)

P5378 = st.number_input("P5-378:", min_value=0.0, max_value=1.0, value=0.01)
P5396 = st.number_input("P5-396:", min_value=0.0, max_value=1.0, value=0.2)

P6378 = st.number_input("P6-378:", min_value=0.0, max_value=1.0, value=0.01)
P6396 = st.number_input("P6-396:", min_value=0.0, max_value=1.0, value=0.2)

P7378 = st.number_input("P7-378:", min_value=0.0, max_value=1.0, value=0.01)
P7396 = st.number_input("P7-396:", min_value=0.0, max_value=1.0, value=0.2)

P8378 = st.number_input("P8-378:", min_value=0.0, max_value=1.0, value=0.01)
P8396 = st.number_input("P8-396:", min_value=0.0, max_value=1.0, value=0.2)

P9378 = st.number_input("P9-378:", min_value=0.0, max_value=1.0, value=0.01)
P9396 = st.number_input("P9-396:", min_value=0.0, max_value=1.0, value=0.2)

P10378 = st.number_input("P10-378:", min_value=0.0, max_value=1.0, value=0.01)
P10396 = st.number_input("P10-396:", min_value=0.0, max_value=1.0, value=0.2)

P11378 = st.number_input("P11-378:", min_value=0.0, max_value=1.0, value=0.01)
P11396 = st.number_input("P11-396:", min_value=0.0, max_value=1.0, value=0.2)

P12378 = st.number_input("P12-378:", min_value=0.0, max_value=1.0, value=0.01)
P12396 = st.number_input("P12-396:", min_value=0.0, max_value=1.0, value=0.2)

P13378 = st.number_input("P13-378:", min_value=0.0, max_value=1.0, value=0.01)
P13396 = st.number_input("P13-396:", min_value=0.0, max_value=1.0, value=0.2)

P14378 = st.number_input("P14-378:", min_value=0.0, max_value=1.0, value=0.01)
P14396 = st.number_input("P14-396:", min_value=0.0, max_value=1.0, value=0.2)



# Process inputs and make predictions
feature_values = [P1378,P1396,P2378,P2396,P3378,P3396,P4378,P4396,P5378,P5396,P6378,P6396,P7378,P7396,P8378,P8396,
P9378,P9396,P10378,P10396,P11378,P11396,P12378,P12396,P13378,P13396,P14378,P14396]
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
    # 创建一个 Matplotlib 图形  
    fig = shap.force_plot(explainer.expected_value, shap_values[:,:,predicted_class], pd.DataFrame([feature_values], columns=feature_names))  
    # 使用 Streamlit 显示 Matplotlib 图形  
    st.pyplot(fig)