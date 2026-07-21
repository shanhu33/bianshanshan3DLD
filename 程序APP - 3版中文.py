import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载保存的 Random Forest 模型
# 确保你的目录下有 'Random Forest.pkl' 文件
model = joblib.load('Random Forest.pkl')

# 特征范围定义（根据您提供的信息）
feature_ranges = {
    "母亲学历": {
        "type": "categorical",
        "options": [0, 1, 2, 3, 4, 5, 6],
        "default": 2,
        "label_map": {
            0: "文盲",
            1: "小学",
            2: "初中",
            3: "高中/中专",
            4: "大专",
            5: "本科",
            6: "研究生及以上"
        },
        "description": "0:文盲, 1:小学, 2:初中, 3:高中/中专, 4:大专, 5:本科, 6:研究生及以上"
    },
    "妊娠末期胎心": {
        "type": "numerical",
        "min": 100,
        "max": 180,
        "default": 140,
        "unit": "次/分",
        "description": "正常范围110-160次/分"
    },
    "胎儿性别": {
        "type": "categorical",
        "options": [1, 2],
        "default": 1,
        "label_map": {1: "女", 2: "男"},
        "description": "1:女, 2:男"
    },
    "妊娠孕次": {
        "type": "numerical",
        "min": 1,
        "max": 20,
        "default": 2,
        "unit": "次",
        "description": "含本次妊娠的怀孕总次数，初产妇为1"
    }
}

# Streamlit 界面
st.set_page_config(
    page_title="妊娠结局风险预测",
    page_icon="🤰",
    layout="wide"
)

# 使用更紧凑的布局
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stButton > button {
        padding: 0.3rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题 - 更紧凑
st.title("🤰 妊娠结局风险预测")

# 特征输入 - 更紧凑
st.markdown("#### 📝 请输入以下特征值：")

# 使用更紧凑的列布局
col1, col2 = st.columns(2, gap="small")

feature_values = []
feature_names = list(feature_ranges.keys())

for i, (feature, properties) in enumerate(feature_ranges.items()):
    # 选择列
    current_col = col1 if i % 2 == 0 else col2
    
    with current_col:
        if properties["type"] == "numerical":
            # 显示带单位的数值输入
            label = f"{feature} ({properties['min']} - {properties['max']} {properties.get('unit', '')})"
            value = st.number_input(
                label=label,
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=properties.get("description", ""),
                key=feature,
                label_visibility="collapsed" if i > 0 else "visible"
            )
        elif properties["type"] == "categorical":
            # 显示带标签的分类选择
            options = properties["options"]
            label_map = properties.get("label_map", {})
            # 创建显示标签列表
            display_options = [f"{opt} ({label_map.get(opt, opt)})" for opt in options]
            selected_display = st.selectbox(
                label=f"{feature}",
                options=display_options,
                help=properties.get("description", ""),
                key=feature,
                label_visibility="collapsed" if i > 0 else "visible"
            )
            # 提取实际数值
            value = options[display_options.index(selected_display)]
        feature_values.append(value)

# 预测按钮 - 更紧凑
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 预测风险", use_container_width=True)

# 预测与结果显示
if predict_button:
    # 转换为模型输入格式
    features = np.array([feature_values])
    
    # 模型预测
    predicted_proba = model.predict_proba(features)[0]
    
    # 获取阳性概率（风险概率）
    risk_probability = predicted_proba[1] * 100

    # 显示预测结果 - 更紧凑的卡片
    st.markdown("---")
    
    # 根据风险概率确定颜色
    if risk_probability < 30:
        color = '#51cf66'
        bg_color = '#f0fff4'
        risk_level = "低风险"
        icon = "🟢"
    elif risk_probability < 60:
        color = '#ffd43b'
        bg_color = '#fffbf0'
        risk_level = "中等风险"
        icon = "🟡"
    else:
        color = '#ff6b6b'
        bg_color = '#fff5f5'
        risk_level = "高风险"
        icon = "🔴"
    
    # 显示主要结果 - 更紧凑的卡片
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 15px 20px;
            border-radius: 15px;
            border: 3px solid {color};
            text-align: center;
            margin: 5px 0;
        ">
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                <div style="font-size: 48px;">{icon}</div>
                <div>
                    <div style="font-size: 32px; font-weight: bold; color: {color};">
                        {risk_probability:.1f}%
                    </div>
                    <div style="font-size: 18px; color: {color};">
                        {risk_level}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 添加一个简短的说明
    st.caption("⚠️ 本预测仅供参考，不能替代专业医疗诊断")

# 侧边栏 - 使用说明（更紧凑）
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. 输入各项特征值
    2. 点击"预测风险"按钮
    3. 查看风险概率
    
    ---
    **特征说明：**
    - **母亲学历**：0-文盲, 1-小学, 2-初中, 3-高中/中专, 4-大专, 5-本科, 6-研究生及以上
    - **妊娠末期胎心**：正常范围110-160次/分
    - **胎儿性别**：1-女, 2-男
    - **妊娠孕次**：含本次妊娠的怀孕总次数，初产妇为1
    
    ---
    ⚠️ 本预测仅供参考，不能替代专业医疗诊断
    """)
    
    with st.expander("ℹ️ 模型信息"):
        st.write("**模型类型：** Random Forest")
        st.write("**特征数量：** 4个")