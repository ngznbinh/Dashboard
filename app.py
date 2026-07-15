import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Cấu hình giao diện Streamlit chuyên nghiệp, trực quan hơn
st.set_page_config(
    page_title="Chăm Sóc Sức Khỏe Châu Đại Dương Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS để tùy chỉnh giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e3d59; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 ĐÁNH GIÁ CHI TIÊU Y TẾ TẠI CÁC QUỐC GIA CHÂU ĐẠI DƯƠNG")
st.caption("Ứng dụng phân tích kinh tế lượng tương tác - Dữ liệu giai đoạn 2000-2023")

@st.cache_data
def load_data():
    df = pd.read_excel('ket_qua_lam_sach_du_lieu.xlsx', sheet_name=0)
    
    # ĐỔI TÊN CỘT: Chuyển 'Urban rate (%)' thành 'Urban_rate' để tránh mọi lỗi ký tự đặc biệt
    df = df.rename(columns={'Urban rate (%)': 'Urban_rate'})
    
    # Đồng bộ hóa tên cột và căn giữa các biến độc lập
    df['uhc_index_centered'] = df['uhc_index'] - df['uhc_index'].mean()
    df['ln_gdp_pc_centered'] = df['ln_gdp_pc'] - df['ln_gdp_pc'].mean()
    df['interact_gdp_developed_c'] = df['developed'] * df['ln_gdp_pc_centered']
    df['uhc_index_squared_c'] = df['uhc_index_centered'] ** 2
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Lỗi khi đọc file dữ liệu: {e}. Vui lòng kiểm tra lại file 'ket_qua_lam_sach_du_lieu.xlsx'")
    st.stop()

# Thiết lập công thức hồi quy chuẩn hóa (đã thay bằng biến Urban_rate cực kỳ an toàn)
formula = 'ln_health_exp ~ uhc_index_centered + ln_gdp_pc_centered + ln_pop + Urban_rate + developed + interact_gdp_developed_c + uhc_index_squared_c'
model_ols = smf.ols(formula, data=df).fit()
model_robust = smf.ols(formula, data=df).fit(cov_type='HC3')

# Tạo thanh điều hướng (Sidebar) để người dùng chọn chức năng phân tích
st.sidebar.header("⚙️ BẢNG ĐIỀU KHIỂN")
analysis_mode = st.sidebar.radio(
    "Chọn nội dung phân tích:",
    ["📊 1. Kết quả hồi quy & Tác động", "🔍 2. Kiểm định khuyết tật mô hình", "🔮 3. Dự báo Chi tiêu Y tế", "📈 4. Xu hướng & Phân tích Panel"]
)

# ==========================================
# PHẦN 1: KẾT QUẢ HỒI QUY & TÁC ĐỘNG
# ==========================================
if analysis_mode == "📊 1. Kết quả hồi quy & Tác động":
    st.subheader("Tổng quan Mô hình Hồi quy OLS (Robust Standard Errors HC3)")
    
    # Chỉ số KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R-squared (R²)", f"{model_ols.rsquared:.4f}", help="Hệ số xác định độ phù hợp của mô hình")
    k2.metric("Adj. R-squared", f"{model_ols.rsquared_adj:.4f}")
    k3.metric("F-Statistic (p-value)", f"{model_ols.fvalue:.2f} ({model_ols.f_pvalue:.2e})")
    k4.metric("Số quan sát (N)", f"{int(model_ols.nobs)}")
    
    st.markdown("---")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.write("**Hệ số tác động chuẩn hóa (Standardized Beta) của từng biến:**")
        std_y = df['ln_health_exp'].std()
        coef_df = pd.DataFrame({
            'Biến': model_robust.params.index[1:],
            'Hệ số hồi quy (β)': model_robust.params.values[1:],
            'P-value': model_robust.pvalues.values[1:]
        })
        
        # Tính toán Standardized Beta
        std_x = {
            'uhc_index_centered': df['uhc_index_centered'].std(),
            'ln_gdp_pc_centered': df['ln_gdp_pc_centered'].std(),
            'ln_pop': df['ln_pop'].std(),
            'Urban_rate': df['Urban_rate'].std(),
            'developed': df['developed'].std(),
            'interact_gdp_developed_c': df['interact_gdp_developed_c'].std(),
            'uhc_index_squared_c': df['uhc_index_squared_c'].std()
        }
        coef_df['Beta chuẩn hóa'] = coef_df.apply(lambda r: r['Hệ số hồi quy (β)'] * (std_x.get(r['Biến'], 1) / std_y), axis=1)
        coef_df['Tác động'] = coef_df['Beta chuẩn hóa'].apply(lambda x: 'Tăng chi tiêu (+)' if x > 0 else 'Giảm chi tiêu (-)')
        coef_df = coef_df.sort_values(by='Beta chuẩn hóa', ascending=True)

        fig_coef = px.bar(
            coef_df, x='Beta chuẩn hóa', y='Biến', color='Tác động',
            color_discrete_map={'Tăng chi tiêu (+)': '#1f77b4', 'Giảm chi tiêu (-)': '#d62728'},
            orientation='h', title="Mức độ tác động tương đối của các nhân tố đến Chi tiêu Y tế"
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_r:
        st.write("**Bảng thông số chi tiết hệ số hồi quy:**")
        display_df = pd.DataFrame({
            'Hệ số (β)': model_robust.params,
            'Sai số chuẩn': model_robust.bse,
            'p-value': model_robust.pvalues
        })
        st.dataframe(display_df.style.format("{:.4f}"))

# ==========================================
# PHẦN 2: KIỂM ĐỊNH KHUYẾT TẬT MÔ HÌNH
# ==========================================
elif analysis_mode == "🔍 2. Kiểm định khuyết tật mô hình":
    st.subheader("Kiểm tra và phát hiện vi phạm giả thiết mô hình cổ điển (CLRM)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Kết quả các kiểm định giả thiết:**")
        _, bp_p, _, _ = het_breuschpagan(model_ols.resid, model_ols.model.exog)
        jb_p = sm.stats.stattools.jarque_bera(model_ols.resid)[1]
        dw_stat = sm.stats.stattools.durbin_watson(model_ols.resid)
        
        test_names = ["Phương sai sai số thay đổi (Breusch-Pagan)", "Kiểm định phân phối chuẩn (Jarque-Bera)", "Kiểm định tự tương quan (Durbin-Watson)"]
        stats_val = [f"p = {bp_p:.5f}", f"p = {jb_p:.5f}", f"DW = {dw_stat:.3f}"]
        
        bp_desc = "🔴 Có vi phạm (Dùng Robust SE)" if bp_p < 0.05 else "🟢 Đạt yêu cầu (Phương sai đồng đều)"
        jb_desc = "🟢 Đạt yêu cầu (Phân phối chuẩn)" if jb_p > 0.05 else "🔴 Không chuẩn (Mẫu lớn không đáng ngại)"
        dw_desc = "🔴 Có tự tương quan bậc 1" if (dw_stat < 1.5 or dw_stat > 2.5) else "🟢 Không tự tương quan"
        status = [bp_desc, jb_desc, dw_desc]
        
        violation_df = pd.DataFrame({"Kiểm định": test_names, "Giá trị": stats_val, "Kết luận": status})
        st.dataframe(violation_df, use_container_width=True)

    with col2:
        st.write("**Kiểm tra hiện tượng Đa cộng tuyến (VIF):**")
        vif_data = pd.DataFrame()
        X = model_ols.model.exog
        vif_data["Biến độc lập"] = model_ols.model.exog_names
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_data = vif_data[vif_data["Biến độc lập"] != "Intercept"]
        
        def color_vif(val):
            color = 'green' if val < 5 else ('orange' if val <= 10 else 'red')
            return f'color: {color}; font-weight: bold'
        st.dataframe(vif_data.style.map(color_vif, subset=['VIF']), use_container_width=True)

# ==========================================
# PHẦN 3: DỰ BÁO CHI TIÊU Y TẾ
# ==========================================
elif analysis_mode == "🔮 3. Dự báo Chi tiêu Y tế":
    st.subheader("Mô phỏng & Dự báo Chi tiêu Y tế tương tác")
    st.write("Thay đổi các giá trị đầu vào bên dưới để dự đoán Chi tiêu y tế thực tế trung bình đầu người:")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        uhc_val = st.slider("Chỉ số Bao phủ Y tế Toàn dân (UHC)", int(df['uhc_index'].min()), int(df['uhc_index'].max()), 70)
        gdp_val = st.slider("Log GDP đầu người (ln_gdp_pc)", float(df['ln_gdp_pc'].min()), float(df['ln_gdp_pc'].max()), 9.0)
        pop_val = st.slider("Log Dân số (ln_pop)", float(df['ln_pop'].min()), float(df['ln_pop'].max()), 13.0)
        urban_val = st.slider("Tỷ lệ Đô thị hóa (%)", float(df['Urban_rate'].min()), float(df['Urban_rate'].max()), 50.0)
        developed_val = st.selectbox("Quốc gia phát triển?", [0, 1], format_func=lambda x: "Có (Phát triển)" if x == 1 else "Không (Đang phát triển)")
        
        # Biến đổi biến để khớp với mô hình đã căn giữa
        uhc_c = uhc_val - df['uhc_index'].mean()
        gdp_c = gdp_val - df['ln_gdp_pc'].mean()
        interact_c = developed_val * gdp_c
        uhc_sq_c = uhc_c ** 2
        
        input_data = pd.DataFrame({
            'Intercept': [1.0], 'uhc_index_centered': [uhc_c], 'ln_gdp_pc_centered': [gdp_c],
            'ln_pop': [pop_val], 'Urban_rate': [urban_val], 'developed': [developed_val],
            'interact_gdp_developed_c': [interact_c], 'uhc_index_squared_c': [uhc_sq_c]
        })

    with col_result:
        prediction = model_ols.get_prediction(input_data)
        pred_summary = prediction.summary_frame(alpha=0.05)
        y_pred = pred_summary['mean'].values[0]
        y_lower = pred_summary['obs_ci_lower'].values[0]
        y_upper = pred_summary['obs_ci_upper'].values[0]
        
        st.info("💡 **Kết quả dự báo từ Mô hình Kinh tế lượng:**")
        st.metric("Chi tiêu y tế dự báo (USD/người)", f"${np.exp(y_pred):,.2f}")
        st.write(f"**Khoảng tin cậy 95%:** Từ **${np.exp(y_lower):,.2f}** đến **${np.exp(y_upper):,.2f}**")
        st.write(f"*Giá trị dự báo dưới dạng logarit (ln_health_exp):* **{y_pred:.4f}**")

# ==========================================
# PHẦN 4: XU HƯỚNG & PHÂN TÍCH PANEL
# ==========================================
else:
    st.subheader("Phân tích Xu hướng chuỗi thời gian & Phần dư theo Quốc gia")
    
    df['Y_pred_OLS'] = model_ols.fittedvalues
    df['Residuals'] = model_ols.resid
    time_trend = df.groupby('Year')[['ln_health_exp', 'Y_pred_OLS']].mean().reset_index()
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=time_trend['Year'], 
        y=time_trend['ln_health_exp'], 
        mode='lines+markers', 
        name='Thực tế (Trung bình)'
    ))
    fig_line.add_trace(go.Scatter(
        x=time_trend['Year'], 
        y=time_trend['Y_pred_OLS'], 
        mode='lines', 
        line=dict(dash='dash'), 
        name='Dự báo mô hình'
    ))
    fig_line.update_layout(
        title="Xu hướng Chi tiêu Y tế thực tế so với Giá trị dự báo", 
        xaxis_title="Năm", 
        yaxis_title="ln_health_exp"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.write("**Mức độ sai số (Phần dư) phân tán theo từng quốc gia:**")
    fig_box = px.box(
        df, 
        x='Entity', 
        y='Residuals', 
        color='Entity', 
        title="Biểu đồ phân phối phần dư (Phát hiện đặc trưng quốc gia)"
    )
    st.plotly_chart(fig_box, use_container_width=True)