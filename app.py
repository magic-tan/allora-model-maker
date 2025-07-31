import streamlit as st # Streamlit 是用来做网页界面的库
import pandas as pd # pandas 是用来处理数据的库
import numpy as np # numpy 是用来做数字计算的库
import datetime # datetime 是用来处理日期的库
import pickle # pickle 通常用来加载用Python保存的模型文件
import os # os 模块用来处理文件和文件夹路径
import joblib # joblib 通常也用来加载用Python保存的模型文件

# --- 1. 加载你训练好的比特币预测模型 (arima 模型) ---
# @st.cache_resource 这个命令告诉 Streamlit，这个函数只运行一次，把模型加载到内存里，这样你的应用会运行得更快！
@st.cache_resource
def load_arima_model():
    model_name_folder = 'arima' # 我们选择的模型是 arima
    model_dir = os.path.join('trained_models', model_name_folder) # 模型的文件夹路径
    
    try:
        model_path = os.path.join(model_dir, "model.pkl") # arima 模型的 .pkl 文件
        scaler_path = os.path.join(model_dir, "scaler.pkl") # arima 模型的 .pkl 缩放器文件
        
        model = joblib.load(model_path) # 使用 joblib 加载模型
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None # 加载缩放器（如果有的话）
        
        return model, scaler
            
    except FileNotFoundError:
        st.error(f"错误：模型文件或缩放器文件在 '{model_dir}' 中没有找到！")
        st.error("请确认 'trained_models/arima' 文件夹及其中的 model.pkl 和 scaler.pkl 已上传。")
        return None, None
    except Exception as e:
        st.error(f"加载 arima 模型时出错：{e}")
        return None, None

# 调用函数来加载你的 arima 模型
my_arima_model, my_arima_scaler = load_arima_model()

# 如果模型没有成功加载，就停止程序，显示错误信息
if my_arima_model is None:
    st.stop()

# --- Streamlit 应用的网页界面标题和介绍 ---
st.title("比特币每日价格预测应用 (使用 ARIMA 模型) 📈")
st.write("欢迎来到我的比特币价格预测器！这里将展示 ARIMA 模型预测的未来价格。")
st.write("请记住：加密货币市场波动性大，任何预测都存在不确定性，不要把这里的预测当做投资建议哦！")

# --- 2. 模拟历史比特币数据 (你可以用真实数据替换它) ---
# 在实际应用中，你可能需要从 Tiingo API 或者文件加载真实的比特币历史数据
def generate_mock_data(start_date, days):
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    prices = np.cumsum(np.random.randn(days) * 100 + 500) + 10000
    prices = np.maximum(prices, 5000) # 确保价格不会太低
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# 生成足够的模拟历史数据，以满足 ARIMA 模型可能需要的输入天数
today = datetime.date.today()
historical_days = 400 # 假设需要足够多的历史数据，可以根据实际情况调整
historical_data = generate_mock_data(today - datetime.timedelta(days=historical_days), historical_days)

st.header("过去一年的比特币历史价格走势 (模拟数据)")
st.line_chart(historical_data['Price'])

# --- 3. 使用 ARIMA 模型进行预测 ---
def predict_future_prices_with_arima(model, scaler, historical_df, forecast_days=7):
    predictions = []
    
    # >>>>> !!! 紧急修改这里 !!! <<<<<
    # 你的 ARIMA 模型需要过去多少天的数据来预测未来？
    # 这个数字必须和你的 ARIMA 模型训练时使用的输入天数一致！
    # 如果你不确定，可能需要查看你训练 ARIMA 模型的代码。
    DAYS_REQUIRED_FOR_MODEL_INPUT = 30 # <--- !!! 请根据你训练的 ARIMA 模型的实际需求修改此数字 !!!

    if len(historical_df) < DAYS_REQUIRED_FOR_MODEL_INPUT:
        st.warning(f"历史数据不足 {DAYS_REQUIRED_FOR_MODEL_INPUT} 天，ARIMA 模型无法进行预测。请提供更多历史数据。")
        return pd.DataFrame()

    # 准备模型输入数据：通常是历史价格的最后 N 天
    # ARIMA 模型通常直接接受原始序列数据
    model_input_data = historical_df['Price'].tail(DAYS_REQUIRED_FOR_MODEL_INPUT).to_numpy()
    
    # 如果模型使用了 scaler，需要对输入数据进行缩放
    if scaler:
        # scikit-learn scaler 通常期望二维输入 (samples, features)
        model_input_data = scaler.transform(model_input_data.reshape(-1, 1)).flatten() 
        # 这里我们假设模型接受展平的一维数组

    # 调用你的 ARIMA 模型进行预测！
    # ARIMA 模型的预测方法通常是 forecast() 或 predict()
    # 并且可能需要 `steps` 参数来预测未来多少步
    try:
        # 这里假设你的 ARIMA 模型有一个 forecast 方法，接受步数参数，并返回预测结果数组
        # 如果你的模型接口不同，需要在这里调整
        forecast_output = model.forecast(steps=forecast_days)
        
        # 如果模型输出是 numpy 数组，直接使用
        if isinstance(forecast_output, np.ndarray):
            predictions = forecast_output.tolist()
        # 如果模型输出是 pandas Series/DataFrame，提取值
        elif isinstance(forecast_output, (pd.Series, pd.DataFrame)):
            predictions = forecast_output.iloc[:, 0].tolist() if isinstance(forecast_output, pd.DataFrame) else forecast_output.tolist()
        else:
            st.error("ARIMA 模型预测结果格式不支持。")
            return pd.DataFrame()

        # 如果模型使用了 scaler，需要对预测结果进行逆缩放
        if scaler:
            # 逆缩放通常也需要二维输入 (samples, features)
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
            
    except Exception as e:
        st.error(f"ARIMA 模型预测时出错：{e}")
        return pd.DataFrame()

    # 准备预测结果的数据框
    last_historical_date = historical_df.index[-1]
    forecast_dates = [last_historical_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)
    return forecast_df

# 让用户选择想要预测未来的天数，用滑块来选择
forecast_days_user_choice = st.slider("选择预测未来天数：", 1, 30, 7)

# 调用 ARIMA 预测函数
predicted_data = predict_future_prices_with_arima(
    my_arima_model, 
    my_arima_scaler, 
    historical_data, 
    forecast_days_user_choice
)

# 如果预测数据不是空的，就显示预测结果
if not predicted_data.empty:
    st.header(f"未来 {forecast_days_user_choice} 天比特币价格预测 (ARIMA 模型)")
    st.line_chart(predicted_data['Predicted Price'])

    # 将历史数据和预测数据合并，这样就可以在一个图表上同时看到历史和未来的走势
    combined_data = pd.concat([historical_data['Price'], predicted_data['Predicted Price']], axis=1)
    combined_data.columns = ['历史价格', '预测价格']

    st.header("历史价格与预测价格对比图")
    st.line_chart(combined_data)

st.markdown("---") # 在网页上显示一条分割线
st.info("学习机器学习和部署应用是很棒的经历！继续加油！")