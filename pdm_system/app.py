from flask import Flask, render_template_string, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("./model/rf_best_model.pkl")

# ✅关键：直接从模型取出训练时真实使用的特征列表，不再读csv获取！
FEATURE_COLS = model.feature_names_in_.tolist()
print("【模型训练时真实特征】", FEATURE_COLS)


HTML_TPL = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>设备预测性维护智能预警系统</title>
<style>
body{font-family:"Microsoft Yahei";max-width:720px;margin:40px auto;padding:0 20px;}
.box{border:1px solid #ccc;padding:24px;border-radius:10px;margin-top:24px;}
.warn{color:#dc3545;font-weight:bold;font-size:19px;}
.safe{color:#198754;font-weight:bold;font-size:19px;}
input,select{padding:6px;width:260px;}
p{margin:12px 0;}
button{padding:8px 22px;cursor:pointer;}
</style>
</head>
<body>
<h2>设备预测性维护智能分析与预警系统</h2>
<div class="box">
<form method="post">
<p>空气温度 Air temperature [K]: <input type="number" step="0.1" name="air_temp" required></p>
<p>工艺温度 Process temperature [K]: <input type="number" step="0.1" name="proc_temp" required></p>
<p>转速 Rotational speed [rpm]: <input type="number" name="rot_speed" required></p>
<p>扭矩 Torque [Nm]: <input type="number" step="0.1" name="torque" required></p>
<p>工具磨损 Tool wear [min]: <input type="number" name="tool_wear" required></p>
<p>产品类型 Type：
<select name="prod_type">
<option value="L">L</option>
<option value="M">M</option>
<option value="H">H</option>
</select>
</p>
<button type="submit">提交预测预警</button>
</form>
</div>
{% if res is not none %}
<div class="box">
<h3>预测结果</h3>
{% if res==1 %}
<p class="warn">⚠️警告：设备存在故障风险！建议及时维护检修</p>
{% else %}
<p class="safe">✅设备状态正常，无故障风险</p>
{% endif %}
</div>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    res = None
    if request.method == "POST":
        air_temp = float(request.form["air_temp"])
        proc_temp = float(request.form["proc_temp"])
        rot_speed = float(request.form["rot_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])
        prod_type = request.form["prod_type"]

        # 和preprocess保持一致计算衍生特征
        power_load = torque * rot_speed * 2 * np.pi / 60
        t_diff = proc_temp - air_temp
        tool_wear_norm = tool_wear / 253.0

        type_h = 1 if prod_type == "H" else 0
        type_l = 1 if prod_type == "L" else 0
        type_m = 1 if prod_type == "M" else 0

        input_dict = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "T_diff[K]": t_diff,
            "Power_load": power_load,
            "Tool_wear_norm": tool_wear_norm,
            "Type_H": type_h,
            "Type_L": type_l,
            "Type_M": type_m
        }

        df_input = pd.DataFrame([input_dict])[FEATURE_COLS]
        pred = model.predict(df_input)[0]
        res = int(pred)
    return render_template_string(HTML_TPL, res=res)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
