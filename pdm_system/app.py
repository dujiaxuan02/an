import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)

model = joblib.load("./model/rf_best_model.pkl")
feature_cols = model.feature_names_in_.tolist()
print("模型特征列表：", feature_cols)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>设备预测性维护智能分析与预警系统</title>
<style>
body {
    font-family: Microsoft Yahei, sans-serif;
    max-width:850px;
    margin:40px auto;
    padding:0 20px;
}
.box{
    border:1px solid #ccc;
    border-radius:8px;
    padding:25px;
    margin-bottom:25px;
}
.item{
    margin-bottom:16px;
}
label{
    display:inline-block;
    width:240px;
}
input,select{
    padding:6px;
    width:280px;
}
button{
    padding:8px 22px;
    cursor:pointer;
}
.warn{
    color:#d62728;
    font-weight:bold;
    font-size:17px;
}
.ok{
    color:#288b36;
    font-weight:bold;
    font-size:17px;
}
.debug-info{
    margin-top:12px;
    color:#666;
}
</style>
</head>
<body>
<h2 align="center">设备预测性维护智能分析与预警系统</h2>
<div class="box">
<form method="post">
<div class="item">
<label>空气温度 Air temperature [K]:</label>
<input type="number" step="0.1" name="air_temp" required>
</div>
<div class="item">
<label>工艺温度 Process temperature [K]:</label>
<input type="number" step="0.1" name="proc_temp" required>
</div>
<div class="item">
<label>转速 Rotational speed [rpm]:</label>
<input type="number" step="1" name="rot_speed" required>
</div>
<div class="item">
<label>扭矩 Torque [Nm]:</label>
<input type="number" step="0.1" name="torque" required>
</div>
<div class="item">
<label>工具磨损 Tool wear [min]:</label>
<input type="number" step="1" name="tool_wear" required>
</div>
<div class="item">
<label>产品类型 Type:</label>
<select name="prod_type">
<option value="L">L</option>
<option value="M">M</option>
<option value="H">H</option>
</select>
</div>
<div>
<button type="submit">提交预测预警</button>
</div>
</form>
</div>

<div class="box">
<h3>预测结果</h3>
{% if result_text %}
    {% if is_fault %}
        <p class="warn">⚠️ {{ result_text }}</p>
    {% else %}
        <p class="ok">✅ {{ result_text }}</p>
    {% endif %}
    <div class="debug-info">模型输出pred = {{pred_val}}</div>
{% else %}
<p>等待提交数据进行预测</p>
{% endif %}
</div>

</body>
</html>
"""


@app.route("/", methods=["GET","POST"])
def index():
    result_text = None
    is_fault = False
    pred_val = None
    if request.method == "POST":
        air_temp = float(request.form["air_temp"])
        proc_temp = float(request.form["proc_temp"])
        rot_speed = float(request.form["rot_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])
        prod_type = request.form["prod_type"]

        t_diff = proc_temp - air_temp
        power_load = (torque * rot_speed) * 2 * np.pi / 60
        tool_wear_norm = tool_wear / 250.0

        type_H = 1 if prod_type == "H" else 0
        type_L = 1 if prod_type == "L" else 0
        type_M = 1 if prod_type == "M" else 0

        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "T_diff[K]": t_diff,
            "Power_load": power_load,
            "Tool_wear_norm": tool_wear_norm,
            "Type_H": type_H,
            "Type_L": type_L,
            "Type_M": type_M
        }])[feature_cols]

        pred = model.predict(input_data)[0]
        pred_val = int(pred)
        print("====预测调试信息====")
        print("pred输出值 = ", pred)

        if pred == 1:
            result_text = "警告：设备存在故障风险！建议及时维护检修"
            is_fault = True
        else:
            result_text = "设备状态正常，无故障风险"
            is_fault = False
        print("result_text=", result_text, " is_fault=", is_fault)

    return render_template_string(HTML_TEMPLATE, result_text=result_text, is_fault=is_fault, pred_val=pred_val)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
