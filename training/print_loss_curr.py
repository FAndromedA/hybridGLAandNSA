import re
import matplotlib.pyplot as plt

# 读取 nohup.out 文件
with open("/root/hybridGlaAndNsa/nohup.out", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 正则表达式匹配 loss 值
loss_pattern = re.compile(r"loss:\s*([\d\.]+)")

temp_loss =0.0
sample_step = 3
loss_values = []
steps = []

# 遍历每一行，提取 loss 值
for i, line in enumerate(lines):
    match = loss_pattern.search(line)
    if match:
        # if i < 500: 
        #     continue

        temp_loss += float(match.group(1))
        if i % 8 == 0:  # 每8行取一次
            sample_step += 1
            if sample_step % 3 == 0: 
                loss_values.append(temp_loss / 8)
                steps.append(len(loss_values) * 8 * 8 * 3)  # accumulate steps = 8, 每8行取一次, so steps = 8 * 8
            temp_loss = 0.0
        
        # if i % 25 == 0:  # 每8行取一次
        #     loss_values.append(float(match.group(1)))
        #     steps.append(len(loss_values) * 8 * 25)

# 绘制 loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_values, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
plt.show()
plt.savefig("loss_curve.png")