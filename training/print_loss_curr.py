import re
import matplotlib.pyplot as plt

def create_picture(width, height, scalex, scaley, label, xlabel, ylabel, title):
    fig = plt.figure(figsize=(width, height), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, scalex)
    ax.set_ylim(0, scaley)
    plt.grid()
    plt.legend(label)
    return fig, ax



def plot_loss_curve1():
    # 读取 nohup.out 文件
    with open("/root/hybridGLAandNSA/nohup_pretrain.out", "r", encoding="utf-8") as f:
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

def plot_loss_curve2():

    loss_values = []
    accuracies = []
    steps = []

    # 匹配 'loss': 后的浮点数
    loss_pattern = re.compile(r"'loss'\s*:\s*([0-9.]+)")
    # 匹配 'mean_token_accuracy': 后的浮点数
    acc_pattern = re.compile(r"'mean_token_accuracy'\s*:\s*([0-9.]+)")

    with open("/root/hybridGLAandNSA/nohup_sft.out", 'r', encoding='utf-8') as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            acc_match = acc_pattern.search(line)

            if loss_match:
                steps.append(len(loss_values) * 10)
                loss_values.append(float(loss_match.group(1)))
            if acc_match:
                accuracies.append(float(acc_match.group(1)))

    steps = steps[::4] # 每4个取一个
    loss_values = loss_values[::4] # 每4个取一个
    accuracies = accuracies[::4] # 每4个取一个
    # 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("loss_curve1.png")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, label="Training Mean Token Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Mean Token Accuracy")
    plt.title("Acc Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("loss_acc1.png")

def plot_loss_curve3():
    steps = []
    loss_values = []
    accuracies = []
    # 'loss': 
    # 匹配 'loss': 后的浮点数
    loss_pattern = re.compile(r"'loss'\s*:\s*([0-9.]+)")
    # 'rewards/accuracies': 
    # 匹配 'rewards/accuracies': 后的浮点数
    acc_pattern = re.compile(r"'rewards/accuracies'\s*:\s*([0-9.]+)")

    eval_steps = []
    eval_loss_values = []
    eval_accuracies = []
    # eval_rewards/accuracies
    # 匹配 'eval_rewards/accuracies': 后的浮点数
    eval_acc_pattern = re.compile(r"'eval_rewards/accuracies'\s*:\s*([0-9.]+)")
    # eval_loss
    # 匹配 'eval_loss': 后的浮点数
    eval_loss_pattern = re.compile(r"'eval_loss'\s*:\s*([0-9.]+)")

    with open("/root/hybridGLAandNSA/nohup_dpo.out", "r", encoding="utf-8") as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            acc_match = acc_pattern.search(line)
            eval_loss_match = eval_loss_pattern.search(line)
            eval_acc_match = eval_acc_pattern.search(line)
            if loss_match:
                steps.append(len(loss_values) * 10)
                loss_values.append(float(loss_match.group(1)))
                assert acc_match is not None, f"acc_match is None, line: {line}"
                accuracies.append(float(acc_match.group(1)))
            
            if eval_loss_match:
                eval_steps.append((len(eval_loss_values) + 1) * 2000)
                eval_loss_values.append(float(eval_loss_match.group(1)))
                assert eval_acc_match is not None, f"eval_acc_match is None, line: {line}"
                eval_accuracies.append(float(eval_acc_match.group(1)))
    
    steps = steps[::2] # 每2个取一个
    loss_values = loss_values[::2] # 每2个取一个
    accuracies = accuracies[::2] # 每2个取一个
    # 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_values, label="Training Loss")
    plt.plot(eval_steps, eval_loss_values, label="Eval Loss", color='orange')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Dpo Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("dpo_loss_curve.png")

    # 绘制 accuracy 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, label="Training Reward Accuracy")
    plt.plot(eval_steps, eval_accuracies, label="Eval Reward Accuracy", color='orange')
    plt.xlabel("Step")
    plt.ylabel("Reward Accuracy")
    plt.title("Dpo Reward Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("dpo_acc_curve.png")
    



if __name__ == "__main__":
    # plot_loss_curve1()
    # plot_loss_curve2()
    plot_loss_curve3()