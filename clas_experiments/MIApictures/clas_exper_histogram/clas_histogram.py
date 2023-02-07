import numpy as np
from matplotlib import pyplot as plt

# 构造数据
labels = ['1', '2', '3', '4', '5','6','7','8','Mean&Var']
#resnet18
# data_a = [92.99, 93.19, 90.62, 92.53, 90.92, 91.29, 93.57, 92.77, 92.24] #baseline
# data_var_a = [0,0,0,0,0,0,0,0,1.27]
# data_b = [93.29, 93.26, 92.83, 92.26, 93.13, 93.73, 93.09, 91.92, 92.94] #LAB_NORM
# data_var_b = [0,0,0,0,0,0,0,0,0.34]
# data_c = [93.65, 94.52, 93.72, 94.92, 95.79, 94.36, 95.65, 96.06, 94.83] #HSV_TEM
# data_var_c = [0,0,0,0,0,0,0,0,0.86]
# data_d = [94.11, 94.79, 93.78, 95.45, 96.53, 94.74, 96.29, 95.73, 95.18] #HSV_NORM
# data_var_d = [0,0,0,0,0,0,0,0,0.98]
# data_e = [95.39, 94.90, 95.12, 95.22, 95.27, 94.91, 96.24, 95.98, 95.38] #SA
# data_var_e = [0,0,0,0,0,0,0,0,0.24]
# data_var = [1.27, 0.34, 0.86, 0.98, 0.24]
#resnet50
data_a = [92.48, 88.28, 88.28, 91.44, 90.35, 86.65, 89.22, 90.81, 89.69] #baseline
data_var_a = [0,0,0,0,0,0,0,0,3.72]
data_b = [92.74, 89.36, 89.85, 91.04, 90.92, 89.16, 88.61, 93.04, 90.59] #LAB_NORM
data_var_b = [0,0,0,0,0,0,0,0, 2.71]
data_c = [91.51, 92.79, 88.91, 90.68, 89.99, 90.98, 92.91, 92.38, 91.27] #HSV_TEM
data_var_c = [0,0,0,0,0,0,0,0, 1.99]
data_d = [92.33, 92.66, 88.76, 92.55, 90.35, 93.35, 93.92, 93.34, 92.16] #HSV_NORM
data_var_d = [0,0,0,0,0,0,0,0, 3.02]
data_e = [94.04, 94.23, 94.03, 94.72, 94.22, 93.73, 93.60, 95.11, 94.21] #SA
data_var_e = [0,0,0,0,0,0,0,0,0.25]


x = np.arange(len(labels))+0.5
width = .15

plt.rcParams['font.family'] = "Times New Roman"
# plots
# 色卡1 #58539f #bbbbd6 #eebabb #d86967 #9e3a3a
fig, ax = plt.subplots(figsize=(20, 12), dpi=200)
error_params=dict(elinewidth=3, ecolor='black')#,capsize=5)#设置误差标记参数
bar_a = ax.bar(x - width / 2, data_a, width, label='baseline', color='#1e592b', lw=.5, yerr=data_var_a, error_kw=error_params)
bar_b = ax.bar(x + width / 2, data_b, width, label='lab_prenorm', color='#4ea59f', lw=.5, yerr=data_var_b, error_kw=error_params)
bar_c = ax.bar(x + width *3 / 2, data_c, width, label='hsv_tem', color='#c7dbd5', lw=.5, yerr=data_var_c, error_kw=error_params)
bar_d = ax.bar(x + width * 5 / 2, data_d, width, label='hsv_prenorm', color='#f6b654', lw=.5, yerr=data_var_d, error_kw=error_params)
bar_e = ax.bar(x + width * 7 / 2, data_e, width, label='self attention', color='#ee6a5b', lw=.5, yerr=data_var_e, error_kw=error_params)
# 定制化设计
ax.tick_params(axis='x', direction='in', bottom=False)
ax.tick_params(axis='y', direction='out', labelsize=30, length=3)
ax.set_xticks(x+0.2)
ax.set_xticklabels(labels, size=30)
ax.set_ylim(bottom=85, top=100)
ax.set_yticks(np.arange(85, 101, step=1))

for spine in ['top', 'right']:
    ax.spines[spine].set_color('black')

ax.legend(fontsize=30, frameon=False)
ax.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax.spines['top'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.spines['right'].set_linewidth('2.0')

ax.set_xlabel('Experiments', fontsize = 30)
ax.set_ylabel('Best Test ACC', fontsize = 30)


# text_font = {'size': '14', 'weight': 'bold', 'color': 'black'}
# ax.text(.03, .93, "(a)", transform=ax.transAxes, fontdict=text_font, zorder=4)
plt.savefig(r'D:\CS\pythonProject\clas_histograms\clas100k_resnet50.png', width=20, height=12,
            dpi=100, bbox_inches='tight')
plt.show()