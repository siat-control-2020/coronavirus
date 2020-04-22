import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(10, 8))
fig.suptitle('Predicted confirmed cases in four countries', fontsize=12)

USA_result = pd.read_csv(r"D:\pycharm_project\coronavirus\jieguo\USA_BAU.csv")
plt.subplot(221)
plt.plot(USA_result['Time'], (USA_result['3']-USA_result['2'])/1000, color='blue')
plt.plot(USA_result['Time'], (USA_result['4']-USA_result['2'])/1000, color='red')
plt.plot(USA_result['Time'], (USA_result['5']-USA_result['2'])/1000, color='green')
plt.plot(USA_result['Time'], (USA_result['6']-USA_result['2'])/1000, color='yellow')
plt.xlabel('Date(day)')
plt.ylabel('Population(1000)')
plt.legend(['Delay 1 weeks', 'Delay 2 weeks', 'Delay 3 weeks', 'Delay 4 weeks'], prop={'size': 8},
           ncol=1, fancybox=True, shadow=True)
plt.title('US', fontsize=10)

Italy_result = pd.read_csv("D:\pycharm_project\coronavirus\jieguo\Italy_BAU.csv")
plt.subplot(222)
plt.plot(Italy_result['Time'], (Italy_result['3']-Italy_result['2'])/1000, color='blue')
plt.plot(Italy_result['Time'], (Italy_result['4']-Italy_result['2'])/1000, color='red')
plt.plot(Italy_result['Time'], (Italy_result['5']-Italy_result['2'])/1000, color='green')
plt.plot(Italy_result['Time'], (Italy_result['6']-Italy_result['2'])/1000, color='yellow')
plt.xlabel('Date(day)')
plt.ylabel('Population(k)')
plt.legend(['Delay 1 weeks', 'Delay 2 weeks', 'Delay 3 weeks', 'Delay 4 weeks'], prop={'size': 8},
           ncol=1, fancybox=True, shadow=True)
plt.title('Italy', fontsize=10)

france_result = pd.read_csv("D:\pycharm_project\coronavirus\jieguo\France_BAU.csv")
plt.subplot(223)
plt.plot(france_result['Time'], (france_result['3']-france_result['2'])/1000, color='blue')
plt.plot(france_result['Time'], (france_result['4']-france_result['2'])/1000, color='red')
plt.plot(france_result['Time'], (france_result['5']-france_result['2'])/1000, color='green')
plt.plot(france_result['Time'], (france_result['6']-france_result['2'])/1000, color='yellow')
plt.xlabel('Date(day)')
plt.ylabel('Population(k)')
plt.legend(['Delay 1 weeks', 'Delay 2 weeks', 'Delay 3 weeks', 'Delay 4 weeks'], prop={'size': 8},
           ncol=1, fancybox=True, shadow=True)
plt.title('France', fontsize=10)

Germany_result = pd.read_csv("D:\pycharm_project\coronavirus\jieguo\Germany_BAU.csv")
plt.subplot(224)
plt.plot(Germany_result['Time'], (Germany_result['3']-Germany_result['2'])/1000, color='blue')
plt.plot(Germany_result['Time'], (Germany_result['4']-Germany_result['2'])/1000, color='red')
plt.plot(Germany_result['Time'], (Germany_result['5']-Germany_result['2'])/1000, color='green')
plt.plot(Germany_result['Time'], (Germany_result['6']-Germany_result['2'])/1000, color='yellow')
plt.xlabel('Date(day)')
plt.ylabel('Population(k)')
plt.legend(['Delay 1 weeks', 'Delay 2 weeks', 'Delay 3 weeks', 'Delay 4 weeks'], prop={'size': 8},
           ncol=1, fancybox=True, shadow=True)
plt.title('Germany', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
