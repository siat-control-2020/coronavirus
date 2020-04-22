import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pandas
from math import *
import datetime
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class Train_Predict_Model:
    def __init__(self, data: pandas.core.frame.DataFrame,
                 population: int, epoch=1000, rateIR=0.01, c=1, b=-3, alpha=0.1):
        self.epoch = epoch
        self.steps = len(data)
        # real data
        self.Infected = list(data['I'])
        self.Resistant = list(data['R'])
        self.Susceptible = list(population - data['I'] - data['R'])
        # estimate value
        self.S_pre = []
        self.I_pre = []
        self.R_pre = []

        # initial value
        self.c = c
        self.b = b
        self.alpha = alpha
        self.rateSI = self._calculate_beta(c=self.c, t=0, b=self.b, alpha=self.alpha)  # 初始beta
        self.rateIR = rateIR
        self.numIndividuals = population  # number of population
        self.results = None
        self.estimation = None
        self.modelRun = False
        self.loss = None
        self.betalist = []

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float):
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def _calculate_loss(self):
        return mean_squared_error(self.Infected, self.I_pre)

    def _calculate_MAPE(self):
        y = np.array(self.Infected)
        y_pred = np.array(self.I_pre)
        mape = np.abs((y - y_pred)) / np.abs(y)
        return np.mean(mape)

    # gradient descent to update parameter
    def _update(self):
        E = 2.71828182846
        alpha_eta = 0.000000000000001
        b_eta = 0.00000000001
        c_eta = 0.0000000000001
        alpha_temp = 0.0
        c_temp = 0.0
        b_temp = 0.0
        for t in range(0, self.steps):
            formula = E ** (self.alpha * (t + self.b))
            formula2 = E ** (-self.alpha * (t + self.b))

            loss_to_beta = -2 * (self.Infected[t] - self.I_pre[t]) * (self.I_pre[t]) * t * self.Susceptible[
                t] / self.numIndividuals

            # use chain rule to calculate partial derivatives
            beta_to_alpha = -self.c * formula * (t + self.b) * (formula - 1) * pow((1 + formula), -3)
            beta_to_b = -self.c * formula * self.alpha * (formula - 1) * pow((1 + formula), -3)
            beta_to_c = formula2 * pow((1 + formula2), -2)

            # calculate gradient
            alpha_temp += loss_to_beta * beta_to_alpha
            b_temp += loss_to_beta * beta_to_b
            c_temp += loss_to_beta * beta_to_c

        self.alpha -= alpha_eta * alpha_temp
        self.b -= b_eta * b_temp
        self.c -= c_eta * c_temp

    def train(self):
        for e in range(self.epoch):
            # estimate value
            self.S_pre = []
            self.I_pre = []
            self.R_pre = []

            for t in range(0, self.steps):
                if t == 0:
                    self.S_pre.append(self.Susceptible[0])
                    self.I_pre.append(self.Infected[0])
                    self.R_pre.append(self.Resistant[0])
                    self.rateSI = self._calculate_beta(c=self.c, t=t, b=self.b, alpha=self.alpha)
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSI)

                else:
                    self.rateSI = self._calculate_beta(c=self.c, t=t, b=self.b, alpha=self.alpha)
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSI)

                    # SIR
                    S_to_I = (self.rateSI * self.Susceptible[t] * self.Infected[t]) / self.numIndividuals
                    I_to_R = (self.rateIR * self.Infected[t])
                    self.S_pre.append(self.Susceptible[t] - S_to_I)
                    self.I_pre.append(self.Infected[t] + S_to_I - I_to_R)
                    self.R_pre.append(self.Resistant[t] + I_to_R)

            if e == (self.epoch - 1):
                self.estimation = pd.DataFrame.from_dict({'Time': list(range(len(self.Susceptible))),
                                                          'Estimated_Susceptible': self.S_pre,
                                                          'Estimated_Infected': self.I_pre,
                                                          'Estimated_Resistant': self.R_pre},
                                                         orient='index').transpose()

            self.loss = self._calculate_loss()
            self._update()

        return self.estimation

    def plot_fitted_result(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Infected'], color='red')
        plt.plot(self.estimation['Time'], real_obs['I'], color='y')

        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=90, fontsize=10)
        plt.xlabel('2020 Date')
        plt.ylabel('Population')
        plt.title('Fitted value in training', fontsize=20)
        plt.legend(['Estimated Infected', 'Actual Infected'], prop={'size': 12},
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=4, fancybox=True, shadow=True)
        plt.show()


class Predict_Model:
    def __init__(self, eons=1000, Susceptible=950, Infected=50, Resistant=0, rateIR=0.01,
                 alpha=0.3, c=5, b=-10, past_days=30):
        self.eons = eons  # how many days to predict
        self.Susceptible = Susceptible
        self.Infected = Infected
        self.Resistant = Resistant
        self.rateSI = None
        self.rateIR = rateIR
        self.numIndividuals = Susceptible + Infected + Resistant  # total population
        self.alpha = alpha
        self.c = c
        self.b = b
        self.past_days = past_days
        self.results = None
        self.modelRun = False

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float, past_days: int):
        t = t + past_days
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def run(self, death_rate):
        Susceptible = [self.Susceptible]
        Infected = [self.Infected]
        Resistant = [self.Resistant]

        for i in range(1, self.eons):
            self.rateSI = self._calculate_beta(c=self.c, t=i, b=self.b, alpha=self.alpha, past_days=self.past_days)
            S_to_I = (self.rateSI * Susceptible[-1] * Infected[-1]) / self.numIndividuals
            I_to_R = (Infected[-1] * self.rateIR)

            Susceptible.append(Susceptible[-1] - S_to_I)
            Infected.append(Infected[-1] + S_to_I - I_to_R)
            Resistant.append(Resistant[-1] + I_to_R)
        self.modelRun = True
        return self.results

    # plot no delay
    def plot_total_infected(self, title, ylabel, xlabel, result1):
        fig, ax = plt.subplots(figsize=(10, 6))

        temp = result1['Infected'] + result1['Resistant']
        plt.plot(result1['Time'], temp/1000, color='blue')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontsize=16)
        plt.show()

    # 画出不延时的对比图
    def plot_active_infected(self, title, ylabel, xlabel, result1):
        fig, ax = plt.subplots(figsize=(10, 6))

        temp = result1['Infected']
        plt.plot(result1['Time'], temp/1000, color='blue')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontsize=16)
        plt.show()


def save_data(path, result1, result2, result3, result4, result5, result6):
    s1 = list(result1['Time'])
    # delay 2 weeks
    s2 = list(result1['Infected'][0:14])
    s2.extend(list(result2['Infected']))
    # delay 3 weeks
    s3 = list(result1['Infected'][0:21])
    s3.extend(list(result3['Infected']))
    # delay 4 weeks
    s4 = list(result1['Infected'][0:28])
    s4.extend(list(result4['Infected']))
    # delay 5 weeks
    s5 = list(result1['Infected'][0:35])
    s5.extend(list(result5['Infected']))
    # delay 6 weeks
    s6 = list(result1['Infected'][0:42])
    s6.extend(list(result6['Infected']))

    result = pd.DataFrame({'Time': s1, '2': s2, '3': s3, '4': s4, '5': s5, '6': s6})
    result.to_csv(path)


def plot_sub_delay(result1, result2, result3, result4, result5, result6):
    fig, ax = plt.subplots(figsize=(10, 6))
    # delay 2 weeks
    temp2 = list(result1['Infected'][0:14])
    temp2.extend(list(result2['Infected']))
    # delay 3 weeks
    temp3 = list(result1['Infected'][0:21])
    temp3.extend(list(result3['Infected']))
    # delay 4 weeks
    temp4 = list(result1['Infected'][0:28])
    temp4.extend(list(result4['Infected']))
    # delay 5 weeks
    temp5 = list(result1['Infected'][0:35])
    temp5.extend(list(result5['Infected']))
    # delay 6 weeks
    temp6 = list(result1['Infected'][0:42])
    temp6.extend(list(result6['Infected']))

    temp = [(temp3[i] - temp2[i])/1000 for i in range(0, len(temp2))]
    plt.plot(result1['Time'], temp, color='blue')
    temp = [(temp4[i] - temp2[i])/1000 for i in range(0, len(temp2))]
    plt.plot(result1['Time'], temp, color='red')
    temp = [(temp5[i] - temp2[i])/1000 for i in range(0, len(temp2))]
    plt.plot(result1['Time'], temp, color='green')
    temp = [(temp6[i] - temp2[i])/1000 for i in range(0, len(temp2))]
    plt.plot(result1['Time'], temp, color='yellow')

    plt.xlabel('date(day)')
    plt.ylabel('population(1000)')
    plt.legend(['Delay 1 weeks', 'Delay 2 weeks', 'Delay 3 weeks', 'Delay 4 weeks'], prop={'size': 8},
               ncol=2, fancybox=True, shadow=True)
    plt.title('Germany', fontsize=16)
    plt.show()


if __name__ == '__main__':
    # data path
    df = pd.read_csv("D:\pycharm_project\coronavirus\dataChina.csv")
    df['date'] = pd.to_datetime(df['date'])
    Italy_population = 60430000
    USA_population = 318892103
    France_population = 66987244
    Germany_population = 82927922
    China_population = 1400050000
    Train_Model = Train_Predict_Model(epoch=500, data=df, population=China_population, rateIR=1 / 7, c=1,
                                      b=-10, alpha=0.025)
    estimate_df = Train_Model.train()
    estimate_beta = Train_Model.rateSI
    estimate_alpha = Train_Model.alpha
    estimate_b = Train_Model.b
    estimate_c = Train_Model.c
    print(estimate_beta)
    print(estimate_alpha)
    print(estimate_b)
    print(estimate_c)
    Train_Model.plot_fitted_result(df)

    I_pre = 38
    R_pre = 3
    S_pre = Italy_population - I_pre - R_pre
    # Pre_Model1：不管控，作为delay的初始数据计算
    Pre_Model1 = Predict_Model(eons=250, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=0.024, b=estimate_b, c=1, past_days=0)
    Pre_result1 = Pre_Model1.run(death_rate=0.07)
    print(Pre_result1['Infected'][14])
    print(Pre_result1['Infected'][21])
    print(Pre_result1['Infected'][28])
    print(Pre_result1['Infected'][35])
    print(Pre_result1['Infected'][42])

    # delay 2 weeks
    I_pre = Pre_result1['Infected'][14]
    R_pre = Pre_result1['Resistant'][14]
    S_pre = Italy_population - I_pre - R_pre
    Pre_Model2 = Predict_Model(eons=236, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=estimate_alpha, b=estimate_b, c=1, past_days=0)
    Pre_result2 = Pre_Model2.run(death_rate=0.07)
    Pre_Model2.plot_active_infected('Predicted active confirmed cases when delay 2 weeks in Germany',  'population(1000)','date(day)',Pre_result2)

    # delay 3 weeks
    I_pre = Pre_result1['Infected'][21]
    R_pre = Pre_result1['Resistant'][21]
    S_pre = Italy_population - I_pre - R_pre
    Pre_Model3 = Predict_Model(eons=229, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=estimate_alpha, b=estimate_b, c=1, past_days=0)
    Pre_result3 = Pre_Model3.run(death_rate=0.07)
    Pre_Model3.plot_active_infected('Predicted active confirmed cases when delay 3 weeks in Germany', 'population(1000)','date(day)',Pre_result3)

    # delay 4 weeks
    I_pre = Pre_result1['Infected'][28]
    R_pre = Pre_result1['Resistant'][28]
    S_pre = Italy_population - I_pre - R_pre
    Pre_Model4 = Predict_Model(eons=222, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=estimate_alpha, b=estimate_b, c=1, past_days=0)
    Pre_result4 = Pre_Model4.run(death_rate=0.07)
    Pre_Model4.plot_active_infected('Predicted active confirmed cases when delay 4 weeks in Germany', 'population(1000)','date(day)',Pre_result4)

    # delay 5 weeks
    I_pre = Pre_result1['Infected'][35]
    R_pre = Pre_result1['Resistant'][35]
    S_pre = Italy_population - I_pre - R_pre
    Pre_Model5 = Predict_Model(eons=215, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=estimate_alpha, b=estimate_b, c=1, past_days=0)
    Pre_result5 = Pre_Model5.run(death_rate=0.07)
    Pre_Model5.plot_active_infected('Predicted active confirmed cases when delay 5 weeks in Germany', 'population(1000)','date(day)', Pre_result5)

    # delay 6 weeks
    I_pre = Pre_result1['Infected'][42]
    R_pre = Pre_result1['Resistant'][42]
    S_pre = Italy_population - I_pre - R_pre
    Pre_Model6 = Predict_Model(eons=208, Susceptible=S_pre, Infected=I_pre, Resistant=R_pre, rateIR=1 / 7,
                               alpha=estimate_alpha, b=estimate_b, c=1, past_days=0)
    Pre_result6 = Pre_Model6.run(death_rate=0.07)
    Pre_Model6.plot_active_infected('Predicted active confirmed cases when delay 6 weeks in Germany', 'population(1000)','date(day)', Pre_result6)

    plot_sub_delay(Pre_result1, Pre_result2, Pre_result3, Pre_result4, Pre_result5, Pre_result6)

    save_data(r'D:\pycharm_project\coronavirus\jieguo\Italy_BAU.csv', Pre_result1, Pre_result2, Pre_result3, Pre_result4, Pre_result5, Pre_result6)