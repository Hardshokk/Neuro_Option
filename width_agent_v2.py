import pickle
import sys
import os
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Lambda, Conv2D, \
    MaxPooling2D, Reshape, Multiply, BatchNormalization, LSTM, UpSampling1D, Add, Dropout, \
    Concatenate, Conv1D, Subtract, PReLU, LeakyReLU
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LambdaCallback
# from tensorflow import set_random_seed
# set_random_seed(2)
path = "C:\\Projects\\Connector_v2\\common_files"
sys.path.append(path)
import common_functions as cf
pd.set_option("max_rows", 1000)


disable_eager_execution()


class EnvironmentData:
    """Среда, создает первичный датасет, затем из него формирует входы нейросети"""

    def __init__(self, symbol, tf, len_sequence, terminal, test=False):
        self.symbol = symbol
        self.tf = tf
        self.len_sequence = len_sequence
        self.terminal = terminal
        self.history_len = 20000
        self.bar_offset = 400
        self.len_slice = self.bar_offset + self.len_sequence
        # self.len_sequence берется два раза тк один кусок будет отрезаться при формировании вектора в 13 и 14 входах
        if not test:
            self.rates_frame, self.symbol_info = cf.get_bars_one_tf(self.symbol, self.tf,
                                                                    300 + self.bar_offset + self.len_sequence,
                                                                    self.history_len + self.bar_offset * 2 + self.len_sequence,
                                                                    self.terminal)
        elif test:
            self.rates_frame, self.symbol_info = cf.get_bars_one_tf(self.symbol, self.tf,
                                                                    1, 300 + self.bar_offset + self.len_sequence,
                                                                    self.terminal)
        self.mx = 1000  # Нормализация ценовых дельт
        self.mx_len_bar = 0.01  # Нормализация количества в барах
        self.dataset = {}

    # ***************************** Блок функций преобразования данных для входов ************************************ #
    @staticmethod
    def create_sequence_from_flatten(sequence, len_one_slice):
        """создаю срезы последовательности длиной len_one_slice"""
        data_list = []
        for i in range(len(sequence)):
            if i < len_one_slice:
                data_list.append(np.zeros((len_one_slice, 1)))
                continue
            data_list.append(np.array(sequence[i-len_one_slice:i]).reshape(len_one_slice, 1))
        return np.array(data_list)

    @staticmethod
    def perform_slice(on_slice, qv_slice):
        """выполняет отрезание неполных данных, которые использовались при подготовке истории"""
        return on_slice[qv_slice:-1]

    # ***************************** Блок функций подготовки данных для входов **************************************** #
    def four_dot(self, rates_frame):
        """формирует первый вход, от текущего закрытия рассчитываются соотношения к предыдущим len_sequence барам"""
        "возвращает лист формата (5800, 10, 4), 10 это длина последовательности, содержит знаки + и -, 10 первых нули"
        data_list = []
        for i in rates_frame.index:
            in_list = []
            "рассчитываю расстояния до четырех точек от текущей close на длину len_sequence"
            for ii in range(self.len_sequence):
                if i < self.len_sequence:
                    "забиваю нулями стартовый недостаток истории"
                    in_list.append(np.zeros(4, dtype=np.float32))
                    continue
                high_ = (rates_frame.at[i, 'close'] - rates_frame.at[i-ii, 'high']) * self.mx
                low_ = (rates_frame.at[i, 'close'] - rates_frame.at[i-ii, 'low']) * self.mx
                open_ = (rates_frame.at[i, 'close'] - rates_frame.at[i-ii, 'open']) * self.mx
                close_ = (rates_frame.at[i, 'close'] - rates_frame.at[i - ii, 'close']) * self.mx
                in_list.append(np.array([open_, high_, low_, close_]))
            data_list.append(in_list)
        return np.array(data_list)

    def select_direct(self, rates_frame):
        """формирует второй вход, присваивает направление -1 или 1"""
        data_list = []
        for i in rates_frame.index:
            data_list.append(1 if rates_frame.at[i, 'close'] - rates_frame.at[i, 'open'] > 0 else -1)
        data_list = np.array(data_list)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def atr_close_open(self, rates_frame):
        """формирует третий вход, считает атр закрытие минус открытие"""
        data_list = []
        for i in rates_frame.index:
            data_list.append((rates_frame.at[i, 'close'] - rates_frame.at[i, 'open']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def dif_high_now_low_pre(self, rates_frame):
        """формирует четвертый вход, просчитывает отклонение хая текущей свечи от лоу предыдущей"""
        data_list = []
        for i in rates_frame.index:
            if i == 0:
                data_list.append(0)
            else:
                data_list.append((rates_frame.at[i, 'high'] - rates_frame.at[i-1, 'low']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def dif_low_now_high_pre(self, rates_frame):
        """формирует пятый вход, просчитывает отклонение хая текущей свечи от лоу предыдущей"""
        data_list = []
        for i in rates_frame.index:
            if i == 0:
                data_list.append(0)
            else:
                data_list.append((rates_frame.at[i, 'low'] - rates_frame.at[i-1, 'high']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def atr_high_low(self, rates_frame):
        """формирует шестой вход, считает атр хай минус лоу"""
        data_list = []
        for i in rates_frame.index:
            data_list.append((rates_frame.at[i, 'high'] - rates_frame.at[i, 'low']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def get_hot_hour(self, rates_frame):
        """формирует седьмой вход, час как хотэнкодинг"""
        data_list = []
        for i in rates_frame.index:
            data_list.append(np.flipud(to_categorical(rates_frame.at[i, 'hour'], 24)))
        data_list = np.array(data_list)
        return data_list

    def get_hot_minute(self, rates_frame):
        """формирует восьмой вход, минута как хотэнкодинг, использовать этот вход только если тф ниже часа"""
        data_list = []
        for i in rates_frame.index:
            # используется 4 т.к. основным тф считается 15 минутка
            number = (60 - (rates_frame.at[i, 'minute'] + 15)) / 15
            data_list.append(np.flipud(to_categorical(number, 4)))
        data_list = np.array(data_list)
        return data_list

    def get_atr_high_shadow(self, rates_frame):
        """формирует девятый вход, атр верхней тени"""
        data_list = []
        for i in rates_frame.index:
            high_div_open = (rates_frame.at[i, 'high'] - rates_frame.at[i, 'open']) * self.mx
            high_div_close = (rates_frame.at[i, 'high'] - rates_frame.at[i, 'close']) * self.mx
            data_list.append(high_div_close if high_div_close < high_div_open else high_div_open)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def get_atr_low_shadow(self, rates_frame):
        """формирует десятый вход, атр нижней тени"""
        data_list = []
        for i in rates_frame.index:
            low_div_open = (rates_frame.at[i, 'open'] - rates_frame.at[i, 'low']) * self.mx
            low_div_close = (rates_frame.at[i, 'close'] - rates_frame.at[i, 'low']) * self.mx
            data_list.append(low_div_close if low_div_close < low_div_open else low_div_open)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def dif_high_now_high_pre(self, rates_frame):
        """формирует одиннадцатый вход, считает разницу между текущим хаем и предыдущим(проверить послед)"""
        data_list = []
        for i in rates_frame.index:
            if i == 0:
                data_list.append(0)
            else:
                data_list.append((rates_frame.at[i, 'high'] - rates_frame.at[i-1, 'high']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def dif_low_now_low_pre(self, rates_frame):
        """формирует двенадцатый вход, считает разницу между текущим лоу и предыдущим(проверить послед)"""
        data_list = []
        for i in rates_frame.index:
            if i == 0:
                data_list.append(0)
            else:
                data_list.append((rates_frame.at[i, 'low'] - rates_frame.at[i-1, 'low']) * self.mx)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def counter_high_qv_bars_back(self, rates_frame):
        """формирует тринадцатый вход, считает количество баров до удара в предыдущую свечу по этой же цене"""
        data_list = []
        for i in rates_frame.index:
            if i < self.bar_offset:
                data_list.append(0)
            else:
                counter = 1
                while not ((rates_frame.at[i, 'high'] <= rates_frame.at[i-counter, 'high']) and
                           (rates_frame.at[i, 'high'] >= rates_frame.at[i-counter, 'low'])):
                    counter += 1
                    if counter >= 400:
                        break
                data_list.append(counter * self.mx_len_bar)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    def counter_low_qv_bars_back(self, rates_frame):
        """формирует четырнадцатый вход, считает количество баров до удара в предыдущую свечу по этой же цене"""
        data_list = []
        for i in rates_frame.index:
            if i < self.bar_offset:
                data_list.append(0)
                continue
            else:
                counter = 1
                while not ((rates_frame.at[i, 'low'] <= rates_frame.at[i-counter, 'high']) and
                           (rates_frame.at[i, 'low'] >= rates_frame.at[i-counter, 'low'])):
                    counter += 1
                    if counter >= 400:
                        break
                data_list.append(counter * self.mx_len_bar)
        data_list = self.create_sequence_from_flatten(data_list, self.len_sequence)
        return data_list

    # ***************************** Блок сборки основного датафрейма ************************************************* #
    def get_targets(self, rates_on_slice):
        """функция подготавливает таргет фреймы"""
        rates_targets = []
        for i in range(len(rates_on_slice)):
            try:
                rates_targets.append(rates_on_slice[i+1][-1])
            except IndexError:
                rates_targets.append([0.])
        return np.array(rates_targets)

    def get_index_for_slice(self, rates_frame):
        """получаю метки времени соответствующие истории для дальнейшей срезки"""
        new_rates_frame = self.perform_slice(rates_frame, self.len_slice)
        new_rates_frame = new_rates_frame.reset_index(drop=True)
        time_rates = new_rates_frame[(new_rates_frame['hour'] >= 7) & (new_rates_frame['hour'] <= 20)]
        print("Срез датафрейма")
        print(time_rates[-10:])
        print("длина размера датафрейма ", len(new_rates_frame))
        index_list = list(time_rates.index)
        print("длина листа индексов ", len(index_list))
        print(index_list)
        return index_list

    @staticmethod
    def dop_selector(before_dop_select, list_index):
        """функция производит дополнительную селекцию по выделенным индексам"""
        selected_list = []
        for index in list_index:
            selected_list.append(before_dop_select[index])
        return np.array(selected_list)

    def create_main_input_dict(self, rates_frame):
        """функция создает единый словарь с данными для входов в нейросеть"""
        main_data_input_dict = {}
        main_target_input_dict = {}
        all_input_func_list = [
            self.four_dot,
            self.select_direct,
            self.atr_close_open,
            self.dif_high_now_low_pre,
            self.dif_low_now_high_pre,
            self.atr_high_low,
            self.get_hot_hour,
            self.get_hot_minute,
            self.get_atr_high_shadow,
            self.get_atr_low_shadow,
            self.dif_high_now_high_pre,
            self.dif_low_now_low_pre,
            self.counter_high_qv_bars_back,
            self.counter_low_qv_bars_back
        ]
        list_index = self.get_index_for_slice(rates_frame)
        for n in range(len(all_input_func_list)):
            rates_on_slice = all_input_func_list[n](rates_frame)
            train_before_dop_select = self.perform_slice(rates_on_slice, self.len_slice)
            main_data_input_dict[n] = self.dop_selector(train_before_dop_select, list_index)
            print(f"Вход {n + 1}  размерность: {main_data_input_dict[n].shape}")
            if (len(rates_on_slice.shape) == 3) and (rates_on_slice.shape[2] == 1):
                target_before_dop_select = self.perform_slice(self.get_targets(rates_on_slice), self.len_slice)
                main_target_input_dict[n] = self.dop_selector(target_before_dop_select, list_index)

                # print("main: ", main_data_input_dict[n][-2:], "target: ", main_target_input_dict[n][-2:])
        return main_data_input_dict, main_target_input_dict


class Environment:
    """Общий класс для всех сред, содержит общие повторяющиеся функции"""
    def __init__(self, symbol, tf, len_sequence, terminal, test):
        self.env_data = EnvironmentData(symbol, tf, len_sequence, terminal, test=test)
        self.data_set_x, self.data_set_y = self.env_data.create_main_input_dict(self.env_data.rates_frame)
        self.y_number_input = 1
        self.reward_list = []
        self.update_reward_list = []
        self.state_gen = None
        self.last_index = None

    def get_reward(self, action, index):
        """Вычисляет награду за последнее отданное действие"""
        y_true = self.data_set_y[self.y_number_input][index][0]
        if (action == 1) and (y_true == 1):
            self.reward_list.append(1)
        elif (action == -1) and (y_true == -1):
            self.reward_list.append(1)
        elif (action == 1) and (y_true == -1):
            self.reward_list.append(-1)
        elif (action == -1) and (y_true == 1):
            self.reward_list.append(-1)
        else:
            raise Exception(f"Немогу определить награду. y_true {y_true}, action {action}")
        # перед возвратом можно переопределить награду в зависимости от накопления
        if self.reward_list[-1] > 0:
            # counter = 1
            # while self.reward_list[-(counter - 1)] > 0:
            #     counter += 1
            #     if counter > len(self.reward_list):
            #         break
            self.update_reward_list.append(1)
        elif self.reward_list[-1] < 0:
            counter = 1
            while self.reward_list[-counter] < 0:
                # чтобы начать отсчет ошибки с -1 нужно counter перенести под условие проверки, сейчас отчет с -2
                counter += 1
                if counter >= len(self.reward_list):
                    break
            self.update_reward_list.append(-counter)
        return self.update_reward_list[-1]

    def create_generator(self):
        """Создаю или обновляю генераторы"""
        self.last_index = None
        self.reward_list.clear()
        self.state_gen = self.story_gen(self.data_set_x)
        self.state_gen.__next__()

    def story_gen(self, dataset):
        """генератор подающий состояние среды (историю)"""
        index = -1
        state = []
        while index < len(dataset[self.y_number_input]):
            yield {'state': state, 'index': index}
            index += 1
            state = {i: dataset[i][index: index+1] for i in dataset.keys()}
            self.last_index = index


class Agent:
    """нейро агент"""
    def __init__(self, load_saving_model=False, name_model_train="", name_model_inf="",
                 load_optimizer_weights=False):

        self.name_last = ""
        self.env_train_train_set = Environment(self.symbol, self.tf, self.qv_bars, self.terminal, False)
        self.env_train_train_set.create_generator()
        self.env_test_train_set = Environment(self.symbol, self.tf, self.qv_bars, self.terminal, False)
        self.env_test_train_set.create_generator()
        self.env_test_test_set = Environment(self.symbol, self.tf, self.qv_bars, self.terminal, True)
        self.env_test_test_set.create_generator()

        if load_saving_model:
            self.model_inf, self.model_train = self.get_model(name_model_inf, name_model_train, load_optimizer_weights)
        else:
            self.model_inf, self.model_train = self.get_model(name_model_inf, name_model_train, load_optimizer_weights)

    def save_test_log(self, metrics, name, variant_test=None):
        """сохраняю в лог метрику"""
        path_save = f"{self.path_dir}\\log"
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        with open(f"{path_save}\\{name}.txt", "a+", encoding="utf-8") as f:
            if variant_test == 0:
                f.write(f"\n ****Трейновый набор*** \n\n")
            elif variant_test == 1:
                f.write(f"\n\n\n ****Тестовый набор*** \n\n")
            for block_name in metrics:
                f.write(f"{block_name}: {metrics[block_name]}\n\n")

    def save_model(self, name):
        """сохранить модель"""
        if not os.path.exists(self.path_dir):
            os.mkdir(self.path_dir)
        self.model_inf.save(f'{self.path_dir}\\model_inf_{name}.h5')
        self.model_inf.save_weights(f'{self.path_dir}\\model_inf_{name}_weights.h5')
        self.model_train.save(f'{self.path_dir}\\model_train_{name}.h5')
        self.model_train.save_weights(f'{self.path_dir}\\model_inf_{name}_weights.h5')
        # сохранение состояния оптимизатора
        symbolic_weights = getattr(self.model_train.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with open(f'{self.path_dir}\\model_inf_{name}_optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

    @staticmethod
    def rewarded_loss(episode_reward):
        """переопределяем лосс"""
        def loss(y_true, y_pred):
            # tmpPred = Lambda(lambda x: keras.backend.clip(x, 0.05, 0.95))(yPred)
            # tmpLoss = Lambda(lambda x: -yTrue * keras.backend.log(x) - (1 - yTrue) *
            #                            (keras.backend.log(1 - x)))(tmpPred)
            print("yPred ", y_pred, "yTrue ", y_true, 'episode_reward ', episode_reward)
            tmpLoss = Lambda(lambda x: keras.backend.mean(keras.backend.square(y_true - x)))(y_pred)
            print(tmpLoss, episode_reward)
            policyLoss = Lambda(lambda x: x * tmpLoss)(episode_reward)
            # policyLoss = Add()([tmpLoss, episode_reward])
            return policyLoss
        return loss
    
    # def create_model(self):
    #     """Архитектура пример, метод переопределяется в блокнотах."""
    #     input_0 = Input((12, 4), batch_size=1, name='my_input_0')
    #     input_1 = Input((12, 1), batch_size=1, name='my_input_1')
    #     input_2 = Input((12, 1), batch_size=1, name='my_input_2')
    #     input_3 = Input((12, 1), batch_size=1, name='my_input_3')
    #     input_4 = Input((12, 1), batch_size=1, name='my_input_4')
    #     input_5 = Input((12, 1), batch_size=1, name='my_input_5')
    #     input_6 = Input(24, batch_size=1, name='my_input_6')
    #     input_7 = Input(4, batch_size=1, name='my_input_7')
    #     input_8 = Input((12, 1), batch_size=1, name='my_input_8')
    #     input_9 = Input((12, 1), batch_size=1, name='my_input_9')
    #     input_10 = Input((12, 1), batch_size=1, name='my_input_10')
    #     input_11 = Input((12, 1), batch_size=1, name='my_input_11')
    #     input_12 = Input((12, 1), batch_size=1, name='my_input_12')
    #     input_13 = Input((12, 1), batch_size=1, name='my_input_13')
    #
    #     lay0 = LSTM(12, activation='relu')(input_0)
    #     lay1 = LSTM(12, activation='relu')(input_1)
    #     lay2 = LSTM(12, activation='relu')(input_2)
    #     lay3 = LSTM(12, activation='relu')(input_3)
    #     lay4 = LSTM(12, activation='relu')(input_4)
    #     lay5 = LSTM(12, activation='relu')(input_5)
    #     lay6 = Dense(24, activation='relu')(input_6)
    #     lay7 = Dense(4, activation='relu')(input_7)
    #     lay8 = LSTM(12, activation='relu')(input_8)
    #     lay9 = LSTM(12, activation='relu')(input_9)
    #     lay10 = LSTM(12, activation='relu')(input_10)
    #     lay11 = LSTM(12, activation='relu')(input_11)
    #     lay12 = LSTM(12, activation='relu')(input_12)
    #     lay13 = LSTM(12, activation='relu')(input_13)
    #
    #     dense = Concatenate()([lay0, lay1, lay2, lay3, lay4, lay5, lay6, lay7, lay8, lay9, lay10, lay11, lay12, lay13])
    #     dense = Dense(500)(dense)
    #     dense = Dense(1, activation='sigmoid', name='my_output_1')(dense)
    #
    #     return [input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10,
    #             input_11, input_12, input_13], [dense]

    def get_model(self, name_model_inf, name_model_train, load_optimizer_weights):
        """создаю каркас нейросети"""
        episode_reward = Input(shape=(1))
        if (not name_model_inf) and (not name_model_train):
            input_lay, output_lay = self.create_model()
            # создаем основную модель
            model_inf = Model(inputs=input_lay, outputs=output_lay)
            # создаем обучаемую модель
            model_train = Model(inputs=[input_lay[0], input_lay[1],  input_lay[2], input_lay[3], input_lay[4],
                                        input_lay[5], input_lay[6], input_lay[7], input_lay[8], input_lay[9],
                                        input_lay[10], input_lay[11], input_lay[12], input_lay[13],
                                        episode_reward], outputs=output_lay)
        elif name_model_inf and name_model_train:
            model_inf = load_model(f"{self.path_dir}\\{name_model_inf}.h5", compile=False)
            model_train = Model(inputs=[model_inf.input, episode_reward],
                                outputs=model_inf.get_layer('my_output_1').get_output_at(0))
        # компиляция обучающей модели
        model_train.compile(optimizer=self.optimizer, loss=self.rewarded_loss(episode_reward),
                            experimental_run_tf_function=False)
        # подгрузить параметры оптимизатора
        if load_optimizer_weights and name_model_inf:
            model_train._make_train_function()
            with open(f'{self.path_dir}\\{name_model_inf}_optimizer.pkl', 'rb') as f:
                weight_values = pickle.load(f)
            model_train.optimizer.set_weights(weight_values)
        model_inf.summary()
        return model_inf, model_train

    def get_situation(self, last_state_dict):
        """Формирует ситуацию для подачи в нейросеть"""
        state = []
        for i in range(14):
            state.append(last_state_dict['state'][i])
        return state, last_state_dict['index']

    def predictor(self, state, index, url_set):
        """выполняет предикт и получает награду"""
        action_pred = self.model_inf.predict(state, batch_size=1)[0][0]
        action_select = np.random.choice(a=[1, -1], size=1, p=[action_pred, 1 - action_pred])
        reward = np.array([url_set.get_reward(action_select[0], index)])
        return action_pred, action_select, reward

    def model_fitting(self, epochs=1):
        """тренируем модель"""
        counter_epochs = 0
        counter_example = 0
        history = None
        while counter_epochs < epochs:
            counter_epochs += 1
            print(f"{'*' * 50} Эпоха {counter_epochs} {'*' * 50}\n")
            self.env_train_train_set.create_generator()
            while True:
                try:
                    last_state_dict = self.env_train_train_set.state_gen.__next__()
                except StopIteration:
                    print("СРАБОТАЛА ОСТАНОВКА ЦИКЛА")
                    break

                state, index = self.get_situation(last_state_dict)
                action_pred, action_select, reward = self.predictor(state, index, self.env_train_train_set)
                history = self.model_train.fit(x=[state[0], state[1],  state[2], state[3], state[4], state[5], state[6],
                                                  state[7], state[8], state[9], state[10], state[11], state[12], state[13],
                                                  reward],
                                               y=action_select,
                                               batch_size=1,
                                               epochs=1,
                                               callbacks=[],
                                               verbose=False)
                if (counter_example == 0) or (divmod(counter_example, 1000)[1] == 0):
                    print(f"Пример № {counter_example}, loss {history.history['loss'][-1]}")
                counter_example += 1
            dn = dt.datetime.now()
            time_offset = f"{dn.hour}{dn.minute}{dn.second}"
            self.name_last = f"full_loss_{round(history.history['loss'][-1], 5)}_{time_offset}"
            self.testing_model(self.name_last)
            self.save_model(self.name_last)
            print("\n\n")

    def update_testing_gen(self):
        """обновляет тестовые генераторы"""
        self.env_test_train_set.create_generator()
        self.env_test_test_set.create_generator()

    def control_testing_model(self, env_object, name):
        """метод выполняет контрольное тестирование модели, на вход принимает готовый генератор"""
        metrics = {'action_pred_list': [], 'action_select_list': [], 'reward_list': []}
        env_object.create_generator()
        while True:
            try:
                last_state_dict = env_object.state_gen.__next__()
                state, index = self.get_situation(last_state_dict)
                action_pred, action_select, reward = self.predictor(state, index, env_object)
                metrics['action_pred_list'].append(action_pred)
                metrics['action_select_list'].append(action_select[0])
                metrics['reward_list'].append(reward[0])
            except StopIteration:
                break
        with open(f"{name}_metrics.dict", 'wb') as f:
            writer = pickle.Pickler(f)
            writer.dump(metrics)
        self.calculate_metrics(1, metrics, name)

    def testing_model(self, name):
        """Тестрирование модели"""
        self.update_testing_gen()
        set_var_list = [self.env_test_train_set, self.env_test_test_set]
        for n in range(2):
            metrics = {'action_pred_list': [], 'action_select_list': [], 'reward_list': []}
            while True:
                try:
                    last_state_dict = set_var_list[n].state_gen.__next__()
                    state, index = self.get_situation(last_state_dict)
                    action_pred, action_select, reward = self.predictor(state, index, set_var_list[n])
                    metrics['action_pred_list'].append(action_pred)
                    metrics['action_select_list'].append(action_select[0])
                    metrics['reward_list'].append(reward[0])
                except StopIteration:
                    break
            self.calculate_metrics(n, metrics, name)

    def calculate_metrics(self, variant, metrics, model_name):
        """Проверки при обучении"""
        # суммарная проверка на повторяемость:
        summary_metrics_dict = {'errors': {}, 'wins': {}, 'sum_plus': 0, 'sum_minus': 0}

        def get_summary_reward(var):
            if not summary_metrics_dict[var].get(counter):
                summary_metrics_dict[var][counter] = 1
            else:
                summary_metrics_dict[var][counter] += 1

        i = 0
        while i < len(metrics['reward_list']):
            counter = 0
            if metrics['reward_list'][i] >= 0:
                try:
                    while metrics['reward_list'][i + counter] >= 0:
                        counter += 1
                    get_summary_reward('wins')
                except IndexError:
                    get_summary_reward('wins')
                    break
            elif metrics['reward_list'][i] < 0:
                try:
                    while metrics['reward_list'][i + counter] < 0:
                        counter += 1
                    get_summary_reward('errors')
                except:
                    get_summary_reward('errors')
                    break
            i += counter
        # суммарная проверка на количество:
        for i in metrics['reward_list']:
            if i >= 0:
                summary_metrics_dict['sum_plus'] += 1
            elif i < 0:
                summary_metrics_dict['sum_minus'] += 1
        # суммарная првоерка на качество предикта:
        summary_metrics_dict['quality_pred'] = metrics['action_pred_list'][:50]
        summary_metrics_dict['quality_select'] = list(metrics['action_select_list'][:50])
        # итоги
        if len(summary_metrics_dict['errors']):
            max_errors = max(list(summary_metrics_dict['errors'].keys()))
            eq_seven = sum(
                [summary_metrics_dict['errors'][i] if i >= 6 else 0 for i in summary_metrics_dict['errors'].keys()])
        else:
            max_errors = 0
            eq_seven = 0
            print(f"{metrics['reward_list']}")
        max_series_wins = max(list(summary_metrics_dict['wins'].keys()))

        if max_errors <= 5:
            print(f"Имя модели: {model_name}; Макс ошибок: {max_errors}; Больше_6: {eq_seven}, "
                  f"Макс профита: {max_series_wins}\n")
            print(f"Результаты: {summary_metrics_dict}")
            summary_metrics_dict['max_errors'] = max_errors
            summary_metrics_dict['Больше_6'] = eq_seven
            summary_metrics_dict['max_profit'] = max_series_wins
        else:
            print(f"Имя модели: {model_name}; Макс ошибок: {max_errors}; Больше_6: {eq_seven}, "
                  f"Макс профита: {max_series_wins}\n")
        self.save_test_log(summary_metrics_dict, model_name, variant_test=variant)
        return summary_metrics_dict


class TesterModel:
    """Класс тестирующий модель. Предназнечен для единоразового теста."""
    def __init__(self, model_name, model_path):
        self.model_inf = load_model(f"{model_path}\\{model_name}")
        dn = dt.datetime.now()
        self.model_path = model_path
        self.name_test = f"{model_name}_test_{dn.year}{dn.month}{dn.day}_{dn.hour}{dn.minute}"

    def control_testing_model(self, env_object):
        """метод выполняет контрольное тестирование модели, на вход принимает готовый генератор"""
        metrics = {'action_pred_list': [], 'action_select_list': [], 'reward_list': []}
        env_object.create_generator()
        while True:
            try:
                last_state_dict = env_object.state_gen.__next__()
                state, index = self.get_situation(last_state_dict)
                action_pred, action_select, reward = self.predictor(state, index, env_object)
                metrics['action_pred_list'].append(action_pred)
                metrics['action_select_list'].append(action_select[0])
                metrics['reward_list'].append(reward[0])
            except StopIteration:
                break
        directory = f"{self.model_path}\\tests"
        if not os.path.isdir(directory):
            os.mkdir(directory)
        with open(f"{directory}\\{self.name_test}_metrics.dict", 'wb') as f:
            writer = pickle.Pickler(f)
            writer.dump(metrics)
        self.calculate_metrics(1, metrics, self.name_test)

    def get_situation(self, last_state_dict):
        """Формирует ситуацию для подачи в нейросеть"""
        state = []
        for i in range(14):
            state.append(last_state_dict['state'][i])
        return state, last_state_dict['index']

    def predictor(self, state, index, url_set):
        """выполняет предикт и получает награду"""
        action_pred = self.model_inf.predict(state, batch_size=1)[0][0]
        action_select = np.random.choice(a=[1, -1], size=1, p=[action_pred, 1 - action_pred])
        reward = np.array([url_set.get_reward(action_select[0], index)])
        return action_pred, action_select, reward

    def calculate_metrics(self, variant, metrics, model_name):
        """Проверки при обучении"""
        # суммарная проверка на повторяемость:
        summary_metrics_dict = {'errors': {}, 'wins': {}, 'sum_plus': 0, 'sum_minus': 0}

        def get_summary_reward(var):
            if not summary_metrics_dict[var].get(counter):
                summary_metrics_dict[var][counter] = 1
            else:
                summary_metrics_dict[var][counter] += 1

        i = 0
        while i < len(metrics['reward_list']):
            counter = 0
            if metrics['reward_list'][i] >= 0:
                try:
                    while metrics['reward_list'][i + counter] >= 0:
                        counter += 1
                    get_summary_reward('wins')
                except IndexError:
                    get_summary_reward('wins')
                    break
            elif metrics['reward_list'][i] < 0:
                try:
                    while metrics['reward_list'][i + counter] < 0:
                        counter += 1
                    get_summary_reward('errors')
                except:
                    get_summary_reward('errors')
                    break
            i += counter
        # суммарная проверка на количество:
        for i in metrics['reward_list']:
            if i >= 0:
                summary_metrics_dict['sum_plus'] += 1
            elif i < 0:
                summary_metrics_dict['sum_minus'] += 1
        # суммарная првоерка на качество предикта:
        summary_metrics_dict['quality_pred'] = metrics['action_pred_list'][:50]
        summary_metrics_dict['quality_select'] = list(metrics['action_select_list'][:50])
        # итоги
        if len(summary_metrics_dict['errors']):
            max_errors = max(list(summary_metrics_dict['errors'].keys()))
            eq_seven = sum(
                [summary_metrics_dict['errors'][i] if i >= 6 else 0 for i in summary_metrics_dict['errors'].keys()])
        else:
            max_errors = 0
            eq_seven = 0
            print(f"{metrics['reward_list']}")
        max_series_wins = max(list(summary_metrics_dict['wins'].keys()))

        # if max_errors <= -1:
        #     print()
        #     # self.save_test_log(summary_metrics_dict, model_name, variant_test=variant)
        # else:
        if max_errors <= 5:
            print("Достигнута цель: Максимальное количество ошибок меньше 5")
        print(f"Имя модели: {model_name}; Макс ошибок: {max_errors}; Больше_6: {eq_seven}, "
              f"Макс профита: {max_series_wins}\n")
        print(f"Результаты: {summary_metrics_dict}")
        summary_metrics_dict['max_errors'] = max_errors
        summary_metrics_dict['Больше_6'] = eq_seven
        summary_metrics_dict['max_profit'] = max_series_wins
        # print(f"Имя модели: {model_name}; Макс ошибок: {max_errors}; Больше_6: {eq_seven}, "
        #       f"Макс профита: {max_series_wins}\n")
        # if (test_data_set and (max_errors > 5)):  # or ((not test_data_set) and (max_errors > 7)):
        #     self.drop_model(model_name)
        return summary_metrics_dict


"""
перенести основу в блокнот, и начать подбирать сети
дальше закодить метрики и логирование
"""

if __name__ == '__main__':
    # agent = Agent()
    # agent.model_fitting(3)

    rates, info = cf.get_bars_one_tf("EURUSD", mt5.TIMEFRAME_M15, 1, 5000, 0)
    print(rates.columns)
    env_data = EnvironmentData("EURUSD", mt5.TIMEFRAME_M15, 12, 0)
    env_data.create_main_input_dict(env_data.rates_frame)
    # direct_list = [1 if rates.loc[x, 'bar_direct'] == 'up_bar' else -1 for x in rates.index]
    # new_dir = [0]
    # for i in direct_list:
    #     new_dir.append(i+new_dir[-1])
    # plt.plot(new_dir)
    # plt.show()