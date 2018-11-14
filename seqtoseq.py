#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\Data_Science\seqtoseq.py
# Project: d:\CodeWareHouse\Data_Science
# Created Date: Wednesday, August 15th 2018, 7:32:57 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Wednesday, 15th August 2018 7:42:20 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector


def build_seqtoseq_model(input_size, max_out_seq_len, hidden_size):

    model = Sequential()

    # Encoder(第一个 LSTM)
    model.add(LSTM(input_dim=input_size,
                   output_dim=hidden_size, return_sequences=False))

    model.add(Dense(hidden_size, activation="relu"))

    # 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
    model.add(RepeatVector(max_out_seq_len))

    # Decoder(第二个 LSTM)
    model.add(LSTM(hidden_size, return_sequences=True))

    # TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))

    model.compile(loss="mse", optimizer='adam')

    return model
