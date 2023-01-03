import os
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class LoadGPSData:
  def __init__(self, sequence_length = 10):
    self.sequence_length = sequence_length
    self.scaler = MinMaxScaler()
    #path for real and simulated data, to get the data from dataset
  def get_data(self, path):
    real_data = self.load_real(os.path.join(path, "data real"))
    sim_data = self.load_simulated(os.path.join(path, "data sim"))
        #Print length of the data
    print(len(real_data), len(sim_data))
    #test = read_csv(test_data)
    all_data = real_data + sim_data
        #shuffle the all the data and defined X as input and Y as output variables
    random.shuffle(all_data)
    X = []
    y = []
    for d in all_data:
      data, label = d
      if len(data)<self.sequence_length:
        print(len(data))
      X.append(data)
      y.append(label)
       #changing data to float and integer
    X = np.array(X).astype("float32")
    y = np.array(y).astype("int")[:, np.newaxis]

    return X, y

#Loading real data and also normalizing to imporove the accuracy
  def load_real(self, path):
    all_sub_dirs = os.listdir(path)
    all_sequences = []
    for sub_dir in all_sub_dirs:
      sub_dir_path = os.path.join(path, sub_dir)
      file_path = os.path.join(sub_dir_path, "dataset.csv")

      f = open(file_path, "r")
      all_data = f.readlines()
      f.close()
      collected_data = []
      i = 0
      for line in all_data:
        if i == 0:
          i += 1
          continue
        line = line.replace("\n", "")
        line = line.split(",")
        collected_data.append(line)

      collected_data = np.array(collected_data).astype("float32")

      self.scaler.fit(collected_data)
      collected_data = self.scaler.transform(collected_data)

      for i in range(0, len(collected_data), self.sequence_length):
        seq = collected_data[i:i+self.sequence_length]
        if len(seq)<self.sequence_length:
          continue
        all_sequences.append([seq, 0])

    return all_sequences

#loading simulated data
  def load_simulated(self, path):
    file_path = os.path.join(path, "data.csv")
    f = open(file_path, "r")
    all_data = f.readlines()
    f.close()
    collected_data = []
    all_sequences = []
    i = 0
    for line in all_data:
      if i == 0:
        print(line)
        i += 1
        continue
      line = line.replace("\n", "")
      line = line.replace(" ", " ")
      line = line.split(",")
      row = []

      for val in line:
        if len(val)>1:
          row.append(val)
      collected_data.append(row)

    collected_data = np.array(collected_data).astype("float32")

    self.scaler = MinMaxScaler()
    self.scaler.fit(collected_data)
    collected_data = self.scaler.transform(collected_data)

    for i in range(0, len(collected_data), self.sequence_length):
      seq = collected_data[i:i+self.sequence_length]
      if len(seq)<self.sequence_length:
        continue
      all_sequences.append([seq, 1])

    return all_sequences

#get test data function to test the model
  def get_test_data(self, path):
    file_path = os.path.join(path, "vdataset.csv")
    f = open(file_path, "r")
    all_data1 = f.readlines()
    f.close()
    collected_data = []
    all_sequences = []
    i = 0
    for line in all_data1:
      if i == 0:
        print(line)
        i += 1
        continue
      line = line.replace("\n", "")
      line = line.replace(" ", "")
      line = line.split(",")
      row = []

      for val in line:
        if len(val)>1:
          row.append(val)
      collected_data.append(row)

    collected_data = np.array(collected_data).astype("float32")

    self.scaler.fit(collected_data)
    collected_data = self.scaler.transform(collected_data)


    for i in range(0, len(collected_data), self.sequence_length):
      seq = collected_data[i:i+self.sequence_length]
      if len(seq)<self.sequence_length:
        continue
      all_sequences.append(seq)

    X = np.array(all_sequences).astype("float32")

    return X