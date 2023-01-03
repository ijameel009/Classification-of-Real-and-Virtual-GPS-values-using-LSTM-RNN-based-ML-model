import os

label_file_path = "dataformat.txt"
data_dir = "data"


required_labels = ['lat', 'lon', 'alt', 'vf','vl','vu']

def load_label_file(file, required_labels):
    f = open(file)
    all_data = f.readlines()
    f.close()


    all_labels_dict = {}
    idx = 0
    for line in all_data:
        line = line.replace("\n", "")
        line = line.split(":")
        lab = line[0]
        if lab in required_labels:
            all_labels_dict[lab] = idx
        idx += 1

    return all_labels_dict

def load_data_file(file):
    f = open(file, "r")
    all_data = f.readlines()
    f.close()

    all_data = all_data[0].replace("\n", "")
    return all_data.split(" ")

def make_dataset(dataset_dir, all_labels_dict):
    all_files = os.listdir(dataset_dir)

    f = open("dataset.csv", "w")
    header = ""
    for lab in all_labels_dict:
        header += f",{lab}"
    f.write(header[1:] + "\n")
    for file_name in all_files:
        path = os.path.join(dataset_dir, file_name)
        file_data = load_data_file(path)
        line = ""
        for lab in all_labels_dict:
            idx = all_labels_dict[lab]
            line += f",{file_data[idx]}"
        f.write(line[1:] + "\n")
    f.close()

all_labels_dict = load_label_file(label_file_path, required_labels)
print(all_labels_dict)

make_dataset(data_dir, all_labels_dict)