import urllib.request
import numpy as np
import wget
import csv
import time


list_file = "subjects_ABIDEI.txt"
subject_IDs = np.genfromtxt(list_file, dtype=str)
csv_file = "Phenotypic_V1_0b_preprocessed1.csv"
out_directory = "D:\ABIDE"

file_name_list = []

with open(csv_file) as csv_:

    reader = csv.DictReader(csv_)
    for row in reader:

        if row['SUB_ID'] in subject_IDs:

            file_name_list.append(str(row['FILE_ID']))

for i in range(len(file_name_list)):

    #file_name = "Pitt_0050003"
    result = "not"
    print(file_name_list[i])
    print("[" + str(i) + "/" + str(len(file_name_list)) + "]")
    url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/" + file_name_list[i] + "_func_preproc.nii.gz"

    target = out_directory + file_name_list[i] + "_func_preproc.nii.gz"

    #response = urllib.request.urlopen(url)
    #html = response.read()

    while (result != target):

        try:
            result = wget.download(url, out = out_directory)
            #print(result)
            time.sleep(5)
        except:
            continue
    #print(result)
