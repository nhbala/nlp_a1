import re

def process_data():
    f = open('./DATASET/train/truthful.txt', 'r')
    file = f.read()
    split_arr = file.split(" ")
    regex = re.compile('[a-zA-Z]')
    filtered = [i for i in split_arr if regex.search(i)]
    final = [x.lower() for x in filtered]
    f_lst = []
    curr_lst = []
    for index in range(len(final)):
        i = final[index]
        if index == 0:
            curr_lst.append("<s>")
            curr_lst.append(i)
        if "\n" not in i:
            curr_lst.append(i)
        else:
            f_lst.append(curr_lst)
            curr_lst = []
            curr_lst.append("<s>")
            split = i.split("\n")
            curr_lst.append(split[1])
    for i in f_lst:
        i.append("<e>")
    return f_lst
