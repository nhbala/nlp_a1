import re

def process_data(text_file):
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

def create_unigram_dict(lst):
    result = {}
    flat_list = []
    for sublist in lst:
        for item in sublist:
            flat_list.append(item)
    for elt in flat_list:
        if elt not in result:
            result[elt] = 1
        else:
            count = result[elt]
            result[elt] = count+1
    return result

def create_bigram_dict(lst):
    result = {}
    flat_list = []
    for sublist in lst:
        for item in sublist:
            flat_list.append(item)
    for elt_index in range(len(flat_lst)):
        if elt_index == len(flat_lst) - 1:
            break
        else:
            curr_tup = (flat_lst[elt_index], flat_lst[elt_index+1])
            if curr_tup == ("<s>", "<e>"):
                continue
            elif curr_tup not in result:
                result[curr_tup] = 1
            else:
                result[curr_tup] += 1
    return result

def create_total_count(dict):
    total_count = 0
    for w in dict:
        total_count += (dict[w])
    return total_count

def unigram_classifer(unigram_dict_real, unigram_dict_fake, test_set):
    test_lst = process_data(test_set)
    pred_lst = []
    for review_index in range(len(test_lst)):
        review = test_lst[review_index]
        fake_prob = helper_unigram(review, unigram_dict_fake)
        fake_perplexity = (fake_prob) ** (-1/len(review))
        real_prob = helper_unigram(review, unigram_dict_real)
        real_perplexity = (real_prob) ** (-1/len(review))
        if fake_perplexity < real_perplexity:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst








def helper_unigram(review_str, unigram_dict):
    curr_prob = 1
    total_count = create_total_count(unigram_dict)
    for w in review_str:
        prob = unigram_dict[w]/total_count
        curr_prob *= prob
    return curr_prob
