import re
import math

def createPredictions():
    train_truthful = process_data('./DATASET/train/truthful.txt', 'r')
    train_deceptive = process_data('./DATASET/train/deceptive.txt', 'r')
    val_truthful = process_data('./DATASET/validation/truthful.txt', 'r')
    val_deceptive = process_data('./DATASET/validation/deceptive.txt', 'r')
    test = process_data('./DATASET/test/test.txt', 'r')

    t_unigram_dict = create_unigram_dict(train_truthful)
    d_unigram_dict = create_unigram_dict(train_deceptive)
    t_bigram_dict = create_bigram_dict(train_truthful)
    d_bigram_dict = create_bigram_dict(train_deceptive)

    # call unigram/bigram classifiers


# text_file is the path of the file to process
def process_data(text_file):
    #f = open('./DATASET/train/truthful.txt', 'r')
    f = open(text_file)
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
    result["<unk>"] = 0
    flat_list = []
    for sublist in lst:
        for item in sublist:
            flat_list.append(item)
    for elt in flat_list:
        if elt not in result:
            result["<unk>"] += 1
            result[elt] = 0
        else:
            count = result[elt]
            result[elt] = count+1
    return result

def create_bigram_dict(lst):
    result = {}
    result["<unk>"] = 0
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
                result["<unk>"] += 1
                result[curr_tup] = 0
            else:
                result[curr_tup] += 1
    return result

def create_total_count(dict):
    total_count = 0
    for w in dict:
        total_count += (dict[w])
    return total_count

# k is the smoothing amount
def unigram_classifer(unigram_dict_real, unigram_dict_fake, test_set, smoothing=False, k=1):
    test_lst = process_data(test_set)
    pred_lst = []
    for review_index in range(len(test_lst)):
        review = test_lst[review_index]
        fake_prob = helper_unigram(review, unigram_dict_fake, smoothing, k)
        fake_perplexity = (fake_prob) ** (-1/len(review))
        real_prob = helper_unigram(review, unigram_dict_real, smoothing, k)
        real_perplexity = (real_prob) ** (-1/len(review))
        if fake_perplexity < real_perplexity:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst

def bigram_classifier(bigram_dict_real, bigram_dict_fake, udict_real, udict_fake,
 test_set, smoothing=False, k=1):
    test_lst = process_data(test_set)
    pred_lst = []
    for review_index in range(len(test_lst)):
        review = test_lst[review_index]
        fake_prob = helper_bigram(review, bigram_dict_fake, udict_fake,smoothing,k)
        fake_perplexity = (fake_prob) ** (-1/(len(review)))
        real_prob = helper_bigram(review, bigram_dict_real, udict_real,smoothing,k)
        real_perplexity = (real_prob) ** (-1/(len(review)))
        if fake_perplexity < real_perplexity:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst

# probabilities is a list of the probabilities to multiply - use this function in the classifiers
def perplexity(probabilities):
    summation = 0
    for p in probabilities:
        summation += -1*math.log(p)
    return math.exp(1/n*summation)

# if smoothing is True, probability will be calculated with add-k smoothing
def helper_bigram(review_str, bigram_dict, unigram_dict, smoothing, k):
    curr_prob = 1
    for w_index in range(0, len(review_str)-1):
        curr_tup = (review_str[w_index], review_str[w_index + 1])
        curr_bigram_count = bigram_dict[curr_tup]
        bottom_number = unigram_dict[review_str[w_index]]
        if smoothing:
            curr_prob *= (curr_bigram_count + k)/(bottom_number + len(unigram_dict))
        else:
            curr_prob *= (curr_bigram_count/bottom_number)
    return curr_prob


def helper_unigram(review_str, unigram_dict, smoothing = False, k = 1):
    curr_prob = 1
    total_count = create_total_count(unigram_dict)
    for w in review_str:
        if smoothing:
            curr_prob *= (unigram_dict[w] + k)/(total_count + len(unigram_dict))
        else:
            curr_prob *= unigram_dict[w]/total_count
    return curr_prob
