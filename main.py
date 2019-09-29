import re
import math

def createPredictions():
    train_truthful = process_data('./DATASET/train/truthful.txt')
    train_fake = process_data('./DATASET/train/deceptive.txt')
    val_truthful = process_data('./DATASET/validation/truthful.txt')
    val_fake = process_data('./DATASET/validation/deceptive.txt')
    # combine validation sets
    val_all = val_truthful + val_fake
    test = process_data('./DATASET/test/test.txt', 'r')

    t_unigram_dict = create_unigram_dict(train_truthful)
    f_unigram_dict = create_unigram_dict(train_deceptive)
    t_bigram_dict = create_bigram_dict(train_truthful)
    f_bigram_dict = create_bigram_dict(train_deceptive)

    # call unigram/bigram classifiers
    unigram_val_preds = unigram_classifier(t_unigram_dict, f_unigram_dict, val_all, True)


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
def unigram_classifier(unigram_dict_real, unigram_dict_fake, test_set, smoothing=False, k=1):
    pred_lst = []
    for review_index in range(len(test_set)):
        review = test_set[review_index]
        fake_pp = helper_unigram(review, unigram_dict_fake, smoothing, k)
        real_pp = helper_unigram(review, unigram_dict_real, smoothing, k)
        if fake_pp < real_pp:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst

def bigram_classifier(bigram_dict_real, bigram_dict_fake, udict_real, udict_fake,
 test_set, smoothing=False, k=1):
    pred_lst = []
    for review_index in range(len(test_set)):
        review = test_set[review_index]
        fake_pp = helper_bigram(review, bigram_dict_fake, udict_fake,smoothing,k)
        real_pp = helper_bigram(review, bigram_dict_real, udict_real,smoothing,k)
        if fake_pp < real_pp:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst

# probabilities is a list of the probs to multiply
def perplexity(probabilities):
    print("perp")
    summation = 0
    for p in probabilities:
        summation += -1*math.log(p)
    return math.exp(1/len(probabilities)*summation)

# Returns the perplexity of a test review
# if smoothing is True, probability will be calculated with add-k smoothing
def helper_bigram(review_str, bigram_dict, unigram_dict, smoothing, k):
    probs = []
    total_count = create_total_count(bigram_dict)
    for w_index in range(0, len(review_str)-1):
        curr_tup = (review_str[w_index], review_str[w_index + 1])
        top_num = bigram_dict.get(curr_tup, 0)
        if top_num == 0:
            top_num = bigram_dict["<unk>"]
            probs.append(top_num/total_count)
        else:
            bottom_number = unigram_dict.get(review_str[w_index], 0)
            if smoothing:
                probs.append(curr_bigram_count + k)/(bottom_number + len(unigram_dict))
            else:
                probs.append(curr_bigram_count/bottom_number)
    return perplexity(probs)


def helper_unigram(review_str, unigram_dict, smoothing = False, k = 1):
    probs = []
    total_count = create_total_count(unigram_dict)
    for w in review_str:
        top_num = unigram_dict.get(w, 0)
        if top_num == 0:
            top_num = unigram_dict["<unk>"]
        if smoothing:
            probs.append((top_num + k)/(total_count + len(unigram_dict)))
        else:
            probs.append(top_num/total_count)
    return perplexity(probs)
