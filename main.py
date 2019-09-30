import re
import math
import random
from sklearn.naive_bayes import GaussianNB


def main():
    train_truthful = process_data('./DATASET/train/truthful.txt')
    train_fake = process_data('./DATASET/train/deceptive.txt')
    val_truthful = process_data('./DATASET/validation/truthful.txt')
    val_fake = process_data('./DATASET/validation/deceptive.txt')
    # combine validation sets
    val_all = val_truthful + val_fake
    test = process_data('./DATASET/test/test.txt')

    t_unigram_dict = create_unigram_dict(train_truthful)
    f_unigram_dict = create_unigram_dict(train_fake)
    t_bigram_dict = create_bigram_dict(train_truthful)
    f_bigram_dict = create_bigram_dict(train_fake)

    # unigram
    unigram_val_preds = unigram_classifier(t_unigram_dict, f_unigram_dict, val_all, True)
    numTrue = len(val_truthful); numFake = len(val_fake); i = 0; numCorrect = 0
    while i < numTrue:
        if unigram_val_preds[i][1] == 0:
            numCorrect+=1
        i+=1
    while i < len(unigram_val_preds):
        if unigram_val_preds[i][1] == 1:
            numCorrect+=1
        i+=1
    accuracy = float(numCorrect)/len(unigram_val_preds)
    print("unigram accuracy: "+ str(accuracy))

    # bigram
    bigram_val_preds = bigram_classifier(t_bigram_dict, f_bigram_dict,
    t_unigram_dict, f_unigram_dict, val_all, True)
    numTrue = len(val_truthful); numFake = len(val_fake); i = 0; numCorrect = 0
    while i < numTrue:
        if bigram_val_preds[i][1] == 0:
            numCorrect+=1
        i+=1
    while i < len(unigram_val_preds):
        if bigram_val_preds[i][1] == 1:
            numCorrect+=1
        i+=1
    accuracy = float(numCorrect)/len(bigram_val_preds)
    print("bigram accuracy" + str(accuracy))

    #naive bayes
    whole_dict = create_unigram_dict_no_unkown(train_truthful+train_fake)
    truth_xs, truth_ys = create_nb_input(train_truthful, whole_dict, 0)
    spam_xs, spam_ys = create_nb_input(train_fake, whole_dict, 1)
    inp_xs = truth_xs+spam_xs
    inp_ys = truth_ys+spam_ys
    gnb = GaussianNB()
    gnb.fit(inp_xs, inp_ys)
    y_pred = gnb.predict(val_all)
    print("Number of mislabeled points out of a total %d points : %d"
    % (len(xs),([0]*(len(val_truthful))+[1]*(len(val_fake)) != y_pred).sum()))


    return accuracy

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


# text_file is the path of the file to process
def process_data_unigram(text_file):
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
        if "\n" not in i:
            curr_lst.append(i)
        else:
            f_lst.append(curr_lst)
            curr_lst = []
            split = i.split("\n")
            curr_lst.append(split[1])
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
            result[elt] = result[elt]+1
    return result

def create_bigram_dict(lst):
    result = {}
    result["<unk>"] = 0
    flat_lst = []
    for sublist in lst:
        for item in sublist:
            flat_lst.append(item)
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

#for naive_bayes
def create_unigram_dict_no_unkown(lst):
    result = {}
    flat_list = []
    for sublist in lst:
        for item in sublist:
            flat_list.append(item)
    for elt in flat_list:
        if elt not in result:
            result[elt] = 1
        else:
            result[elt] = result[elt]+1
    return result

def create_nb_input(lst, dic, truthful):
    xs = []
    for sublist in lst:
        new_dict = dic.fromkeys(dic, 0)
        for elt in sublist:
            new_dict[elt] = new_dict[elt]+1
        l = new_dict.values()
        xs.append(l)
    return xs, [truthful]*(len(lst))

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
                probs.append((top_num + k)/(bottom_number + len(unigram_dict)))
            else:
                probs.append(top_num/bottom_number)
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

if __name__ == "__main__":
    main()
