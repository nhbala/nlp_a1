import re
import math
import random
from sklearn.naive_bayes import GaussianNB

def main():
    #test = process_data('./DATASET/test/test.txt')

    # UNIGRAM
    train_truthful = process_data_unigram('./DATASET/train/truthful.txt')
    train_fake = process_data_unigram('./DATASET/train/deceptive.txt')
    t_unigram_dict = create_unigram_dict(train_truthful)
    f_unigram_dict = create_unigram_dict(train_fake)
    val_truthful = process_data_unigram('./DATASET/validation/truthful.txt')
    val_fake = process_data_unigram('./DATASET/validation/deceptive.txt')
    # combine validation sets
    val_all = val_truthful + val_fake
    unigram_val_preds = unigram_classifier(t_unigram_dict, f_unigram_dict, val_all, True, 0.5)
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
    print(len(val_all))

    # BIGRAM
    train_truthful = process_data_bigram('./DATASET/train/truthful.txt')
    train_fake = process_data_bigram('./DATASET/train/deceptive.txt')
    t_bigram_dict = create_bigram_dict(train_truthful)
    f_bigram_dict = create_bigram_dict(train_fake)
    t_unigram_dict = create_unigram_dict(train_truthful)
    f_unigram_dict = create_unigram_dict(train_fake)
    val_truthful = process_data_bigram('./DATASET/validation/truthful.txt')
    val_fake = process_data_bigram('./DATASET/validation/deceptive.txt')
    # combine validation sets
    val_all = val_truthful + val_fake
    bigram_val_preds = bigram_classifier(t_bigram_dict, f_bigram_dict,
    t_unigram_dict, f_unigram_dict, val_all, True, 0.1)
    numTrue = len(val_truthful); numFake = len(val_fake); i = 0; numCorrect = 0
    while i < numTrue:
        if bigram_val_preds[i][1] == 0:
            numCorrect+=1
        i+=1
    while i < len(bigram_val_preds):
        if bigram_val_preds[i][1] == 1:
            numCorrect+=1
        i+=1
    accuracy = float(numCorrect)/len(bigram_val_preds)
    print("bigram accuracy: " + str(accuracy))
    print(len(val_all))

    #naive bayes unigram
    whole_dict = create_unigram_dict_no_unkown(train_truthful+train_fake)
    truth_xs, truth_ys = create_nb_input(train_truthful, whole_dict, 0)
    spam_xs, spam_ys = create_nb_input(train_fake, whole_dict, 1)
    inp_xs = truth_xs+spam_xs
    inp_ys = truth_ys+spam_ys
    gnb = GaussianNB()
    gnb.fit(inp_xs, inp_ys)
    val_xs, dont_use_this = create_nb_input(val_all, whole_dict, 0)
    y_pred = gnb.predict(val_xs)
    print("Number of mislabeled points out of a total %d points : %d"
    % (len(val_xs),([0]*(len(val_truthful))+[1]*(len(val_fake)) != y_pred).sum()))

    #naive bayes unigram + bigram
    whole_dict = create_unigram_dict_no_unkown(train_truthful+train_fake)
    whole_dict2 = create_bigram_dict_no_unkown(train_truthful+train_fake)
    whole_dict3 = create_trigram_dict_no_unkown(train_truthful+train_fake)
    truth_xs, truth_ys = create_nb_input_bigram(train_truthful, whole_dict, whole_dict2, whole_dict3, 0)
    spam_xs, spam_ys = create_nb_input_bigram(train_fake, whole_dict, whole_dict2, whole_dict3, 1)
    inp_xs = truth_xs + spam_xs
    inp_ys = truth_ys + spam_ys
    gnb = GaussianNB()
    gnb.fit(inp_xs, inp_ys)
    val_xs, dont_use_this = create_nb_input_bigram(val_all, whole_dict, whole_dict2, whole_dict3, 0)
    y_pred = gnb.predict(val_xs)
    print("Number of mislabeled points out of a total %d points : %d"
    % (len(val_xs),([0]*(len(val_truthful))+[1]*(len(val_fake)) != y_pred).sum()))

    #calculate perplexity values
    fake_w_real = perplexity_all(val_fake, t_bigram_dict, t_unigram_dict)
    fake_w_fake = perplexity_all(val_fake, f_bigram_dict, f_unigram_dict)
    real_w_fake = perplexity_all(val_truthful, f_bigram_dict, f_unigram_dict)
    real_w_real = perplexity_all(val_truthful, t_bigram_dict, t_unigram_dict)
    print("fake_w_real:" + str(fake_w_real))
    print("fake_w_fake:" + str(fake_w_fake))
    print("real_w_real:" + str(real_w_real))
    print("real_w_fake:" + str(real_w_fake))


# text_file is the path of the file to process
def process_data_bigram(text_file):
    f = open(text_file)
    file = f.read()
    split_arr = file.split(" ")
    final = [x.lower() for x in split_arr]
    f_lst = []
    curr_lst = []
    for index in range(len(final)):
        i = final[index]
        if re.match("^[A-Za-z0-9_-]*$", i) == False:
            continue
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
    print("bigram length:" + str(len(f_lst)))
    return f_lst


# text_file is the path of the file to process
def process_data_unigram(text_file):
    #f = open('./DATASET/train/truthful.txt', 'r')
    f = open(text_file)
    file = f.read()
    split_arr = file.split(" ")
    #regex = re.compile('[a-zA-Z]')
    #filtered = [i for i in split_arr if regex.search(i)]
    #final = [x.lower() for x in filtered]
    final = [x.lower() for x in split_arr]
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
    print("unigram length:" + str(len(f_lst)))
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
            r = random.randint(0,10)
            if r <= 0: # 10% of first occurences go to unknown
                result["<unk>"] += 1
                result[elt] = 0
            else:
                result[elt] = 1
        else:
            result[elt] = result[elt]+1
    return result

def create_bigram_dict(lst):
    result = {}
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
                result[curr_tup] = 1
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

def create_bigram_dict_no_unkown(lst):
    result = {}
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
                result[curr_tup] = 1
            else:
                result[curr_tup] += 1
    return result

def create_trigram_dict_no_unkown(lst):
    result = {}
    flat_lst = []
    for sublist in lst:
        for item in sublist:
            flat_lst.append(item)
    for elt_index in range(len(flat_lst)):
        if elt_index == len(flat_lst) - 2:
            break
        else:
            curr_tup = (flat_lst[elt_index], flat_lst[elt_index+1], flat_lst[elt_index+2])
            if curr_tup[0] == "<s>" and curr_tup[1] == "<e>" or curr_tup[1] == "<s>" and curr_tup[2] == "<e>":
                continue
            elif curr_tup not in result:
                result[curr_tup] = 1
            else:
                result[curr_tup] += 1
    return result

def create_nb_input(lst, dic, truthful):
    xs = []
    for sublist in lst:
        new_dict = dic.fromkeys(dic, 0)
        for elt in sublist:
            if elt in dic:
                new_dict[elt] = new_dict[elt]+1
        l = list(new_dict.values())
        xs.append(l)
#    ys = []
#    for i in lst:
#        ys.append([truthful])
    return xs, [truthful]*(len(lst))

def create_nb_input_bigram(lst, dic, dic2, dic3, truthful):
    xs = []
    for sublist in lst:
        new_dict = dic.fromkeys(dic, 0)
        for elt in sublist:
            if elt in dic:
                new_dict[elt] = new_dict[elt]+1
        l = list(new_dict.values())

        new_dict2 = dic2.fromkeys(dic2, 0)
        for elt_index in range(len(sublist)):
            if elt_index == len(sublist) - 1:
                break
            else:
                curr_tup = (sublist[elt_index], sublist[elt_index+1])
                if curr_tup in dic2:
                    new_dict2[curr_tup] = new_dict2[curr_tup]+1
        l = l + list(new_dict2.values())

        new_dict3 = dic3.fromkeys(dic3, 0)
        for elt_index in range(len(sublist)):
            if elt_index == len(sublist) - 2:
                break
            else:
                curr_tup = (sublist[elt_index], sublist[elt_index+1], sublist[elt_index+2])
                if curr_tup in dic3:
                    new_dict3[curr_tup] = new_dict3[curr_tup]+1
        l = l + list(new_dict3.values())
        xs.append(l)
#    ys = []
#    for i in lst:
#        ys.append([truthful])
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
        fake_pp = perplexity(helper_unigram(review, unigram_dict_fake, smoothing, k))
        real_pp = perplexity(helper_unigram(review, unigram_dict_real, smoothing, k))
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
        fake_pp = perplexity(helper_bigram(review, bigram_dict_fake, udict_fake,smoothing,k))
        real_pp = perplexity(helper_bigram(review, bigram_dict_real, udict_real,smoothing,k))
        if fake_pp < real_pp:
            pred_lst.append((review_index, 1))
        else:
            pred_lst.append((review_index, 0))
    return pred_lst


# returns the perplexity of the whole corpus
def perplexity_all(validation_word_lst, bigram_dict, unigram_dict):
    flat_lst = []
    for sub_lst in validation_word_lst:
        for item in sub_lst:
            flat_lst.append(item)
    probs_lst = helper_bigram(flat_lst, bigram_dict, unigram_dict, smoothing=True, k=0.1)
    return perplexity(probs_lst)


# returns perplexity of one review given list of probabilities
def perplexity(probabilities):
    summation = 0
    for p in probabilities:
        summation += -1*math.log(p)
    return math.exp(1/len(probabilities)*summation)

# Returns the probs of a test review
# if smoothing is True, probability will be calculated with add-k smoothing
def helper_bigram(review_str, bigram_dict, unigram_dict, smoothing = False, k = 1):
    probs = []
    total_count = create_total_count(bigram_dict)
    for w_index in range(0, len(review_str)-1):
        curr_tup = (review_str[w_index], review_str[w_index + 1])
        top_num = bigram_dict.get(curr_tup, 0)
        bottom_number = unigram_dict.get(review_str[w_index], 0)
        if smoothing:
            probs.append(float(top_num + k)/(bottom_number + k*len(unigram_dict)))
        else:
            probs.append(float(top_num)/bottom_number)
    return probs


def helper_unigram(review_str, unigram_dict, smoothing = False, k = 1):
    probs = []
    total_count = create_total_count(unigram_dict)
    for w in review_str:
        top_num = unigram_dict.get(w, 0)
        if top_num == 0:
            top_num = unigram_dict["<unk>"]
            #top_num = k
        if smoothing:
            probs.append(float(top_num + k)/(total_count + k*len(unigram_dict)))
        else:
            probs.append(float(top_num)/total_count)
    return (probs)

if __name__ == "__main__":
    main()
