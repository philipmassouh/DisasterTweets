# Philip Massouh
import csv

def read_training(name):
    train_positive, train_negative = [], []
    with open(name, 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            if row[4] == "1":
                train_positive.append(row[3])
            else:
                train_negative.append(row[3])
    return train_positive, train_negative

def read_testing(name):
    test_unknown = []
    with open(name, 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile):
            test_unknown.append((row[0],row[3]))
    return test_unknown

# Read in training and testing data
train_positive, train_negative = read_training("data/train.csv")
test_unknown = read_testing("data/test.csv")

# Calculate simple probabilities
total = len(train_positive) + len(train_negative)
prob_pos = len(train_positive) / total
prob_neg = len(train_negative) / total  

# remove all symbols, empty strings and turn into lowercase
import re
def clean_sentence(string_argument):
    return list(filter(None, re.sub(r'[,\.!?]',' ', 
        re.sub(r'[^\w]', ' ', string_argument)).lower().split(' ')))

# create and populate dictionaries
main_dict, pos_dict, neg_dict = {}, {}, {}

def populate_dictionary(data, dictionary):
    for sentence in data:
        words = clean_sentence(sentence)
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary.update({word: 1})

populate_dictionary(train_positive, main_dict)
populate_dictionary(train_negative, main_dict)
populate_dictionary(train_positive, pos_dict)
populate_dictionary(train_negative, neg_dict)

# find probabilities
def find_probability(word_list, prob, main_dic, t_dic, alpha):
    answer = prob
    for word in word_list:
        numerator = alpha
        if word in t_dic:
            numerator += t_dic[word]
        denominator = sum(t_dic.values()) + alpha*len(main_dict)
        answer = answer * (float(numerator) / float(denominator))
    return answer

# make predictions
def test(test_list, alpha, sentiments):
    predictions = []
    for sentence in test_list:
        words = clean_sentence(sentence[1])
        probabilities = [
            find_probability(words, prob_pos, main_dict, pos_dict, alpha),
            find_probability(words, prob_neg, main_dict, neg_dict, alpha)
        ]
        predicition = sentiments[probabilities.index(max(probabilities))]
        if predicition == 1:
            predictions.append((sentence[0], "1"))
        else:
            predictions.append((sentence[0], "0"))
    return predictions

# run predictions with a range of alphas
sentiments = [1, 0]
predictions = test(test_unknown, 0.2, sentiments)

f = open('results.csv', 'w+', encoding='utf-8')
for prediction in predictions:
    f.write(f"{prediction[0]}, {prediction[1]} \n")

