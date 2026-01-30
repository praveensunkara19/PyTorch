
import io
import os
import unicodedata
import string
import glob


import torch
# import torchvision
import random

# Cleaning the data


#alphabet small + capital letters + " .,;"
All_LETTERS = string.ascii_letters + ".,;'"
N_LETTERS = len(All_LETTERS)  #a,b,c,e ...... X,Y,Z,.,,;,'


#Turn unicode strings to plain ASCII

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in All_LETTERS
    )

def load_data():
    #Build the category dictionary, a list of names per langage
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    
    #Read the Files and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('..\\data\\names\\*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        # print(f'getting the nemes form{filename}')

        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

"""
To represert a single letter, we use a 'one-hot vector' of 
size <1xn_letter>, A one-hot vector is filling with 0s
except for 1 at index of the current letter, eg. 'b' = <0 1 0 0 0 ..>
"""
# Find letter index form all_letters, eg 'a' = 0
def letter_to_index(letter):
    return All_LETTERS.find(letter)

# Turn a line into a <line_length x 1 x n_letter>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

# just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

#Trun a line inot a <line_length x 1 x n_letters>, 
#or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1

    return tensor

def random_training_example(category_lines, all_categories):

    def random_choice(a):
        random_idx = random.randint(0, len(a)-1)
        return a[random_idx]
    
    category = random_choice(all_categories) 
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

if __name__ == '__main__':
    print(All_LETTERS)
    print(unicode_to_ascii('S`lurak`sha')) # gives Sluraksha

    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])
    print(all_categories[0])

    # print(letter_to_tensor('J')) #[1, 56]
    # print(letter_to_tensor('J').shape) #[1, 56]
    # print(line_to_tensor('Jones').size()) #[5,1,57]
