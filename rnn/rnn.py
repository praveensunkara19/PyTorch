import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import All_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        
        # Concatenate along dim=0 (vertically)
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

#loading the data
category_lines, all_categories = load_data()
n_categories = len(category_lines)



n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# one single character
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(next_hidden.size())

# whole sequence
input_tensor = line_to_tensor('Almeida')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.size())
# print(next_hidden.size())


# output layer
def category_form_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

print(category_form_output(output))

#Training phase
criterion = nn.NLLLoss() #negative likelihood loss
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    return output, loss.item()


# Loss plots
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0:
        guess = category_form_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f'{i} {i/n_iters * 100} {loss:.4f} {line} / {guess} {correct}')


plt.figure()
plt.plot(all_losses)
plt.show()

#predicgtion 

def predict(input_line):
    print(f'\n>{input_line}')
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_form_output(output)
        print(guess)

while True:
    sentance = input('Input:')
    if sentance == 'quit':
        break
    
    predict(sentance)





