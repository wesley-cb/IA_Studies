import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
from nltk.corpus import words as nltk_words
import nltk

# Make sure to download the nltk words corpus if you haven't already
nltk.download('words')

# List of words (from NLTK corpus)
words = [word.lower() for word in nltk_words.words() if len(word) <= 8]  # Limiting words to a maximum of 8 letters

# Reduce the number of words to 500
sampled_words = random.sample(words, 500)

# Function to count the number of vowels in a word
def count_vowels(word):
    return sum(1 for letter in word if letter in 'aeiouáéíóúã')

# Generate the dataset based on the number of letters and vowels
data = []
for word in sampled_words:
    num_letters = len(word)
    num_vowels = count_vowels(word)
    data.append({
        'word': word,
        'num_letters': num_letters,
        'num_vowels': num_vowels
    })

# Dataset for training
class WordGuessDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = [entry['word'] for entry in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_data = self.data[idx]
        # Input: number of letters and number of vowels
        x = torch.tensor([word_data['num_letters'], word_data['num_vowels']], dtype=torch.float32)
        y = self.labels.index(word_data['word'])  # Output: index of the word
        return x, y


print(f"Initializing the dataset with {len(data)} words...")
dataset = WordGuessDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("Dataset initialized.")

# Neural network model
class WordGuessNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, output_size),  # Output size = number of words
        )

    def forward(self, x):
        return self.layers(x)

# Function to guess the word
def guess_word(model, num_letters, num_vowels):
    test_input = torch.tensor([num_letters, num_vowels], dtype=torch.float32).unsqueeze(
        0)  # Input: number of letters and vowels
    output = model(test_input)
    predicted_index = output.argmax(dim=1).item()  # Get the index of the predicted word
    return dataset.labels[predicted_index]

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 2  # Input: number of letters and number of vowels
output_size = len(dataset.labels)  # Total number of words
model = WordGuessNetwork(input_size, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.007)

# Training function
def train(model, dataloader, loss_function, optimizer):
    model.train()
    cumloss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        pred = model(x)
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumloss += loss.item()
    return cumloss / len(dataloader)

i = 0
correct_word = random.choice(sampled_words)  # Word to be tested
num_letters_correct = len(correct_word)
num_vowels_correct = count_vowels(correct_word)
print(correct_word)
print(f"Using device: {device}")
print(torch.cuda.is_available())

while (i < 5):
    print("\n///////////////////////////////////////////////////////////////////")
    i = i + 1
    epochs = 1000

    for t in range(epochs):
        # Training
        train_loss = train(model, dataloader, loss_function, optimizer)
        random_word = random.choice(sampled_words)
        num_letters_random = len(random_word)
        num_vowels_random = count_vowels(random_word)
        predicted_word = guess_word(model, num_letters_random, num_vowels_random)

        if t % 100 == 0:
            print(f"Epoch {t}, Loss: {train_loss:.4f}, Predicted word: {predicted_word}")

        # If the correct word is guessed, stop training
        if predicted_word == correct_word:
            print(f"Guessed correctly! The word is {correct_word}")
            break

    # Final test after training
    print("\nFinal result after training:")
    for attempt in range(epochs):
        predicted_word = guess_word(model, num_letters_correct, num_vowels_correct)
        print(f"Attempt {attempt + 1}: Predicted word - {predicted_word}, Correct word: {correct_word}")

        if predicted_word == correct_word:
            print(f"Guessed correctly! The word is {correct_word}")
            break
    else:
        print(f"Failed to guess the word. The correct word was: {correct_word}")
