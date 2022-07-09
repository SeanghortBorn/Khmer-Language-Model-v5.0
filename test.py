testing_words = ['apple', 'Python', 'C++', 'JavaScript', 'p', 'r', 'o', 'b', 'e', 'p','r','o','g','r','a','m','i','z']
i = 0
b_sz = 5
while i + b_sz <= len(testing_words):
    y = testing_words[i:i + b_sz] # input testing words
    print(f'y: {y}')
    x = []
    for _ in y:
        x.append(_)
    print(x)