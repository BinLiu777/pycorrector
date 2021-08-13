import pickle
with open("default.pkl","rb") as f:
    data = pickle.load(f)
    lines = data.split('\n')
with open('pku_dict.txt', 'w') as f:
    for line in lines:
        cuts = line.split()
        for c in cuts:
            f.write(c + ' ' + '1' + '\n')

