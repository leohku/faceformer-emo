import pickle

# Open the pickle file for reading
with open('/home/leoho/repos/FaceFormer/BIWI/templates.pkl', 'rb') as f:
    # Load the contents of the file
    data = pickle.load(f)

# Do something with the loaded data
print(data)